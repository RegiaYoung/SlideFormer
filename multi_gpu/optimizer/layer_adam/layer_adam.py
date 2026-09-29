# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
import glob
from typing import Optional, List, Dict, Any, Tuple, Iterable, DefaultDict
import torch
from .builder import CPUAdamLoader
from collections import defaultdict, deque, OrderedDict

class LayerAdam:
    """
    Layer-wise Adam optimizer for large models.
    所有层使用同一个优化器实例，但可以单独对某一层执行更新
    """
    def __init__(
            self,
            lr=1e-3,
            bias_correction=True,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            adamw_mode=True,
            fp32_optimizer_state=True,
            num_layer: int = 0,
            nvme_offload_fraction: float = 0.0, # 卸载比例
            offload_dir: str='/NVME1',
            prefetch: bool = True, # 预取层数
        ):
        """创建管理所有层参数的全局优化器，初始时不包含任何参数
        Args:
            lr: 学习率
            bias_correction: 是否应用偏差校正
            betas: Adam的beta参数
            eps: 数值稳定性参数
            weight_decay: 权重衰减系数
            adamw_mode: 是否使用AdamW模式
            fp32_optimizer_state: 是否使用FP32优化器状态
            num_layer: 总层数
            nvme_offload_fraction: 每层中要卸载的参数比例
            offload_dir: NVMe卸载目录
            prefetch: 是否开启预取
        """
        # 优化器参数
        self.defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adamw_mode=adamw_mode
        )

        # 初始化参数组列表
        self.param_groups = []

        # 加载CPU Adam内核
        self.cpu_adam = CPUAdamLoader().load()

        # 创建优化器实例 - 使用唯一ID标识此优化器
        self.optimizer_id = 0  # 使用单个优化器ID即可
        self.cpu_adam.create_adam(
            self.optimizer_id,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            adamw_mode,
            True  # 启用日志输出
        )

        self.fp32_optimizer_state = fp32_optimizer_state
        self.num_layer = num_layer

        # 参数状态字典
        self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)

        # NVMe卸载相关配置
        assert 0.0 <= nvme_offload_fraction <= 1.0
        self.nvme_offload_fraction = nvme_offload_fraction
        self.prefetch = prefetch
        self.pin_memory = nvme_offload_fraction > 0.0

        # 层的状态张量
        self.exp_avg_flat = OrderedDict()
        self.exp_avg_sq_flat = OrderedDict()

        if self.nvme_offload_fraction > 0.0:
            try:
                from tensornvme import DiskOffloader
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Please install tensornvme to use NVMeOptimizer")

            # 创建NVME卸载器
            assert offload_dir is not None, "offload_dir cannot be None"
            self.offload_dir = offload_dir
            self.offloader = DiskOffloader(self.offload_dir, 16, 'uring')
            self.offload_numel = OrderedDict()  # Layer卸载

        else:
            self.offload_dir = None
            self.offloader = None

    def __del__(self):
        """清理资源"""
        # 销毁CPU Adam优化器
        if hasattr(self, 'cpu_adam') and hasattr(self, 'optimizer_id'):
            try:
                self.cpu_adam.destroy_adam(self.optimizer_id)
            except:
                pass

        # 清理NVMe卸载文件
        if hasattr(self, 'offloader'):
            del self.offloader
            if self.offload_dir and os.path.exists(self.offload_dir):
                try:
                    for file in glob.glob(os.path.join(self.offload_dir, "offload-*")):
                        os.remove(file)
                except OSError:
                    pass

    def _get_layer_numel(self, layer_idx: int) -> int:
        """获取层的参数数量"""
        numel = 0
        for p in self.param_groups[layer_idx]['params']:
            numel += p.numel()
        return numel

    def add_layer_params(self, layer_idx: int, params: Iterable[torch.nn.Parameter]) -> int:
        """将层的参数添加到优化器

        Args:
            layer_idx: 层的索引
            params: 层的参数列表
        """
        # 为这个层创建参数组
        params = list(params)  # 确保是列表，因为可能是generator
        if not params:
            return -1  # 没有参数可添加

        # 创建参数组
        param_group = {
            'params': params,
            'layer_idx': layer_idx,
            'step': 0,
            **self.defaults
        }

        # 添加参数组到优化器
        self.param_groups.append(param_group)

        # 计算层的总参数数量
        layer_numel = sum(p.numel() for p in params)
        # 创建两个大 tensor 用于存储优化器状态

        state_type = torch.float32 if self.fp32_optimizer_state else params[0].dtype

        self.exp_avg_flat[layer_idx] = torch.zeros(
            layer_numel,
            dtype=state_type,
            device=torch.device('cpu')
        )

        self.exp_avg_sq_flat[layer_idx] = torch.zeros(
            layer_numel,
            dtype=state_type,
            device=torch.device('cpu')
        )

        # 创建视图映射
        offset = 0
        for p in params:
            size = p.numel()
            shape = p.shape

            # 初始化参数状态
            if p not in self.state:
                self.state[p] = {}

            # 创建状态视图
            self.state[p]['exp_avg'] = self.exp_avg_flat[layer_idx][offset:offset+size].view(shape)
            self.state[p]['exp_avg_sq'] = self.exp_avg_sq_flat[layer_idx][offset:offset+size].view(shape)

            offset += size

        if self.offloader is not None:

            if layer_idx != self.num_layer - 1:
                self._offload_layer(layer_idx)
            else:
                self.offloader.sync_write_events()

        return layer_idx

    def _load_layer(self, layer_idx: int):
        """执行预取操作"""
        if self.offloader is None:
            return

        if layer_idx < 0:
            layer_idx = self.num_layer + layer_idx

        assert layer_idx in self.exp_avg_flat

        self.offloader.async_read(self.exp_avg_flat[layer_idx])
        if self.nvme_offload_fraction > 0.5:
            self.offloader.async_read(self.exp_avg_sq_flat[layer_idx])

    def _offload_layer(self, layer_idx: int):
        """执行卸载操作"""
        if self.offloader is None:
            return

        assert layer_idx in self.exp_avg_flat

        self.offloader.async_write(self.exp_avg_flat[layer_idx])
        if self.nvme_offload_fraction > 0.5:
            self.offloader.async_write(self.exp_avg_sq_flat[layer_idx])


    def _pre_step(self, layer_idx: int):
        """预取下一层的状态"""
        if self.offloader is None:
            return

        if self.prefetch:
                # 同步上一批的读写事件
                self.offloader.sync_read_events()
                self._load_layer(layer_idx - 1)
        else:
            # 加载当前层的状态
            self._load_layer(layer_idx)

    def _post_step(self, layer_idx: int):
        """执行后处理操作"""
        if self.offloader is None:
            return

        if self.prefetch:
                self.offloader.sync_write_events()

        self._offload_layer(layer_idx)


    def torch_adam_update(self, data, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps,
                        weight_decay, bias_correction1, bias_correction2):
        """PyTorch实现的Adam更新逻辑"""
        grad = grad.to(data.dtype)

        if weight_decay != 0:
            if self.defaults['adamw_mode']:
                data.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(data, alpha=weight_decay)

        # 更新动量和方差
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1

        data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, layer_idx: int = None, closure=None):
        """执行优化步骤，仅更新指定层的参数

        Args:
            layer_idx: 要更新的层索引，如果为None则更新所有层
            closure: 评估模型的闭包函数
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 确定要更新的参数
        if layer_idx is not None:
            # 只更新特定层
            param_groups_to_update = [self.param_groups[layer_idx]]
        else:
            # 更新所有层
            assert self.offloader is None, "开启NVME卸载时，只能按顺序更新特定层"
            param_groups_to_update = self.param_groups

        self._pre_step(layer_idx)

        # 开始更新
        for param_group in param_groups_to_update:
            beta1, beta2 = param_group['betas']
            param_group['step'] += 1

            for p in param_group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                target_device = p.device


                # CPU上的参数更新
                if target_device.type == "cpu":
                    assert p.data.numel() == p.grad.data.numel(), "参数和梯度应当具有相同大小"

                    # 使用新的CPU Adam API, 来自deepspeed, 支持fp32 fp16 bf16
                    self.cpu_adam.adam_update(
                        self.optimizer_id,
                        param_group['step'],
                        param_group['lr'],
                        beta1,
                        beta2,
                        param_group['eps'],
                        param_group['weight_decay'],
                        param_group['bias_correction'],
                        p.data,
                        p.grad.data,
                        state['exp_avg'],
                        state['exp_avg_sq']
                    )


                else:
                    raise RuntimeError(f"不支持的设备类型: {target_device.type}")


        self._post_step(layer_idx)

        return loss

    @torch.no_grad()
    def step_with_grad_views(self, layer_idx: int = None, grad_views: Dict[torch.nn.Parameter, torch.Tensor] = None, closure=None):
        """执行优化步骤，使用提供的梯度视图而不是param.grad

        Args:
            layer_idx: 要更新的层索引
            grad_views: 参数到梯度视图的映射字典
            closure: 评估模型的闭包函数
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 确定要更新的参数
        if layer_idx is not None:
            # 只更新特定层
            param_groups_to_update = [self.param_groups[layer_idx]]
        else:
            # 更新所有层
            assert self.offloader is None, "开启NVME卸载时，只能按顺序更新特定层"
            param_groups_to_update = self.param_groups

        self._pre_step(layer_idx)

        # 开始更新
        for param_group in param_groups_to_update:
            beta1, beta2 = param_group['betas']
            param_group['step'] += 1

            for p in param_group['params']:
                # 使用提供的梯度视图而不是p.grad
                grad = grad_views.get(p) if grad_views else None
                if grad is None:
                    print(f"Warning: 参数 {p} 没有提供梯度视图，跳过更新")
                    continue

                state = self.state[p]
                target_device = p.device


                # CPU上的参数更新
                if target_device.type == "cpu":
                    assert p.data.numel() == grad.numel(), "参数和梯度应当具有相同大小"
                    # 使用CPU Adam API，直接传递梯度视图而不是p.grad.data
                    self.cpu_adam.adam_update(
                        self.optimizer_id,
                        param_group['step'],
                        param_group['lr'],
                        beta1,
                        beta2,
                        param_group['eps'],
                        param_group['weight_decay'],
                        param_group['bias_correction'],
                        p.data,
                        grad,  # 直接使用梯度视图
                        state['exp_avg'],
                        state['exp_avg_sq']
                    )
                else:
                    raise RuntimeError(f"不支持的设备类型: {target_device.type}")

        self._post_step(layer_idx)

        return loss

    @torch.no_grad()
    def begin_chunk_step(self, layer_idx: int):
        """Begin a chunked step: increment step counter and prefetch NVMe states.
        Call once before the first step_chunk for this layer.
        Returns the param_group for passing to step_chunk.
        """
        param_group = self.param_groups[layer_idx]
        param_group['step'] += 1
        self._pre_step(layer_idx)
        return param_group

    @torch.no_grad()
    def step_chunk(self, param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat, param_group):
        """Process one chunk of flat contiguous parameter/gradient tensors.
        All four tensors must be sliced to the same [cs:ce] range by the caller.
        This allows the D2H thread to produce grad chunks while the update thread
        consumes and applies Adam to each chunk immediately.
        """
        self.cpu_adam.adam_update(
            self.optimizer_id,
            param_group['step'],
            param_group['lr'],
            param_group['betas'][0],
            param_group['betas'][1],
            param_group['eps'],
            param_group['weight_decay'],
            param_group['bias_correction'],
            param_flat,
            grad_flat,
            exp_avg_flat,
            exp_avg_sq_flat
        )

    @torch.no_grad()
    def end_chunk_step(self, layer_idx: int):
        """End a chunked step: offload NVMe optimizer states.
        Call once after all step_chunk calls for this layer.
        """
        self._post_step(layer_idx)

    def update_learning_rate(self, new_lr):
        """更新优化器的学习率

        Args:
            new_lr: 新的学习率值
        """
        # 更新默认学习率
        self.defaults['lr'] = new_lr


        # 更新所有参数组的学习率
        for param_group in self.param_groups:
            param_group['lr'] = new_lr
