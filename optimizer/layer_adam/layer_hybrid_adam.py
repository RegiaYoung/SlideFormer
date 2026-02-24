import math
import os
import glob
from typing import Optional, List, Dict, Any, Tuple, Iterable, DefaultDict
import torch
from .builder import CPUAdamLoader, FusedAdamLoader
from collections import defaultdict, deque, OrderedDict
from .multi_tensor_apply import MultiTensorApply  # 修改为相对导入
# from tensornvme import DiskOffloader
# from torch.optim import Optimizer

multi_tensor_applier = MultiTensorApply(2048 * 32)

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
            weight_decay=weight_decay
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
        
        # 配置
        self.adamw_mode = adamw_mode
        self.fp32_optimizer_state = fp32_optimizer_state
        self.num_layer = num_layer
        
        # 参数状态字典
        self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        
        # NVMe卸载相关配置
        assert 0.0 <= nvme_offload_fraction <= 1.0
        self.nvme_offload_fraction = nvme_offload_fraction
        self.prefetch = prefetch

        # 层的状态张量
        self.exp_avg_flat = OrderedDict()
        self.exp_avg_sq_flat = OrderedDict()
        
        self.on_gpu = set()
        # 加载CUDA Adam内核
        try:
            self.cuda_adam = FusedAdamLoader().load()
            self.multi_tensor_adam = self.cuda_adam.multi_tensor_adam
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda:0')
            self.on_gpu = set()
        except Exception as e:
            raise AssertionError(f"Cuda Adam Load Error: {e}")
        
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
        
    def add_layer_params(self, layer_idx: int, params: Iterable[torch.nn.Parameter], device: torch.device) -> int:
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
            device=device
        )

        self.exp_avg_sq_flat[layer_idx] = torch.zeros(
            layer_numel,
            dtype=state_type,
            device=device
        )
        
        # 计算优化器状态大小并在设备上分配
        if device.type == 'cuda':
            # 可以添加GPU内存使用警告
            self.on_gpu.add(layer_idx)
            gpu_mem_required = layer_numel * 2 * (4 if state_type == torch.float32 else 2)  # 字节数
            print(f"注意：第 {layer_idx} 层将使用 {gpu_mem_required/1024/1024:.2f} MB GPU内存用于优化器状态")
        
        # 创建视图映射
        offset = 0
        for p in params:
            size = p.numel()
            shape = p.shape
            
            # 初始化参数状态
            if p not in self.state:
                self.state[p] = {'step': 0}
            
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

        if layer_idx in self.on_gpu:
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

        if layer_idx in self.on_gpu:
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
            if self.adamw_mode:
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
            g_l, p_l, m_l, v_l = [], [], [], []
            group_step = 0
            beta1, beta2 = param_group['betas']
            for p in param_group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                target_device = p.device
                
                # 更新步数
                state['step'] += 1
                
                # CPU上的参数更新
                if target_device.type == "cpu":
                    assert p.data.numel() == p.grad.data.numel(), "参数和梯度应当具有相同大小"

                    # 使用新的CPU Adam API, 来自deepspeed, 支持fp32 fp16 bf16
                    self.cpu_adam.adam_update(
                        self.optimizer_id,
                        state['step'],
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

                # # GPU上的参数更新
                elif target_device.type == "cuda":
                    assert state['exp_avg'].device.type == "cuda", "exp_avg应当在GPU上"
                    assert state['exp_avg_sq'].device.type == "cuda", "exp_avg_sq应当在GPU上"
                    
                    g_l.append(p.grad.data)
                    p_l.append(p.data)
                    m_l.append(state['exp_avg'])
                    v_l.append(state['exp_avg_sq'])
                    # bias_correction1 = 1 - beta1 ** state['step']
                    # bias_correction2 = 1 - beta2 ** state['step']
                    
                    # self.torch_adam_update(
                    #     p.data,
                    #     p.grad.data,
                    #     state['exp_avg'],
                    #     state['exp_avg_sq'],
                    #     param_group['lr'],
                    #     beta1,
                    #     beta2,
                    #     param_group['eps'],
                    #     param_group['weight_decay'],
                    #     bias_correction1,
                    #     bias_correction2
                    # )
                
                else:
                    raise RuntimeError(f"不支持的设备类型: {target_device.type}")
                
            # GPU上的参数更新
            if len(g_l) > 0:
                multi_tensor_applier(
                    self.multi_tensor_adam,
                    self._dummy_overflow_buf,
                    [g_l, p_l, m_l, v_l],
                    param_group['lr'],
                    beta1,
                    beta2,
                    param_group['eps'],
                    group_step,
                    1 if self.adamw_mode else 0,
                    1 if param_group['bias_correction'] else 0,
                    param_group['weight_decay']
                )

        # if accumulated_numel is not None:
        #     actual_offload_ratio = accumulated_numel / self.layer_numel[layer_idx]
        #     print(f"Layer {layer_idx}: Target offload ratio: {self.nvme_offload_fraction}, Actual: {actual_offload_ratio:.4f}")
        #     print(f"Offloaded {len(state_tensors)//2} parameters, {accumulated_numel} elements")
        # print(f"Layer {layer_idx} post step")
        
        self._post_step(layer_idx)
        
        return loss