# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformer_layer import (
    build_decoder_attention_mask,
    split_transformer_model,
)
from utils.model_compat import (parameter_sizes, prepare_layer_inputs, router_aux_loss,
                                save_text_weights, validate_model)
from utils.log_mem import log_memory_stats
from utils.module_utils import _print_module_structure
from utils.timer import format_timing_label
import concurrent.futures
import time
import os
from collections import defaultdict
import queue
from typing import Optional
from optimizer import LayerAdam
# from optimizer.hybrid_adam.layer_adam_nvme import LayerAdam
from sliding_checkpoint import init_ac_gpu_prefetch_pool
import torch.cuda.nvtx as nvtx

class SlideFormerOffloader(nn.Module):
    """使用滑动窗口管理transformer模型的CPU-GPU交换"""
    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        ac_offload_nvme: bool = False,
        offload_dir: str = '/RAID0',
        double_buffer: bool = False,
        enable_timing: bool = True,
        enable_memory_stats: bool = True,
        optimizer_kwargs: Optional[dict] = None,
        auto_backward_in_forward: bool = True,
    ):
        
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.double_buffer = double_buffer
        self.auto_backward_in_forward = auto_backward_in_forward
        self._embed_update_pending = False

        # 保存对基础模型的引用，以便后续保存
        self.base_model = model
        validate_model(model)
                
        # 添加参数检查
        # self.window_size = 1
        # if self.window_size < 1:
        #     raise ValueError("window_size must be >= 1")
        if not isinstance(device, torch.device):
            raise TypeError("device must be torch.device")
        if dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("dtype must be float16 or bfloat16")
        
        # 保存计时控制标志
        self.enable_timing = enable_timing
        # 保存内存统计控制标志
        self.enable_memory_stats = enable_memory_stats
        
        # 创建独立的线程池，分别用于参数更新、H2D传输和D2H传输
        self.update_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.h2d_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.d2h_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if double_buffer else self.h2d_executor
        
        # 修改GPU缓存队列大小策略
        gpu_cache_size = 2 if self.double_buffer else 1
        self.decoder = model.get_decoder()
        self.decoder_layers = self.decoder.layers
        self.gpu_unit_size = min(1 + gpu_cache_size, len(self.decoder_layers))
        self.gpu_cache_queue = queue.Queue(maxsize=self.gpu_unit_size)
        
        # 计算模型所有层中最大参数数量
        self.max_param_size, self.total_param = self._calculate_max_param_size(model)

        # cpu_grad缓存张量
        if self.double_buffer:
            self.cpu_grad_tensor = [
                torch.empty(self.max_param_size, dtype=self.dtype, device=torch.device("cpu"), pin_memory=True),
                torch.empty(self.max_param_size, dtype=self.dtype, device=torch.device("cpu"), pin_memory=True)
            ]
        else:
            self.cpu_grad_tensor = [torch.empty(self.max_param_size, dtype=self.dtype, device=torch.device("cpu"), pin_memory=True)]
        
        self.bf16_param_tensor = torch.empty(self.max_param_size, dtype=self.dtype, device=torch.device("cpu"), pin_memory=True)  # torch.float16
        
        # 初始化GPU缓存
        self._init_gpu_cache()
        
        # ---- tie detection ----
        self.tied_param = None
        self.tied_grad_accum = None
        self.is_tied = (model.get_input_embeddings().weight is model.get_output_embeddings().weight)
        if self.is_tied:
            print("[INFO] Detected tied weights between input and output embeddings.")
            self.tied_param = model.get_input_embeddings().weight
            self.tied_grad_accum = torch.zeros(self.tied_param.numel(), dtype=self.dtype, device=torch.device("cpu"), pin_memory=True)
        
        # 创建共享的LayerAdam优化器
        default_optimizer_kwargs = {
            "lr": 1e-5,
            "bias_correction": True,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "fp32_optimizer_state": True,
            "num_layer": len(self.decoder_layers) + 2,  # 包括嵌入层和输出层
            "nvme_offload_fraction": 0.0, # 0为关闭，目前是0/0.5/1三档
            "offload_dir": offload_dir,
            "prefetch": True
        }
        optimizer_kwargs = optimizer_kwargs or default_optimizer_kwargs
        self.layer_optimizer = LayerAdam(**optimizer_kwargs)

        # 保存模型配置
        self.config = self.decoder.config
        self.ac_offload_nvme = ac_offload_nvme
        self.offload_dir = offload_dir
        
        # 分解模型组件并传入线程池和优化器
        components = split_transformer_model(
            model, device, dtype, 
            update_executor=self.update_executor,
            h2d_executor=self.h2d_executor,
            d2h_executor=self.d2h_executor,
            gpu_cache_queue=self.gpu_cache_queue,
            cpu_grad_tensor=self.cpu_grad_tensor,
            bf16_param_tensor=self.bf16_param_tensor,
            tied_param=self.tied_param,
            tied_grad_accum=self.tied_grad_accum,
            enable_timing=self.enable_timing,
            layer_optimizer=self.layer_optimizer,
            double_buffer=self.double_buffer,
        )
        
        # 模型Layer
        self.transformer_layers = components['transformer_layers']
        self.rotary_emb = components['rotary_emb']
        
        # 快捷访问
        self.embed_layer = self.transformer_layers[0]  # idx = -1
        self.output_layer = self.transformer_layers[-1]  # idx = num_layers

        # 预估内存占用：复用已有统计量，保持公式简洁
        self.dtype_size_bytes = torch.tensor([], dtype=self.dtype).element_size()
        self.managed_param_numel = self.total_param
        self.stage_buffer_bytes = self.max_param_size * self.dtype_size_bytes

        self.param_size_bytes = self.managed_param_numel * 4
        self.grad_size_bytes = len(self.cpu_grad_tensor) * self.stage_buffer_bytes
        self.convert_size_bytes = self.stage_buffer_bytes
        self.tied_grad_size_bytes = (
            self.tied_param.numel() * self.dtype_size_bytes if self.is_tied else 0
        )

        optimizer_state_elem_size = (
            4 if getattr(self.layer_optimizer, "fp32_optimizer_state", True)
            else self.dtype_size_bytes
        )
        self.opt_state_bytes = self.managed_param_numel * optimizer_state_elem_size * 2
        nvme_offload_fraction = getattr(self.layer_optimizer, "nvme_offload_fraction", 0.0)
        self.opt_offload_bytes = self.opt_state_bytes * nvme_offload_fraction

        self._activation_layout_signature = None
        
        # Loss计算
        # self.loss_function = model.loss_function

        # 注册每一层的钩子
        self._register_layer_hooks()

        self.layer_tensors = None
        self.layer_tensor_files = None

        # 打印模型结构
        print("\nModel Structure:")
        _print_module_structure(self.embed_layer)
        _print_module_structure(self.transformer_layers[1])
        _print_module_structure(self.output_layer)

    def __del__(self):
        """确保线程池正确关闭并打印统计信息"""
        # if hasattr(self, 'transformer_layers'):
        #     print("\nFinal Layer Statistics:")
        #     for layer in self.transformer_layers:
        #         if layer.timer:
        #             layer.timer.print_stats()
        if hasattr(self, 'update_executor'):
            self.update_executor.shutdown()
        if hasattr(self, 'h2d_executor'):
            self.h2d_executor.shutdown()
        if hasattr(self, 'd2h_executor'):
            self.d2h_executor.shutdown()

    def _calculate_max_param_size(self, model):
        return parameter_sizes(model)

    def _init_gpu_cache(self):
        """初始化GPU缓存单元"""
        if self.gpu_unit_size < 1:
            raise ValueError("gpu_unit_size must be >= 1")
        
        for _ in range(self.gpu_unit_size):
            cache_unit = {
                "param": torch.empty(self.max_param_size, dtype=self.dtype, device=self.device),
                "grad": torch.empty(self.max_param_size, dtype=self.dtype, device=self.device),
                # Reuse of a GPU cache unit must wait until the previous D2H
                # stream work that reads from it has finished.
                "reuse_ready": torch.cuda.Event(),
            }
            cache_unit["reuse_ready"].record(torch.cuda.current_stream(self.device))
            self.gpu_cache_queue.put(cache_unit)
        
    def _register_layer_hooks(self):
        """为每一层注册管理钩子"""
        num_layers = len(self.transformer_layers)
        
        def get_pre_forward(idx):
            def hook(module, input):
                """前向传播前预加载窗口中的下一层"""
                next_layer_idx = idx + 1
                if next_layer_idx == num_layers-1:
                    self.transformer_layers[next_layer_idx].to_device_async(is_bwd=True)
                elif next_layer_idx < num_layers:
                    self.transformer_layers[next_layer_idx].to_device_async()
                module.wait_for_h2d()
                return input
            return hook
            
        def get_post_forward(idx):
            def hook(module, input, output):
                """前向传播后：只有对于不在最后窗口的层才需要卸载"""
                next_needed = idx + 1
                if next_needed < num_layers:  # 不是最后窗口的层才需要卸载
                    module.to_offload_async()
                return output
            return hook
            
        def get_pre_backward(idx):
            def hook(module, grad_output):
                """反向传播前预加载前一层"""
                # 预加载前一层（如果存在）
                prev_layer_idx = idx - 1
                if prev_layer_idx >= 0:
                    if prev_layer_idx == 0 and self.is_tied:
                        # Tied embedding backward must wait until output D2H has landed.
                        self.output_layer.wait_for_d2h()
                        self.tied_param.grad = None
                    self.transformer_layers[prev_layer_idx].to_device_async(is_bwd=True)
                # module.h2d_ready.synchronize()  # 等待数据传输完成
                module.wait_for_h2d()
                # print(f"Pre-Backward: {idx}")
                if self.enable_timing:   
                    module._backward_start_time = time.perf_counter()  # 开始计时
                nvtx.range_push(f"Backward_Compute_Layer_{idx}")
            return hook
            
        def get_post_backward(idx):
            def hook(module, grad_input, grad_output):
                """反向传播后更新并卸载所有层"""
                nvtx.range_pop()
                if self.enable_timing:
                    start = getattr(module, '_backward_start_time', None)
                    if start is not None and module.timer:
                        duration = time.perf_counter() - start
                        # print(f"Backward time for layer {idx}: {duration:.2f}s")
                        module.timer.record_time("backward", duration)
                        delattr(module, '_backward_start_time')
                        
                module.compute_ready_bw.record(torch.cuda.current_stream())
                if idx == 0:
                    # Integer embedding inputs have no gradients: this hook fires
                    # before parameter accumulation. Drain it after backward().
                    self._embed_update_pending = True
                    return
                # print(f"Post-Backward: {idx}")
                # Embed的call func导致hook不准确，导致第一层在反向过程中提前释放，手动后向
                # if idx == 0:
                #     return None
                
                prev_update = None
                
                check_offset = 2 if self.double_buffer else 1
                check_idx = idx + check_offset
                
                if check_idx < num_layers:
                     prev_update = self.transformer_layers[check_idx].update_finished
                elif self.double_buffer and check_idx == num_layers: 
                     # Double Buffering Special Case:
                     pass

                module.to_offload_async(is_bwd=True, prev_update=prev_update)
                module.update_params()
            return hook
        
        # 为所有层注册钩子
        for idx, layer in enumerate(self.transformer_layers):
            layer.register_forward_pre_hook(get_pre_forward(idx))
            layer.register_forward_hook(get_post_forward(idx))
            layer.register_full_backward_pre_hook(get_pre_backward(idx))
            layer.register_full_backward_hook(get_post_backward(idx))


    def _launch_pending_embed_update(self) -> None:
        if not self._embed_update_pending:
            return

        prev_update = None
        check_idx = 2 if self.double_buffer else 1
        if check_idx < len(self.transformer_layers):
            prev_update = self.transformer_layers[check_idx].update_finished

        self.embed_layer.compute_ready_bw.record(torch.cuda.current_stream(self.device))
        self.embed_layer.to_offload_async(is_bwd=True, prev_update=prev_update)
        self.embed_layer.update_params()
        self._embed_update_pending = False


    def wait_for_completion(self) -> None:
        """Drain outstanding H2D, D2H, and update work for the current step."""
        self._launch_pending_embed_update()
        for layer in self.transformer_layers:
            layer.wait_for_h2d()
        for layer in self.transformer_layers:
            layer.wait_for_d2h()
        for layer in self.transformer_layers:
            layer.wait_for_update()
        torch.cuda.synchronize(self.device)

    def print_layer_stats(self, layer_idx=None, show_history=False):
        """打印层统计信息
        Args:
            layer_idx: 可选，指定层的索引。如果为None，打印所有层的统计
            show_history: True则显示历史平均值，False则显示当前step的统计
        """
        if not self.enable_timing:
            return
            
        if layer_idx is not None:
            # 打印特定层的统计信息
            if 0 <= layer_idx < len(self.transformer_layers):
                layer = self.transformer_layers[layer_idx]
                if layer.timer:
                    layer.timer.print_stats(show_history)
            return
            
        # 计算所有层的统计
        all_stats = defaultdict(list)
        if show_history:
            # 收集所有层的历史数据
            for layer in self.transformer_layers:
                if layer.timer:
                    layer.timer.flush_cuda_spans(force=True)
                    for op, times in layer.timer.timings.items():
                        all_stats[op].extend(times)
            print("\nHistorical Average Layer Statistics:")
            for op, times in all_stats.items():
                avg = sum(times) / len(times)
                print(f"  {format_timing_label(op)}: {avg*1000:.2f}ms (avg over {len(times)} calls)")
        else:
            # 收集当前step的数据
            for layer in self.transformer_layers:
                if layer.timer:
                    layer.timer.flush_cuda_spans(force=True)
                    for op, duration in layer.timer.current_step.items():
                        all_stats[op].append(duration)
            print("\nCurrent Step Layer Statistics:")
            for op, durations in all_stats.items():
                avg = sum(durations) / len(durations)
                print(f"  {format_timing_label(op)}: {avg*1000:.2f}ms (avg across {len(durations)} layers)")
    
    def _print_memory_occupancy(self):
        """打印预估内存占用"""
        
        print(f"Model Type: {self.config.model_type}")
        summary = (
            f"Estimated Memory Occupancy: {self.param_size_bytes / (1024 ** 3):.2f} GB (params fp32) + "
            f"{self.grad_size_bytes / (1024 ** 3):.2f} GB (grad buffers) + "
            f"{self.convert_size_bytes / (1024 ** 3):.2f} GB (convert buffer)"
        )
        if self.tied_grad_size_bytes > 0:
            summary += f" + {self.tied_grad_size_bytes / (1024 ** 3):.2f} GB (tied grad)"
        summary += (
            f" + {self.activation_bytes / (1024 ** 3):.2f} GB (activations) + "
            f"{self.opt_state_bytes / (1024 ** 3):.2f} GB (optimizer state)"
        )
        if self.opt_offload_bytes > 0:
            summary += f" - {self.opt_offload_bytes / (1024 ** 3):.2f} GB (os offload)"
        if self.ac_offload_nvme:
            summary += f" - {self.activation_bytes / (1024 ** 3):.2f} GB (ac offload)"
        print(summary)

        estimated_memory = (
            self.param_size_bytes
            + self.grad_size_bytes
            + self.convert_size_bytes
            + self.tied_grad_size_bytes
            + self.activation_bytes
            + self.opt_state_bytes
            - self.opt_offload_bytes
        )
        if self.ac_offload_nvme:
            estimated_memory -= self.activation_bytes
        print(f"Total CPU Estimated Memory: {estimated_memory / (1024 ** 3):.2f} GB")

    def _allocate_activation_tensors_on_cpu(self, batch_size, seq_length, hidden_size):
        """
            根据实际输入尺寸预分配CPU tensors, 适应简化后的单一checkpoint设计
        """
        num_layers = len(self.transformer_layers)
        self.layer_tensors = []
        
        for _ in range(num_layers-2):  # 不包括嵌入层和输出层
            layer_cpu_tensors = [
                torch.empty(
                    (batch_size, seq_length, hidden_size),
                    dtype=self.dtype,
                    pin_memory=True,
                )
            ]
            self.layer_tensors.append(layer_cpu_tensors)
        
        # 更新每个transformer层的layer_tensors引用
        for i, layer in enumerate(self.transformer_layers):
            # 跳过嵌入层和输出层
            if 0 < i < len(self.transformer_layers) - 1:
                layer.layer_cpu_tensors = self.layer_tensors

        # 初始化 GPU activation prefetch buffer pool (所有层shape相同，用第一层做模板)
        init_ac_gpu_prefetch_pool(self.layer_tensors[0], self.device)
                
    def _allocate_activation_tensors_on_nvme(self, batch_size, seq_length, hidden_size):
        """
            根据实际输入尺寸预分配activation tensors文件路径
        """
        base_dir = self.offload_dir + '/activation'
        num_layers = len(self.transformer_layers)
        self.layer_tensor_files = []
        
        # 确保基础目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        for i in range(num_layers-2):  # 不包括嵌入层和输出层
            layer_dir = os.path.join(base_dir, f"layer_{i+1}")
            os.makedirs(layer_dir, exist_ok=True)
            hidden_path = os.path.join(layer_dir, f"hidden_state")
            self.layer_tensor_files.append([hidden_path])
                    
        # 更新每个transformer层的NVMe路径引用
        for i, layer in enumerate(self.transformer_layers):
            if 0 < i < len(self.transformer_layers) - 1:
                layer.layer_nvme_paths = self.layer_tensor_files
                layer.ac_offload_nvme = self.ac_offload_nvme
        
        print(f"NVMe activation files init, for {len(self.layer_tensor_files)} decoder layers")

    def _ensure_activation_storage(self, batch_size, seq_length, hidden_size):
        signature = (
            batch_size,
            seq_length,
            hidden_size,
            self.ac_offload_nvme,
        )
        if self._activation_layout_signature == signature:
            return

        self.layer_tensors = None
        self.layer_tensor_files = None

        if self.ac_offload_nvme:
            self._allocate_activation_tensors_on_nvme(
                batch_size,
                seq_length,
                hidden_size,
            )
        else:
            self._allocate_activation_tensors_on_cpu(
                batch_size,
                seq_length,
                hidden_size,
            )

        hidden_bytes = batch_size * seq_length * hidden_size * torch.tensor([], dtype=self.dtype).element_size()
        self.activation_bytes = (len(self.transformer_layers) - 2) * hidden_bytes
        self._activation_layout_signature = signature
        self._print_memory_occupancy()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        position_ids=None,
        return_dict=None,
    ):
        """模型的前向传播，保留 finetuning 常用的 Hugging Face 参数子集。

        注意：
        - `labels is not None` 时只执行一次输出层并返回 loss，不会额外重算 logits。
        - `auto_backward_in_forward=False` 仅表示 backward 由外部触发，不改变上述输出语义。
        """
        # 启用异常检测, 
        # torch.autograd.set_detect_anomaly(True, check_nan=True)

        return_dict = True if return_dict is None else return_dict
        if self._embed_update_pending:
            raise RuntimeError(
                "Previous backward still has a pending embedding update. "
                "Call wait_for_completion() immediately after loss.backward()."
            )

        self.transformer_layers[0].to_device_async()

        if self.enable_memory_stats:
            print("Begin:"+log_memory_stats())
            
        hidden_states = self.embed_layer(input_ids)

        layer_inputs = prepare_layer_inputs(
            self.decoder, hidden_states, attention_mask, position_ids,
        )
        router_outputs = []

        self._ensure_activation_storage(
            batch_size=hidden_states.shape[0],
            seq_length=hidden_states.shape[1],
            hidden_size=hidden_states.shape[2],
        )
            
        # torch.cuda.empty_cache() # (gpu: -500MB)
        
        # Forward
        for idx in range(1, len(self.transformer_layers)-1):
            result = self.transformer_layers[idx](hidden_states, **layer_inputs[idx - 1])
            if self.transformer_layers[idx].collect_router:
                hidden_states, routing = result
                router_outputs.append(routing)
            else:
                hidden_states = result

        # 4. 损失计算
        loss = None
        logits = None
        
        if labels is not None:
            
            # Norm + LM Head + CrossEntropyLoss
            loss = self.output_layer(hidden_states, labels=labels)
            auxiliary = router_aux_loss(self.base_model, router_outputs, attention_mask)
            if auxiliary is not None:
                loss = loss + self.base_model.router_aux_loss_coef * auxiliary
            
            # logits = logits = self.output_layer(hidden_states)
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
            
            if self.enable_memory_stats:
                print("Forward:" + log_memory_stats())
            
            # 兼容旧版训练脚本：默认在forward内触发backward
            if self.auto_backward_in_forward:
                loss.backward()
                self.wait_for_completion()

                if self.enable_memory_stats:
                    print("Backward:"+log_memory_stats()) # (第二次回来多了3G)

                # 显示当前step所有层的平均
                self.print_layer_stats(show_history=False)

        else:
            logits = self.output_layer(hidden_states, labels=None)
            
        if not return_dict:
            return tuple(value for value in (loss, logits) if value is not None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )
    
    def update_learning_rate(self, new_lr):
        """更新优化器的学习率"""
        self.layer_optimizer.update_learning_rate(new_lr)
    
    def save_pretrained(self, output_dir):
        """Save text weights while retaining the live FP32 master parameters."""
        self.wait_for_completion()
        save_text_weights(self.base_model, output_dir, self.dtype)
        return output_dir
