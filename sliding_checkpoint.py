# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd.graph import saved_tensors_hooks
from collections import deque, OrderedDict
from typing import Any, Tuple, Optional, List, Dict

# 全局预取队列
fifo_prefetch_queue = deque()
cp_stream = torch.cuda.Stream()
write_events = OrderedDict()

# ---- Activation prefetch GPU buffer pool ----
# Round-robin pre-allocated GPU buffers avoid allocator work during backward.
_ac_gpu_buffers: Optional[List[List[torch.Tensor]]] = None
_ac_buffer_idx: int = 0


def init_ac_gpu_prefetch_pool(
    template_cpu_tensors: List[torch.Tensor],
    device: torch.device,
    pool_size: int = 2,
) -> None:
    """Pre-allocate GPU buffers matching the activation shapes.

    Args:
        template_cpu_tensors: One layer's list of CPU activation tensors
                              whose shapes / dtypes are used as templates.
        device: Target GPU device.
        pool_size: Number of buffer sets (2 is sufficient for the pipeline).
    """
    global _ac_gpu_buffers, _ac_buffer_idx
    _ac_gpu_buffers = [
        [torch.empty(t.shape, dtype=t.dtype, device=device) for t in template_cpu_tensors]
        for _ in range(pool_size)
    ]
    _ac_buffer_idx = 0
    print(f"[AC Pool] Initialized {pool_size} GPU prefetch buffer sets on {device}")


def _get_next_prefetch_buffers(fallback_tensors, device):
    """Return the next pre-allocated GPU buffer set (round-robin).

    Falls back to dynamic allocation when the pool is not initialised
    (e.g. first step before shapes are known).
    """
    global _ac_buffer_idx
    if _ac_gpu_buffers is not None:
        bufs = _ac_gpu_buffers[_ac_buffer_idx % len(_ac_gpu_buffers)]
        _ac_buffer_idx += 1
        return bufs
    if fallback_tensors is not None:
        return [torch.empty_like(t, device=device) for t in fallback_tensors]
    return None

class SlidingCheckpoint(saved_tensors_hooks):
    """基于save_on_cpu实现的transformer层tensor管理机制，使用单一checkpoint模式"""
    def __init__(
        self,
        layer_idx: int,  # 当前decoder层索引
        layer_tensors: List[List[torch.Tensor]] = None,  # 所有层的预分配CPU hidden_state tensors
        gds_offload: bool = False,
        file_paths: List[List[str]] = None,
        is_last_layer: bool = False,  # 是否是最后一层
        device: str = 'cuda:0',
        pin_memory: bool = True,
        stream: Optional[torch.cuda.Stream] = None,
        timing_recorder: Any = None,
        enable_timing: bool = False,
    ):
        self.layer_idx = layer_idx - 1
        self.device = device
        self.stream = stream or cp_stream
        self.is_last_layer = is_last_layer
        self.gds_offload = gds_offload
        self.timing_recorder = timing_recorder
        self.enable_timing = bool(
            enable_timing and timing_recorder is not None and not gds_offload
        )
        self._fwd_d2h_start_event = None
        
        # 根据模式初始化存储资源
        if gds_offload:
            assert file_paths is not None, "必须提供file_paths当启用GPU Direct Storage时"
            import kvikio
            self.file_paths = file_paths
            self._gds_prefetch_bufs = None
            
        else:
            assert layer_tensors is not None, "必须提供layer_tensors当使用CPU内存时"
            self.layer_tensors = layer_tensors
        
        # tensor计数器(用于pack和unpack). 零大小占位tensor不会计数。
        self.pack_counter = 0
        self.unpack_counter = 0
        
        # 必要的同步事件
        self.pre_pack_event = torch.cuda.Event()
        self.pre_unpack_event = torch.cuda.Event()
        self.post_unpack_event_prefetch = torch.cuda.Event()
        
        def _cpu_pack_hook(tensor: torch.Tensor) -> Tuple[torch.device, Any]:
            """将tensor打包到当前层的CPU空间"""
            # 如果tensor为空，直接返回
            if tensor.size() == torch.Size([0]) or not pin_memory:
                return (tensor.device, tensor.cpu())
            
            # 最后一层直接返回，跳过CPU拷贝
            if self.is_last_layer:
                return (tensor.device, tensor)
                
            # 获取当前层的CPU tensors
            current_tensors = self.layer_tensors[self.layer_idx]     
            cpu_tensor = current_tensors[self.pack_counter]
            
            if self.pack_counter == 0:
                self.pre_pack_event.record(stream=torch.cuda.default_stream())
                
            # 异步复制到CPU
            with torch.cuda.stream(self.stream):
                if self.pack_counter == 0:
                    self.stream.wait_event(self.pre_pack_event)
                    if self.enable_timing:
                        self._fwd_d2h_start_event = torch.cuda.Event(enable_timing=True)
                        self._fwd_d2h_start_event.record(self.stream)
                    # self.pre_pack_event.synchronize()
                cpu_tensor.copy_(tensor, non_blocking=True)
                
            self.pack_counter += 1 
            
            return (tensor.device, cpu_tensor)
            
        def _cpu_unpack_hook(packed: Tuple[torch.device, Any]) -> torch.Tensor:
            """从CPU解包tensor,在适当时机预取下一层"""
            device, tensor = packed

            if tensor.size() == torch.Size([0]) or not pin_memory:
                return tensor.to(device, non_blocking=pin_memory)
            
            # 在解包第一个真实tensor时触发预取下一层
            if self.unpack_counter == 0:
                self.pre_unpack_event.record(stream=torch.cuda.default_stream())
                
                next_layer_idx = self.layer_idx - 1
                if next_layer_idx >= 0:
                    next_tensors = self.layer_tensors[next_layer_idx]
                    temp_prefetch_buffers = _get_next_prefetch_buffers(next_tensors, self.device)
                    
                    # 异步预取下一层保存的activation tensors
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(self.pre_unpack_event)
                        if self.enable_timing:
                            ac_prefetch_start = torch.cuda.Event(enable_timing=True)
                            ac_prefetch_start.record(self.stream)
                        for cpu_tensor, gpu_buffer in zip(next_tensors, temp_prefetch_buffers):
                            gpu_buffer.copy_(cpu_tensor, non_blocking=True)
                        if self.enable_timing:
                            ac_prefetch_end = torch.cuda.Event(enable_timing=True)
                            ac_prefetch_end.record(self.stream)
                            self.timing_recorder.record_cuda_span(
                                "ac_bwd_h2d",
                                ac_prefetch_start,
                                ac_prefetch_end,
                            )
                        self.post_unpack_event_prefetch.record(stream=self.stream)
                         
                    fifo_prefetch_queue.append((temp_prefetch_buffers, self.post_unpack_event_prefetch))
                
            # 最后一层直接返回
            if self.is_last_layer:
                result = tensor
                self.unpack_counter += 1
            else:
                if not fifo_prefetch_queue:
                    print("Prefetch queue is empty!")
                    return tensor.to(device, non_blocking=pin_memory)
                
                next_tensors, unpack_event_prefetch = fifo_prefetch_queue[0]
                if self.unpack_counter == 0:
                    current_stream = torch.cuda.current_stream()
                    current_stream.wait_event(unpack_event_prefetch)
                
                result = next_tensors[self.unpack_counter]

                self.unpack_counter += 1
                if self.unpack_counter == self.pack_counter:
                    fifo_prefetch_queue.popleft()
                    
            return result
        
        def _gds_pack_hook(tensor: torch.Tensor) -> Tuple[Any, Any, Any]:
            """GDS模式打包：异步写入NVMe"""
            if tensor.size() == torch.Size([0]):
                return (tensor.cpu(), tensor.shape, tensor.dtype)
            
            if self.is_last_layer:
                return (tensor, tensor.shape, tensor.dtype)
            
            if self.layer_idx not in write_events:
                write_events[self.layer_idx] = {}
            
            file_path = self.file_paths[self.layer_idx][self.pack_counter]
            
            self.pre_pack_event.record(stream=torch.cuda.default_stream())
            # 异步写入文件
            with kvikio.CuFile(file_path, "w") as f:
                self.stream.wait_event(self.pre_pack_event)
                write_future = f.raw_write_async(tensor.detach(), self.stream.cuda_stream)
                # print(f"Packing tensor to GDS: {file_path}, shape={tensor.shape}, dtype={tensor.dtype}")
                write_events[self.layer_idx][self.pack_counter] = write_future
                # event = torch.cuda.Event()
                # event.record(stream=self.stream)
                # event.synchronize()
                            
            self.pack_counter += 1
            return (None, tensor.shape, tensor.dtype)
                
            
        def _gds_unpack_hook(packed: Tuple[Any, Any, Any]) -> torch.Tensor:
            """GDS模式解包：异步预取下一层并读取当前层"""
            tensor, shape, dtype = packed
            
            if tensor is not None and tensor.size() == torch.Size([0]):
                return tensor.to(self.device, non_blocking=pin_memory)
            
            self.pre_unpack_event.record(stream=torch.cuda.default_stream())

            next_layer_idx = self.layer_idx - 1
            if next_layer_idx >= 0:
                next_layer_path = self.file_paths[next_layer_idx][self.unpack_counter]
                write_events[next_layer_idx][self.unpack_counter].check_bytes_done()
                
                if self.unpack_counter == 0:
                    self._gds_prefetch_bufs = _get_next_prefetch_buffers(None, self.device)

                with kvikio.CuFile(next_layer_path, "r") as f:
                    self.stream.wait_event(self.pre_unpack_event)
                    if self._gds_prefetch_bufs is not None:
                        buffer = self._gds_prefetch_bufs[self.unpack_counter]
                    else:
                        buffer = torch.empty(shape, dtype=dtype, device=self.device)
                    future = f.raw_read_async(buffer, self.stream.cuda_stream)
                    fifo_prefetch_queue.append((buffer, future))

            # 最后一层直接返回GPU张量
            if self.is_last_layer:
                result = tensor
                self.unpack_counter += 1
            else:
                # 从预取队列获取缓冲区
                if not fifo_prefetch_queue:
                    raise RuntimeError("GDS预取队列为空")

                buffer, ready_event = fifo_prefetch_queue.popleft()
                ready_event.check_bytes_done()
                # event.synchronize()
                result = buffer
                self.unpack_counter += 1

            return result
            
        super().__init__(_gds_pack_hook if self.gds_offload else _cpu_pack_hook, _gds_unpack_hook if self.gds_offload else _cpu_unpack_hook)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (
            self.enable_timing
            and not self.is_last_layer
            and self.pack_counter > 0
            and self._fwd_d2h_start_event is not None
        ):
            with torch.cuda.stream(self.stream):
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record(self.stream)
            self.timing_recorder.record_cuda_span(
                "ac_fwd_d2h",
                self._fwd_d2h_start_event,
                end_event,
            )
        return super().__exit__(exc_type, exc_val, exc_tb)

# 保留原有的save_on_cpu类实现
class save_on_cpu(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.
        torch.autograd.graph的基础实现
    """

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            # print(f"Packing tensor to CPU (with pin_memory): shape={tensor.shape}, dtype={tensor.dtype}")
            return (tensor.device, packed)

        def unpack_from_cpu(packed: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            # print(f"Unpacking tensor from CPU: shape={tensor.shape}, dtype={tensor.dtype}")
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)
