import torch
from torch.autograd.graph import saved_tensors_hooks
from collections import deque, OrderedDict
from typing import Any, Tuple, Optional, List, Dict

# 全局预取队列
fifo_prefetch_queue = deque()
cp_stream = torch.cuda.Stream()
write_events = OrderedDict()

class SlidingCheckpoint(saved_tensors_hooks):
    """基于save_on_cpu实现的transformer层tensor管理机制，使用单一checkpoint模式"""
    def __init__(
        self,
        layer_idx: int,  # 当前层索引
        layer_tensors: List[List[torch.Tensor]] = None,  # 所有层的预分配CPU tensors
        gds_offload: bool = False,
        file_paths: List[List[str]] = None,
        is_last_layer: bool = False,  # 是否是最后一层
        no_mask: bool = False,
        device: str = 'cuda:0',
        pin_memory: bool = True,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        self.layer_idx = layer_idx - 1
        self.device = device
        self.stream = stream or cp_stream
        self.is_last_layer = is_last_layer
        self.no_mask = no_mask 
        self.gds_offload = gds_offload
        
        # 根据模式初始化存储资源
        if gds_offload:
            assert file_paths is not None, "必须提供file_paths当启用GPU Direct Storage时"
            import kvikio
            self.file_paths = file_paths
            
        else:
            assert layer_tensors is not None, "必须提供layer_tensors当使用CPU内存时"
            self.layer_tensors = layer_tensors
        
        # tensor计数器(用于pack和unpack)
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
                    # self.pre_pack_event.synchronize()
                cpu_tensor.copy_(tensor, non_blocking=True)
                
            self.pack_counter += 1 
            
            return (tensor.device, cpu_tensor)
            
        def _cpu_unpack_hook(packed: Tuple[torch.device, Any]) -> torch.Tensor:
            """从CPU解包tensor,在适当时机预取下一层"""
            device, tensor = packed
            
            if tensor.size() == torch.Size([0]) or not pin_memory:
                # device, tensor = packed
                return tensor.to(device, non_blocking=pin_memory)
            
            # 在解包第一个tensor时触发预取下一层
            if self.unpack_counter == 0:
                
                # torch.cuda.default_stream().synchronize()
                self.pre_unpack_event.record(stream=torch.cuda.default_stream())
                
                # 预取下一层的两个tensor
                next_layer_idx = self.layer_idx - 1
                if next_layer_idx >= 0:
                    # 预取下一层的两个tensor
                    next_tensors = self.layer_tensors[next_layer_idx]
                    
                    temp_prefetch_buffers =[torch.empty_like(t, device = self.device) for t in next_tensors] # 注意此处device
                    
                    # 异步预取下一层的所有tensor
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(self.pre_unpack_event)
                        for cpu_tensor, gpu_buffer in zip(next_tensors, temp_prefetch_buffers):
                            gpu_buffer.copy_(cpu_tensor, non_blocking=True)
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
                    # unpack_event_prefetch.synchronize()
                    # 让当前GPU流（计算流）等待预取流完成，而不是阻塞CPU
                    current_stream = torch.cuda.current_stream()
                    current_stream.wait_event(unpack_event_prefetch)
                
                # 根据顺序决定使用哪个预取的tensor
                result = next_tensors[self.unpack_counter]

                # 使用完所有tensor后移除这组预取结果
                self.unpack_counter += 1
                if self.no_mask or self.unpack_counter == self.pack_counter:  # 两个tensor都已经unpacked
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
                # 预取下一层的两个tensor
                # print(next_layer_idx, self.unpack_counter)
                next_layer_path = self.file_paths[next_layer_idx][self.unpack_counter]
                # print(f"Prefetching file: {next_layer_path}")
                # 创建GPU缓冲区
                write_events[next_layer_idx][self.unpack_counter].check_bytes_done()
                
                with kvikio.CuFile(next_layer_path, "r") as f:
                    self.stream.wait_event(self.pre_unpack_event)
                    buffer = torch.empty(shape, dtype=dtype, device=self.device)
                    future = f.raw_read_async(buffer, self.stream.cuda_stream)
                    # event = torch.cuda.Event()
                    # event.record(stream=self.stream)
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
