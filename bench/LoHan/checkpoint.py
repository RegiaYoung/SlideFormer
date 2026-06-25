import torch
from torch.autograd.graph import saved_tensors_hooks
import queue
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.nn.parameter import Parameter
import functools
from nvme_ds.partitionrd_activation_swapper import PartitionedActStatus
import types
from nvtx import nvtx_wrap
from see_mem import see_memory_usage

fifo_queue = queue.Queue()
addmm_id = 0
# class save_on_cpu(saved_tensors_hooks):
#     def __init__(self, pin_memory=False, act_stream=torch.cuda.current_stream(), chp_id = [0], chp_list = [], act_swapper = None):

#         self.pre_unpack_event = torch.cuda.Event()
#         self.pre_pack_event = torch.cuda.Event()
#         self.post_unpack_event = torch.cuda.Event()
#         self.post_unpack_event_prefetch = torch.cuda.Event()
#         self.stream = act_stream
#         self.chp_id = chp_id
#         self.chp_list = chp_list
#         self.act_swapper = act_swapper

#         @nvtx_wrap
#         def pack_to_cpu(tensor):
#             # self._convert_to_manage_act(tensor, self.chp_id[0])
#             # print(self.is_manage_param(tensor))
#             # if self.is_manage_param(tensor):
#             #     print(tensor.summary())


#             if not pin_memory:
#                 return (tensor.device, tensor.cpu())

#             if tensor.size() == torch.Size([0]):
#                 packed = torch.empty(
#                             tensor.size(),
#                             dtype=tensor.dtype,
#                             layout=tensor.layout,
#                             pin_memory=(torch.cuda.is_available() and not tensor.is_sparse)
#                             )
            
#                 packed.copy_(tensor)
#                 return (tensor.device, packed)
#             packed = self.chp_list[self.chp_id[0]]
            

#             if not hasattr(packed, 'nvme_status'):
#                 self._convert_to_manage_act(packed, self.chp_id[0])
#                 # print(packed.summary())
#                 # partitioned_tensor = torch.empty(tensor.size(), dtype=tensor.dtype, device='cpu')
#                 # partitioned_tensor = partitioned_tensor.pin_memory()
#             tensor_size = tensor.numel()
#             if self.act_swapper.swappable_tensor(numel=tensor_size):
#                 buffer = self.act_swapper.get_buffer(packed, tensor_size)
#                 partitioned_tensor = torch.ones(1, dtype=torch.float16, device=buffer.device)
#                 partitioned_tensor.data = buffer.data
#             packed.manage = partitioned_tensor.view(tensor.shape)
#             packed.nvme_status = PartitionedActStatus.AVAILABLE
#             packed.sb_shape = tensor.shape
#             packed.sb_numel = packed.manage.numel()
#             packed.id = self.chp_id[0]
#                 # print(packed.summary())

#             self.chp_id[0] += 1

#             # print(packed.manage.size())
#             # print(tensor.size())
#             # packed = torch.empty(
#             #     tensor.size(),
#             #     dtype=tensor.dtype,
#             #     layout=tensor.layout,
#             #     pin_memory=(torch.cuda.is_available() and not tensor.is_sparse))
#             # self.pre_pack_event.record(stream=torch.cuda.default_stream())
#             # torch.cuda.default_stream().synchronize()
#             with torch.cuda.stream(self.stream):
#                 self.pre_pack_event.record(stream=torch.cuda.current_stream())
#                 self.pre_pack_event.synchronize()
#                 # packed.copy_(tensor, non_blocking=True)
#                 packed.manage.copy_(tensor, non_blocking=False)
#                 # print(packed.size())
#                 # if hasattr(packed, 'nvme_status'):
#                 #     print(id(packed.manage))
#                 #     print(packed.summary())

#             # torch.cuda.default_stream().synchronize()

#             # print('BEF', packed.manage.size())
#             # see_memory_usage('BEF swap_out_and_release')
#             self.act_swapper.swap_out_and_release(packed)
#             # print('AFT', packed.manage.size())
#             # see_memory_usage('AFT swap_out_and_release')
#             # packed.copy_(tensor)
            
#             return (tensor.device, packed)

#         @nvtx_wrap
#         def unpack_from_cpu(packed):
#             device, tensor = packed
#             # print(hasattr(tensor, 'nvme_status'))
#             # if self.is_manage_param(tensor):
#             #     print(tensor.summary())
            
#             # tensor = tensor.manage


#             if tensor.size() == torch.Size([0]):
#                 device, tensor = packed
#                 return tensor.to(device, non_blocking=pin_memory)
#             self.pre_unpack_event.record(stream=torch.cuda.default_stream())
#             # torch.cuda.default_stream().synchronize()

#             if fifo_queue.empty():
                
                
#                 with torch.cuda.stream(self.stream):
#                     self.pre_unpack_event.synchronize()
#                     self._ensure_availability_of_partitioned_acts(tensor)
                    
#                     tensor.manage = tensor.manage.view(tensor.sb_shape)
#                     result = tensor.manage.to(device, non_blocking=False)
#                     self.post_unpack_event.record(stream=self.stream)

#                 self.post_unpack_event.synchronize()

#                 self.act_swapper.remove_activation_and_release_buffers([tensor])
#             else:
#                 temp_prefetch, self.post_unpack_event_prefetch = fifo_queue.get()
#                 self.post_unpack_event_prefetch.synchronize()
#                 result = temp_prefetch

#             self.chp_id[0] -= 1
#             if self.chp_id[0] != 0:
#                 self._ensure_availability_of_partitioned_acts(self.chp_list[self.chp_id[0] - 1])
#                 self.chp_list[self.chp_id[0] - 1].manage = self.chp_list[self.chp_id[0] - 1].manage.view(self.chp_list[self.chp_id[0] - 1].sb_shape)
#                 temp_prefetch = torch.empty(self.chp_list[self.chp_id[0] - 1].manage.size(), dtype=torch.float16, device='cuda:0')
#                 fifo_queue.put((temp_prefetch, self.post_unpack_event_prefetch))
#                 with torch.cuda.stream(self.stream):
                    
#                     # print('2', self.chp_list[self.chp_id[0] - 1].manage.size())   
#                     temp_prefetch.copy_(self.chp_list[self.chp_id[0] - 1].manage, non_blocking=False)
#                     self.post_unpack_event_prefetch.record(stream=self.stream)
#                 self.act_swapper.remove_activation_and_release_buffers([self.chp_list[self.chp_id[0] - 1]])
            
#             return result
            
#             # return tensor.to(device, non_blocking=pin_memory)

#         super().__init__(pack_to_cpu, unpack_from_cpu)

#     def _convert_to_manage_act(self, act_block, chp_id):

#         act_block.nvme_status = PartitionedActStatus.AVAILABLE

#         act_block.sb_shape = act_block.shape

#         act_block.sb_numel = act_block.numel()

#         act_block.manage = None

#         act_block.id = chp_id

#         def summary(slf: torch.Tensor) -> dict:
#             return {
#                 "id": slf.id,
#                 "status": slf.nvme_status,
#                 "numel": slf.sb_numel,
#                 "sb_shape": tuple(slf.sb_shape),
#             }
        
#         act_block.summary = types.MethodType(summary, act_block)
#     def is_manage_param(self, act_block):
#         """
#         判断参数是否被系统管理
#         """
#         if not torch.is_tensor(act_block):
#             return False
#         return hasattr(act_block, 'nvme_status')

#     @nvtx_wrap
#     def _ensure_availability_of_partitioned_acts(self, act_block):
#         swap_in_list = []
#         swap_in_flight = []


#         if act_block.nvme_status == PartitionedActStatus.NOT_AVAILABLE:
#             # print('get NOT_AVAILABLE', param.sb_id)
#             swap_in_list.append(act_block)
#         if act_block.nvme_status == PartitionedActStatus.INFLIGHT:
#             swap_in_flight.append(act_block)

#         if len(swap_in_list) > 0:
#             self.act_swapper.swap_in(swap_in_list[0], async_op=False)
#         elif len(swap_in_flight) > 0:
#             self.act_swapper.synchronize_reads()


class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory=False, act_stream=torch.cuda.current_stream(), chp_id = [0], chp_list = [], act_swapper = None):

        self.pre_unpack_event = torch.cuda.Event()
        self.pre_pack_event = torch.cuda.Event()
        self.post_unpack_event = torch.cuda.Event()
        self.post_unpack_event_prefetch = torch.cuda.Event()
        self.stream = act_stream
        self.chp_id = chp_id
        self.chp_list = chp_list
        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())

            if tensor.size() == torch.Size([0]):
                packed = torch.empty(
                            tensor.size(),
                            dtype=tensor.dtype,
                            layout=tensor.layout,
                            pin_memory=(torch.cuda.is_available() and not tensor.is_sparse)
                            )
            
                packed.copy_(tensor)
                return (tensor.device, packed)
            # print(self.chp_id[0])
            packed = self.chp_list[self.chp_id[0]]
            self.chp_id[0] += 1
            # packed = torch.empty(
            #     tensor.size(),
            #     dtype=tensor.dtype,
            #     layout=tensor.layout,
            #     pin_memory=(torch.cuda.is_available() and not tensor.is_sparse))
            self.pre_pack_event.record(stream=torch.cuda.default_stream())
            # torch.cuda.default_stream().synchronize()
            with torch.cuda.stream(self.stream):
                self.pre_pack_event.synchronize()
                packed.copy_(tensor, non_blocking=True)

            # packed.copy_(tensor)
            
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            device, tensor = packed

            if tensor.size() == torch.Size([0]):
                device, tensor = packed
                return tensor.to(device, non_blocking=pin_memory)
            self.pre_unpack_event.record(stream=torch.cuda.default_stream())
            # torch.cuda.default_stream().synchronize()

            if fifo_queue.empty():

                with torch.cuda.stream(self.stream):
                    self.pre_unpack_event.synchronize()
                    result = tensor.to(device, non_blocking=True)
                    self.post_unpack_event.record(stream=self.stream)
                
                self.post_unpack_event.synchronize()
            
            else:
                temp_prefetch, self.post_unpack_event_prefetch = fifo_queue.get()
                self.post_unpack_event_prefetch.synchronize()
                result = temp_prefetch
                
            self.chp_id[0] -= 1
            if self.chp_id[0] != 0:
                temp_prefetch = torch.empty(self.chp_list[self.chp_id[0] - 1].size(), dtype=torch.float16, device='cuda:0')
                fifo_queue.put((temp_prefetch, self.post_unpack_event_prefetch))
                with torch.cuda.stream(self.stream):
                    temp_prefetch.copy_(self.chp_list[self.chp_id[0] - 1], non_blocking=True)
                    self.post_unpack_event_prefetch.record(stream=self.stream)
            
            
            return result
            
            # return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)

act_stream = torch.cuda.Stream()

def _detach_to_cpu(x, extra_param, act_pack):
    if isinstance(x, torch.Tensor):

        # packed = torch.empty(
        #         x.size(),
        #         dtype=x.dtype,
        #         layout=x.layout,
        #         pin_memory=True)
        packed = act_pack[extra_param]
        torch.cuda.default_stream().synchronize()
        
        with torch.cuda.stream(act_stream):
            # temp_event = torch.cuda.Event()
            # temp_event.record(stream=torch.cuda.current_stream())
            # temp_event.synchronize()
            packed.copy_(x, non_blocking=True)
    # print(extra_param)
    return packed

def _to_cuda(packed, extra_param, act_pack):
    if isinstance(packed, torch.Tensor):
        # torch.cuda.default_stream().synchronize()
        with torch.cuda.stream(act_stream):
            # packed = act_pack[extra_param]
            packed =  packed.to('cuda:0', non_blocking=True)
        return packed
    
    return packed

def _get_default_policy1(allow_list=None):
    _default_allow_list = [

    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(func, *args, **kwargs):
        return func in allow_list

    return _default_policy

def _get_default_policy2(allow_list=None):
    _default_allow_list = [
        # torch.ops.aten.addmm.default,
        # torch.ops.aten.mm.default,
    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(func, *args, **kwargs):
        return func in allow_list

    return _default_policy
def get_selective_offloading_checkpoint_modes1():
    policy_fn = _get_default_policy1()
    cpu_storage = []

    class CachingMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if policy_fn(func, *args, **kwargs):
                # print(kwargs)
                out = func(*args, **kwargs)
                # Detach and move tensors to cpu
                out_detached_cpu = tree_map(_detach_to_cpu, out)
                cpu_storage.append(out_detached_cpu)
                return out
            return func(*args, **kwargs)

    class CachedMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if policy_fn(func, *args, **kwargs):
                # Detach and move tensors back to cuda
                # print(kwargs)
                out = tree_map(_to_cuda, cpu_storage.pop(0))
                return out
            return func(*args, **kwargs)

    return CachingMode(), CachedMode()

def get_selective_offloading_checkpoint_modes2(swap_list, act_pack):
    if len(swap_list) != 0:
        allow_list = [
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        ]
    else:
        allow_list = None
    policy_fn = _get_default_policy2(allow_list)
    cpu_storage = []
    
    class CachingMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            global addmm_id
            kwargs = {} if kwargs is None else kwargs
            # if len(args) == 3 and isinstance(args[0], Parameter) and isinstance(args[1], torch.Tensor) and isinstance(args[2], torch.Tensor):
            #     if addmm_id % 4 == 3:
            #         print(addmm_id)
            # # print('-------------------')
            #         if policy_fn(func, *args, **kwargs):
            #             # print(kwargs)
            #             out = func(*args, **kwargs)
            #             # Detach and move tensors to cpu

            #             print(func, 'gpu -> cpu', out.size(), addmm_id)
                        
            #             out_detached_cpu = tree_map(_detach_to_cpu, out)
            #             cpu_storage.append((out_detached_cpu, addmm_id))
            #             addmm_id += 1
            #             return out
            #     addmm_id += 1

            if len(args) == 3 and isinstance(args[0], Parameter) and isinstance(args[1], torch.Tensor) and isinstance(args[2], torch.Tensor):

                if addmm_id in swap_list:
                    
        # print('-------------------')
                    if policy_fn(func, *args, **kwargs):
                        # print(kwargs)
                        out = func(*args, **kwargs)
                        # Detach and move tensors to cpu

                        # print(func, 'gpu -> cpu', out.size(), addmm_id)
                        # print('save', addmm_id, out.size())

                        detach_to_cpu_with_param = functools.partial(_detach_to_cpu, extra_param=addmm_id, act_pack=act_pack)

                        out_detached_cpu = tree_map(detach_to_cpu_with_param, out)
                        cpu_storage.append((out_detached_cpu, addmm_id))
                        addmm_id += 1
                        return out
                addmm_id += 1

            # if policy_fn(func, *args, **kwargs):
            #     # print(kwargs)
            #     out = func(*args, **kwargs)
            #     # Detach and move tensors to cpu

            #     print(func, 'gpu -> cpu', out.size(), addmm_id)
            #     addmm_id += 1

            #     out_detached_cpu = tree_map(_detach_to_cpu, out)
            #     cpu_storage.append(out_detached_cpu)
            #     return out
            

            # out = func(*args, **kwargs)
            # for i in out:
            #     print('re-compute', addmm_id, i.size())
            return func(*args, **kwargs)

    class CachedMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            global addmm_id
            kwargs = {} if kwargs is None else kwargs
            # if len(args) == 3 and isinstance(args[0], Parameter) and isinstance(args[1], torch.Tensor) and isinstance(args[2], torch.Tensor):
            #     addmm_id -= 1
            #     if addmm_id % 4 == 2:
            #         # print(addmm_id)
            # # print('-------------------')
            #         if policy_fn(func, *args, **kwargs):
            #             # print(kwargs)
            #             # Detach and move tensors back to cuda
            #             print(func, 'gpu <- cpu', cpu_storage[0][0].size(), addmm_id, cpu_storage[0][1])
            #             out = tree_map(_to_cuda, cpu_storage.pop(0))   
            #             out = out[0]          
            #             return out

            if len(args) == 3 and isinstance(args[0], Parameter) and isinstance(args[1], torch.Tensor) and isinstance(args[2], torch.Tensor):
                addmm_id -= 1
                # print('cached ', addmm_id)
                # print(args[0].size())
                # print(args[1].size())
                # print(args[2].size())
                if len(cpu_storage) != 0:
                    # print(cpu_storage[0][1])
                    if (cpu_storage[0][1] % 2 == 0 and cpu_storage[0][1] == addmm_id - 1 ) or (cpu_storage[0][1] % 2 == 1 and cpu_storage[0][1] == addmm_id + 1):
                        # if addmm_id % 4 == 2:
                        # print(addmm_id)
                # print('-------------------')
                        if policy_fn(func, *args, **kwargs):
                            # print(kwargs)
                            # Detach and move tensors back to cuda
                            # print(func, 'gpu <- cpu', cpu_storage[0][0].size(), cpu_storage[0][1])

                            detach_to_cuda_with_param = functools.partial(_to_cuda, extra_param=cpu_storage[0][1], act_pack=act_pack)
                            out = tree_map(detach_to_cuda_with_param, cpu_storage.pop(0)[0])   
                            # out = out[0]          
                            return out

            # if policy_fn(func, *args, **kwargs):
            #     # print(kwargs)
            #     # Detach and move tensors back to cuda
            #     addmm_id -= 1
            #     print(func, 'gpu <- cpu', cpu_storage[0].size(), addmm_id)
            #     out = tree_map(_to_cuda, cpu_storage.pop(0))             
            #     return out
            # print(args)

            return func(*args, **kwargs)

    return CachingMode(), CachedMode()
