import torch
from functools import wraps
from utils import print_format
from torch.nn.modules import Module
import os
import itertools
import torch.nn as nn
from typing import Callable, Iterable
import functools
from torch.nn import Parameter
from enum import Enum
from torch import Tensor
from typing import List
import types
from nvtx import nvtx_wrap
from debug import debug_param2name_id_shape, debug_param2name_id
from nvme_ds.partitioned_param_swapper import AsyncPartitionedParameterSwapper, PartitionedParamStatus
from typing import Deque, Set
import json
from collections import namedtuple
from see_mem import see_memory_usage

fetch_param_hindle_list = Deque()
fetch_new_param_hindle_list = {}

id_2_param = {}
max_id = 0

fetch_param_stream = torch.cuda.Stream()
# fetch_param_stream = torch.cuda.default_stream()

param_count = 0
_orig_torch_empty = torch.empty
_orig_torch_zeros = torch.zeros
_orig_torch_ones = torch.ones
_orig_torch_full = torch.full
prefetch_end = 0
param_block_id = 0
prefetch_block_fw_id = 0
forward_time = 0
global_id = 0
swap_fw_flag = True
class ParamStatus(Enum):
    """参数状态（是否可用）

    Args:
        Enum
    """
    # on GPU
    READY = 1

    # on CPU or NVM
    NO_READY = 2

    # fetching
    INFLIGHT = 3

    
@nvtx_wrap
class FetchHandle:
    def __init__(self, param: Parameter) -> None:
        """从CPU获取参数到param.data
        Args:
            param (Parameter): 单个参数

        Raises:
            RuntimeError: 确保是INFLIGHT状态
        """
        self.temp_event = torch.cuda.Event()
        self.param = param
        if param.status != ParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")

        # param.data = param.manage.data.to(device='cuda:0',non_blocking=True).view(param.sb_shape)
        # self.__param = param
        @nvtx_wrap
        def FetchHandle_synchronize():
            pass
            # torch.cuda.current_stream().synchronize()
        FetchHandle_synchronize()
        with torch.cuda.stream(fetch_param_stream):
            param.data = param.manage.data.to(device='cuda:0',non_blocking=True).view(param.sb_shape)
            self.__param = param
            self.temp_event.record(stream=fetch_param_stream)

    def wait(self) -> None:
        """
        等待数据传输完成, 将状态更新为AVAILABLE
        """
        self.temp_event.synchronize()
        @nvtx_wrap
        def FetchHandle_wait_synchronize():
            pass
            # torch.cuda.current_stream().synchronize()
        FetchHandle_wait_synchronize()
        self.__param.status = ParamStatus.READY

@nvtx_wrap
class FetchCoalescedHandle:
    def __init__(self, params: List[Parameter]) -> None:
        """从CPU获取参数组到param.data

        Args:
            params (List[Parameter]): 参数组

        Raises:
            RuntimeError: 确保是INFLIGHT状态
        """
        self.__params = params
        self.__complete = False
        self.temp_event = torch.cuda.Event()

        # for param in self.__params:
        #     if param.status != ParamStatus.INFLIGHT:
        #         raise RuntimeError(f"expected param {param.ds_summary()} to not be available")
        #     param.data = param.manage.data.to(device='cuda:0',non_blocking=True).view(param.sb_shape)
        @nvtx_wrap
        def FetchCoalescedHandle_synchronize():
            pass
            # torch.cuda.current_stream().synchronize()
        FetchCoalescedHandle_synchronize()
        with torch.cuda.stream(fetch_param_stream):
            for param in self.__params:
                if param.status != ParamStatus.INFLIGHT:
                    raise RuntimeError(f"expected param {param.ds_summary()} to not be available")
                param.data = param.manage.data.to(device='cuda:0',non_blocking=True).view(param.sb_shape)
                self.temp_event.record(stream=fetch_param_stream)

    def wait(self) -> None:
        """
        等待参数组传输完成, 将状态更新为AVAILABLE
        """

        self.temp_event.synchronize()
        if self.__complete:
            return
        @nvtx_wrap
        def FetchCoalescedHandle_wait_synchronize():
            pass
            # torch.cuda.current_stream().synchronize()
        FetchCoalescedHandle_wait_synchronize()
        for param in self.__params:
            assert param.status == ParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            param.status = ParamStatus.READY
        self.__complete = True

def is_manage_param(parameter):
    """
    判断参数是否被系统管理
    """
    if not torch.is_tensor(parameter):
        return False
    return hasattr(parameter, 'sb_id')

def get_all_subclasses(cls):
    """获取当前module的所有sub module
    """
    subclass_list = []
    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)
    return set(subclass_list)

def shutdown_init_context():
    """将初始化阶段所有替换的默认方法复原
    """
    def _disable_class(cls):
        cls.__init__ = cls._old_init

    for subclass in get_all_subclasses(torch.nn.modules.module.Module):
        _disable_class(subclass)
    torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
    torch.Tensor.__new__ = torch.Tensor.__old_new__
    torch.empty = _orig_torch_empty
    torch.zeros = _orig_torch_zeros
    torch.ones = _orig_torch_ones
    torch.full = _orig_torch_full

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse))


def iter_params(module: Module, recurse=False):
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))

@nvtx_wrap
def free_param(param):
    """释放掉参数的底层存储
    """
    # param.data.record_stream(torch.cuda.current_stream())
    param.data = torch.empty(0, dtype=param.dtype, device='cuda:0')
    # param.data = torch.empty(0, dtype=torch.float32, device='cuda:0')
    
    param.status = ParamStatus.NO_READY
    
def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:

    def new_tensor(cls, *args) -> Tensor:
        device = torch.device('cuda:0')
        tensor = _orig_torch_empty(0, device=device).new_empty(*args)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)

        return tensor

    return new_tensor

def zero_wrapper_for_fp_tensor_constructor(fn: Callable, target_fp_dtype: torch.dtype) -> Callable:

    def wrapped_fn(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            kwargs['device'] = torch.device('cuda:0')
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor.to(target_fp_dtype)

        return tensor

    return wrapped_fn

def fetch_coalesced(params: Iterable[Parameter]):
    for param in params:
        if param.status != ParamStatus.NO_READY:
            raise RuntimeError('fetch_coalesced error')
        param.status = ParamStatus.INFLIGHT

    params = sorted(params, key=lambda p: p.sb_id)
    if len(params) == 1:
        param, = params
        return FetchHandle(param)
    return FetchCoalescedHandle(params)

def isinstance_namedtuple(obj: object) -> bool:
    """
    Is this an instance of namedtuple/NamedTuple?
    From: https://stackoverflow.com/a/62692640
    """
    return isinstance(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')
def is_builtin_type(obj):
    # https://stackoverflow.com/a/17795199
    return obj.__class__.__module__ == '__builtin__' or obj.__class__.__module__ == "builtins"

def _apply_to_tensors_only(module, functional, backward_function, outputs):

    if isinstance(outputs, (tuple, list)):
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional, backward_function, output)
            touched_outputs.append(touched_output)

        if isinstance_namedtuple(outputs):
            # namedtuples require a slightly different syntax.
            return outputs.__class__(*touched_outputs)

        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in outputs.keys():
            outputs[key] = _apply_to_tensors_only(module, functional, backward_function, outputs[key])
        return outputs

    elif isinstance(outputs, torch.Tensor):
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_outputs = functional.apply(module, backward_function, outputs)

        # restore zero param attributes if those get stripped by `backward_function`
        if not is_manage_param(touched_outputs) and is_manage_param(outputs):
            touched_outputs.ds_param_alias = outputs
        return touched_outputs
    else:
        if not is_builtin_type(outputs):
            global warned
            if not warned:
                warned = True
        return outputs

class PreBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, "applied_pre_backward_ref_cnt"):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        # print(f"After Forward: {ctx.module.__class__.__name__}")
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print('backward -- ', ctx.module.id)
        # print(f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args

class PostBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        if output.requires_grad:
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
            # print(f"After Backward: {ctx.module.__class__.__name__}")
        return (None, None) + args


class SB_Init(object):

    def __init__(self):
        path = os.getcwd()
        self.file = open(path + "/model_info.txt", 'w')
        print_format('SB_init BEGIN INIT', opt_type='event', loc=self.file)
        self.dtype = torch.float16

    def __enter__(self):

        def partition_after(f):
            @wraps(f)
            def wrapper(module, *args, **kwargs):

                is_child_module = False
                if not hasattr(module, "_ds_child_entered"):
                    is_child_module = True
                    setattr(module, "_ds_child_entered", True) 

                f(module, *args, **kwargs)

                if is_child_module:
                    delattr(module, "_ds_child_entered")

                    self._post_init_method(module)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)

        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _enable_class(subclass)

        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        torch.Tensor.__old_new__ = torch.Tensor.__new__
        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.empty = zero_wrapper_for_fp_tensor_constructor(_orig_torch_empty, self.dtype)
        torch.zeros = zero_wrapper_for_fp_tensor_constructor(_orig_torch_zeros, self.dtype)
        torch.ones = zero_wrapper_for_fp_tensor_constructor(_orig_torch_ones, self.dtype)
        torch.full = zero_wrapper_for_fp_tensor_constructor(_orig_torch_full, self.dtype)
        print_format('SB_init ENTER', opt_type='event', loc=self.file)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print_format(f"\nModel Param count : {param_count}", loc=self.file)
        print_format('SB_init EXIT', opt_type='event', loc=self.file)
        self.file.close()

        shutdown_init_context()
    
    def _post_init_method(self, module):
        pass


class Init(SB_Init):
    param_id = 0
    param_persistence_threshold = 20000000
    model_persistence_threshold = 20000000
    num_persisted_parameters = 0
    num_persisted_elements = 0
    apply_param_persistence = False

    def __init__(self,
                 module=None,
                 is_nvme=False,
                 is_nvme_async=False,
                 config='/home/lcy/asplos/config.json',
                 ):
        super().__init__()
        self.local_device = torch.device('cuda:0')
        torch.cuda.set_device(self.local_device)
        self.remote_device = 'cpu'
        self.pin_memory = True
        self.is_nvme = is_nvme
        self.is_nvme_async = is_nvme_async
        self.dtype = torch.half
        see_memory_usage("Init")
        def json_object_hook(d): 
            return namedtuple('X', d.keys())(*d.values())
        with open(config) as f: 
            ds_config = json.load(f, object_hook=json_object_hook)

        self._ds_config = ds_config
        print(self._ds_config.zero_config.offload_param)
        if self.is_nvme:
            self.param_swapper = AsyncPartitionedParameterSwapper(self._ds_config, self.dtype)
        else:
            self.param_swapper = None
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            self._convert_to_zero_parameters(module.parameters(recurse=True))


    def _convert_to_zero_parameters(self, param_list):
        for param in param_list:
            if is_manage_param(param):
                continue
            self._convert_to_deepspeed_param(param)
            param.partition()

    def _post_init_method(self, module):
        print_format(f'Module {module.__class__.__name__}', loc=self.file)

        global param_count
        for name, param in module.named_parameters(recurse=False):

            param_count += param.numel()
            if not is_manage_param(param):
                self._convert_to_deepspeed_param(param)
                # print(param)
                print_format(f'id : {param.sb_id :<20} | name : {name:<20} | size : {str(tuple(param.size())):<20} | count : {param.numel():<10}', loc=self.file)
                param.partition()

    def _convert_to_deepspeed_param(self, param):

        param.status = ParamStatus.READY

        param.sb_shape = param.shape

        param.ds_numel = param.numel()

        # Stores the partitioned copy of the tensor
        param.manage = None

        # Keeps track of how many active sub-modules need this param at any given point in time
        param.ds_active_sub_modules = set()
        param.nvme_swapper = self.param_swapper

        # 小于指定阈值的参数驻留在GPU上
        if param.ds_numel <= Init.param_persistence_threshold:
            param.ds_persist = True
            Init.num_persisted_parameters += 1
            Init.num_persisted_elements += param.ds_numel
        else:
            param.ds_persist = False

        param.sb_id = Init.param_id
        Init.param_id += 1

        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            if param_list is None:
                param_list = [cls]
            self._partition(param_list, has_been_updated=has_been_updated)

        def all_gather_coalesced(params: Iterable[Parameter], safe_mode: bool = False):
            # for param in params:
                # print(param.manage.nvme_status)
            global max_id
            with torch.cuda.stream(fetch_param_stream):

                if self.is_nvme:
                    self._ensure_availability_of_partitioned_params(params)
                # for param in params:
                    # print(param.manage.nvme_status)
            return fetch_coalesced(params)
        
        def aligned_size():
            return self._aligned_size(param)

        def partition_numel():
            return self._partition_numel(param)

        def convert_to_zero_parameters(param_list):
            self._convert_to_zero_parameters(param_list)

        def ds_summary(slf: torch.Tensor, use_debug_name: bool = False) -> dict:
            return {
                "id": debug_param2name_id(slf) if use_debug_name else slf.sb_id,
                "status": slf.status.name,
                "numel": slf.numel(),
                "ds_numel": slf.ds_numel,
                "shape": tuple(slf.shape),
                "ds_shape": tuple(slf.sb_shape),
                "requires_grad": slf.requires_grad,
                "grad_shape": tuple(slf.grad.shape) if slf.grad is not None else None,
                "persist": slf.ds_persist,
                "active_sub_modules": slf.ds_active_sub_modules,
                "ds_tensor.shape": slf.manage.shape if slf.manage is not None else None
            }

        def all_gather(param_list=None, async_op=False, hierarchy=0):
            if self.is_nvme:
                self._ensure_availability_of_partitioned_params(param_list)
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)
        # Collectives for gathering and partitioning parameters
        param.partition = partition
        param.all_gather = all_gather
        param.all_gather_coalesced = all_gather_coalesced

        # Partitioning size utilities
        param.aligned_size = aligned_size
        param.partition_numel = partition_numel

        param.convert_to_zero_parameters = convert_to_zero_parameters
        param.ds_summary = types.MethodType(ds_summary, param)
    
    @nvtx_wrap
    def _ensure_availability_of_partitioned_params(self, params):
        swap_in_list = []
        swap_in_flight = []

        for param in params:
            # print('shoudle have', param.sb_shape)
            # print(param.manage.nvme_status)
            if param.manage.nvme_status == PartitionedParamStatus.NOT_AVAILABLE:
                # print('get NOT_AVAILABLE', param.sb_id)
                assert param.manage.final_location == 'nvme' and param.status == ParamStatus.NO_READY
                swap_in_list.append(param)
            if param.manage.nvme_status == PartitionedParamStatus.INFLIGHT:
                print('INFLIGHT', param.sb_id)
                assert param.manage.final_location == 'nvme' and param.status == ParamStatus.NO_READY
                swap_in_flight.append(param)
        if len(swap_in_list) > 0:
            swap_in_list[0].nvme_swapper.swap_in(swap_in_list, async_op=False)
        elif len(swap_in_flight) > 0:
            swap_in_flight[0].nvme_swapper.synchronize_reads()

    def _all_gather(self, param_list, async_op=False, hierarchy=None):


        handles = []
        all_gather_list = []
        for param in param_list:
            if param.status == ParamStatus.NO_READY:
                all_gather_list.append(param)

        if not async_op:
            if len(param_list) == 1:
                ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            else:
                ret_value = self._allgather_params_coalesced(all_gather_list, hierarchy)

            for param in all_gather_list:
                param.status = ParamStatus.READY
            return ret_value

    def _allgather_params(self, param_list, hierarchy=0):
        if len(param_list) == 0:
            return

        partition_size = sum([param.manage.ds_numel for param in param_list])

        tensor_size = partition_size
        flat_tensor = torch.empty(tensor_size, dtype=param_list[0].dtype, device=self.local_device)
        flat_tensor.requires_grad = False
        partitions = []

        partitions.append(flat_tensor)

        for param in param_list:
            param_numel = param.manage.ds_numel
            partitions[0].narrow(0, 0, param_numel).copy_(param.manage.data)


        for param in param_list:
            param_partition_size = param.manage.ds_numel
            param_size = param.ds_numel
            replicated_tensor = torch.empty(param.sb_shape, dtype=param.dtype, device=self.local_device)

            numel_to_copy = min(param_size - 0, param_partition_size)

            part_to_copy = partitions[0].narrow(0, 0, numel_to_copy)

            replicated_tensor.view(-1).narrow(0, 0, numel_to_copy).copy_(part_to_copy)


            param.data = replicated_tensor.data
            # print(f'replicated_tensor {param.expand_as(param).grad_fn}')

        return None

    def _allgather_params_coalesced(self, param_list, hierarchy=0):
        pass

    def _aligned_size(self, param):
        return param.ds_numel


    def _partition_numel(self, param):
        return param.manage.ds_numel


    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            self._partition_param(param, has_been_updated=has_been_updated)
            param.status = ParamStatus.NO_READY


    def _partition_param(self, param, buffer=None, has_been_updated=False):
        assert param.status is not ParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"
        # print(param.device)
        global reuse_buffers
        if param.status is ParamStatus.READY:

            # if param.manage is not None and not param.ds_persist:
            if param.manage is not None:
                # if param.manage.ds_numel == 257315840 or param.manage.ds_numel ==104857600:
                #     return
                free_param(param)
                if param.manage.final_location == 'nvme' and self.is_nvme:
                    # print(f"Param {param.sb_id} partition - since it exists in nvme")
                    # print(f'free {param.sb_id}')
                    param.nvme_swapper.remove_partition_and_release_buffers([param])
                return

            tensor_size = self._aligned_size(param)

            if param.manage is None:   
                final_location = None
                if self.is_nvme and self.param_swapper.swappable_tensor(numel=tensor_size):
                    final_location = 'nvme'
                    buffer = self.param_swapper.get_buffer(param, tensor_size)
                    partitioned_tensor = torch.empty(0, dtype=param.dtype, device=buffer.device)
                    partitioned_tensor.data = buffer.data
                    # print(f"ID {param.sb_id} Initializing partition for the first time for nvme offload.")
                else:
                    
                    if param.ds_persist:
                        device = self.local_device
                    else:
                        device = self.remote_device

                    partitioned_tensor = torch.empty(tensor_size, dtype=param.dtype, device=device)

                    if device == 'cpu' and self.pin_memory:
                        partitioned_tensor = partitioned_tensor.pin_memory()

                partitioned_tensor.requires_grad = False
                param.manage = partitioned_tensor
                param.manage.ds_numel = tensor_size
                # param.manage.status = ParamStatus.READY
                param.manage.nvme_status = PartitionedParamStatus.AVAILABLE
                param.manage.final_location = final_location


            one_dim_param = param.contiguous().view(-1)
            src_tensor = one_dim_param.narrow(0, 0, tensor_size)
            @nvtx_wrap
            def partition_copy():
                param.manage.narrow(0, 0, tensor_size).copy_(src_tensor, non_blocking=False)
            partition_copy()
            free_param(param)

            if param.manage.final_location == 'nvme' and self.is_nvme:
                self.param_swapper.swap_out_and_release([param])
                # print(f"ID {param.sb_id} Offloaded to nvme offload and buffers released.")
def SB_hook(model, is_new_param_async, fw_time, is_swap_and_recompute):

    def _pre_forward_module_hook(module, input, *args):
        global forward_time, prefetch_block_fw_id, global_id
        if module.id == 0:
            global_id += 1
            forward_time = 0
            prefetch_block_fw_id = 0
        test_pre_forward(module, input)
        
        if global_id == 1:
            nonlocal start_event
            start_event.record()
            start_event.synchronize()

    @nvtx_wrap
    @torch.no_grad()
    def test_pre_forward(module, input):
        global prefetch_block_fw_id, fw_flag
        
        # print('------------------------------')
        # print(f'↓↓↓↓↓↓↓↓↓↓ bef {module.__class__.__name__} forward start {torch.cuda.memory_allocated() / (1024 * 1024)} MB')
        params_to_fetch = frozenset(iter_params(module))

        # print(module.__class__.__name__)
        # for name, param in module.named_parameters():
        #     print(param.size())
        # print()
        if not hasattr(module, 'ds_grads_remaining'):
            pass

        partitioned_params = []
        for param in params_to_fetch:
            # print('param.sb_id', param.sb_id, param.sb_shape)
            if param.status == ParamStatus.NO_READY:
                partitioned_params.append(param)
                # print(param.sb_id)
            elif param.status == ParamStatus.INFLIGHT:
                if is_new_param_async:
                    # print('wait param.sb_id', param.sb_id)
                    fetch_new_param_hindle_list[param.sb_id].wait()
                    del fetch_new_param_hindle_list[param.sb_id]
                else:
                    pass

        if partitioned_params:
            # print(module.id, 'not ready ', partitioned_params[0].sb_shape)
            handle = partitioned_params[0].all_gather_coalesced(partitioned_params)
            handle.wait()
            
            swap_persisted_params = [
                p for p in partitioned_params if p.ds_persist and p.manage.final_location == 'nvme'
            ]
            if swap_persisted_params:
                # print(swap_persisted_params)
                swap_persisted_params[0].nvme_swapper.remove_partition_and_release_buffers(swap_persisted_params)
            # handle.wait()
            import time
            # print(module.__class__.__name__)
        
        # for param in params_to_fetch:
        #     print(param)

        if not input[0].grad_fn and module.id == 0:
            fw_flag = True
            prefetch_block_fw_id = 0
        elif (module.__class__.__name__ == 'GPT2MLP' or module.__class__.__name__ == 'GPT2Attention' or module.__class__.__name__ == 'Linear') and input[0].grad_fn and fw_flag:
            fw_flag = False
            prefetch_block_fw_id = 0
            FIFO_buffer_gpu.clear()
        
        # print(input[0].grad_fn)
        # print('fw_flag', fw_flag)
        if is_new_param_async:
            index = 0
            flag = 0
            while index < len(FIFO_buffer_gpu):
                for params in FIFO_buffer_gpu[index]:
                    # print(params.sb_id, params.status)
                    if params.status == ParamStatus.READY and not fw_flag:
                        flag = 1
                        break
                    elif params.status == ParamStatus.NO_READY and fw_flag:
                        flag = 1
                        break
                if flag:
                    # 如果符合状态，弹出元素
                    element = FIFO_buffer_gpu.pop(index)
                else:
                    # 如果不符合状态，继续遍历下一个元素
                    index += 1
                flag = 0

            if fw_flag:
                prefetch_block = fw_prefetch_block
            else:
                prefetch_block = bw_prefetch_block

            # print(prefetch_block_length, prefetch_block_fw_id)
            while len(FIFO_buffer_gpu) < FIFO_buffer_gpu_length and prefetch_block_length > prefetch_block_fw_id and fw_flag:
                FIFO_buffer_gpu.append(prefetch_block[prefetch_block_fw_id])
                partitioned_params = []
                for params in prefetch_block[prefetch_block_fw_id]:
                    if params.status == ParamStatus.NO_READY:
                        # print(f'get {params.sb_id} {params.sb_shape}')
                        handle = params.all_gather_coalesced([params])
                        fetch_new_param_hindle_list[params.sb_id] = handle
                prefetch_block_fw_id += 1
            

    def _post_forward_module_hook(module, input, output):
        # print(output)
        global forward_time, prefetch_block_fw_id, global_id, swap_fw_flag
        if global_id == 1:
            nonlocal end_event
            end_event.record()
            end_event.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            forward_time += elapsed_time_ms
            # print(forward_time) 
            if len(fw_time):
                fw_time.pop(0)
                fw_time.append(forward_time)
            else:
                fw_time.append(forward_time)

        # if module.id == length_prefetch - 1:
        #     # print('forward_time', forward_time)
        #     forward_time = 0
        #     prefetch_block_fw_id = 0
        test_post_forward(module, input)

    @nvtx_wrap
    @torch.no_grad()
    def test_post_forward(module, input):
        # pass
        # print(f'↓↓↓↓↓↓↓↓↓↓ aft {module.__class__.__name__} forward start {torch.cuda.memory_allocated() / (1024 * 1024)} MB')
        # print(module.__class__.__name__)
        # print(input[0].grad_fn)
        global swap_fw_flag
        if swap_fw_flag and input[0].grad_fn == None:
            swap_fw_flag = False
        for name, param in module.named_parameters(recurse=False):
            if param.sb_id == 1 and input[0].grad_fn == None:
                swap_fw_flag = True

            # print(swap_fw_flag)
            if input[0].grad_fn == None or (is_swap_and_recompute and swap_fw_flag):
                # print(param.sb_id, '!!!!!!!!!!!')
                param.partition()
        # print(f'↑↑↑↑↑↑↑↑↑↑ aft {module.__class__.__name__} forward end {torch.cuda.memory_allocated() / (1024 * 1024)} MB')

    def _pre_backward_module_hook(module, inputs, output):
        
        def _run_before_backward_function(sub_module):
            if sub_module.applied_pre_backward_ref_cnt > 0:
                pre_sub_module_backward_function(sub_module)
                sub_module.applied_pre_backward_ref_cnt -= 1

        return _apply_to_tensors_only(module, PreBackwardFunction, _run_before_backward_function, output)

    @nvtx_wrap
    @torch.no_grad()
    def pre_sub_module_backward_function(module):
        # print(f'↓↓↓↓↓↓↓↓↓↓ bef {module.__class__.__name__} backward end{torch.cuda.memory_allocated() / (1024 * 1024)} MB')
        params_to_fetch = frozenset(iter_params(module))

        partitioned_params = []
        for param in params_to_fetch:
            if param.status == ParamStatus.NO_READY:
                partitioned_params.append(param)

        if partitioned_params:
            handle = partitioned_params[0].all_gather_coalesced(partitioned_params)
            swap_persisted_params = [
                p for p in partitioned_params if p.ds_persist and p.manage.final_location == 'nvme'
            ]
            if swap_persisted_params:
                # print(swap_persisted_params)
                swap_persisted_params[0].nvme_swapper.remove_partition_and_release_buffers(swap_persisted_params)
            handle.wait()
        # print(f'↑↑↑↑↑↑↑↑↑↑ bef {module.__class__.__name__} backward end{torch.cuda.memory_allocated() / (1024 * 1024)} MB')

    def _post_backward_module_hook(module, inputs):
        module.ds_grads_remaining = 0

        def _run_after_backward_function(sub_module):
            if sub_module.ds_grads_remaining == 0:
                post_sub_module_backward_function(sub_module)

        return _apply_to_tensors_only(module, PostBackwardFunction, _run_after_backward_function, inputs)

    @nvtx_wrap
    @torch.no_grad()
    def post_sub_module_backward_function(module):
        params_to_fetch = frozenset(iter_params(module))
        # for param in params_to_fetch:
        #     print(param.grad)
        # pass
        # print(f'↓↓↓↓↓↓↓↓↓↓ aft {module.__class__.__name__} backward end{torch.cuda.memory_allocated() / (1024 * 1024)} MB')
        for name, param in module.named_parameters(recurse=False):
            param.partition()
        # print(f'↑↑↑↑↑↑↑↑↑↑ aft {module.__class__.__name__} backward end{torch.cuda.memory_allocated() / (1024 * 1024)} MB')

    def _register_hooks_recursively(module, count=[0]):
        my_count = count[0]
        module.id = my_count
        global max_id
        prefetch_module.append(module)
        for name, param in module.named_parameters(recurse=False):
            prefetch_param.append(param)
            id_2_param[param.sb_id] = param
            if param.sb_id > max_id:
                max_id = param.sb_id
        for child in module.children():
            count[0] = count[0] + 1
            _register_hooks_recursively(child, count=count)


        forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))

        forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))

        backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))

        backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))
    
    def reverse_grouped_list(lst, k):
        # 先检查列表长度是否能被k整除
        if len(lst) % k != 0:
            return "Error: The list length must be divisible by k"
        
        # 将列表分为一组k个元素的子列表
        groups = [lst[i:i+k] for i in range(0, len(lst), k)]
        
        # 反转子列表的列表
        reversed_groups = groups[::-1]

        # 将子列表连结成一个列表
        reversed_lst = [item for sublist in reversed_groups for item in sublist]
        
        return reversed_lst

    def rearrange(lst):
        # 创建一个新的列表来存放重新排序后的元素
        new_lst = []

        # 检查列表长度是否可以被3整除
        if len(lst) % 3 != 0:
            print("Warning: list length is not divisible by 3. The last group may not be full of 3 elements for a full move.")

        # 每3个元素为一组，进行处理
        for i in range(0, len(lst), 3):
            # 如果当前组的元素数量不足3个，跳过
            if i + 2 >= len(lst):
                new_lst.extend(lst[i:])
                break
            # 将第一个元素放到第二个和第三个元素之后
            new_lst.extend([lst[i+1], lst[i+2], lst[i]])

        return new_lst

    prefetch_module = []
    prefetch_param = []

    forward_hooks = []
    backward_hooks = []

    num_next_module = 2
    creat_block_index = []
    param_id_to_hindle = {}

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    
    FIFO_buffer_gpu = [] 
    FIFO_buffer_gpu_length = 3
    fw_flag = True
    

    _register_hooks_recursively(model)
    see_memory_usage("after register hooks")

    length_prefetch = len(prefetch_module)

    fw_prefetch_block = []
    temp_block = []
    
    for sub_module in prefetch_module:
        params_to_fetch = frozenset(iter_params(sub_module))
        for param in params_to_fetch:
            if not param.ds_persist:

                if len(param.sb_shape) == 2:
                    if param.sb_shape[0] / param.sb_shape[1] == 4 or param.sb_shape[1] / param.sb_shape[0] == 4:
                        fw_prefetch_block.append([param])
                    elif param.sb_shape[0] / param.sb_shape[1] == 1 or param.sb_shape[0] / param.sb_shape[1] == 3:
                        temp_block.append(param)

                    if len(temp_block) == 2 and temp_block[0].sb_shape[0] / temp_block[1].sb_shape[0] == 3:
                        # print(temp_block[1].sb_shape[0])
                        # print(temp_block[0].sb_shape[0])
                        fw_prefetch_block.append(temp_block)
                        temp_block = []
    
    # print(len(fw_prefetch_block))
    # print('test fw_prefetch_block')
    # for i in fw_prefetch_block:
    #     print(len(i))
    #     if len(i) == 1:
    #         print(i[0].sb_shape, i[0].sb_id)
    #     else:
    #         print(i[0].sb_shape, i[0].sb_id, i[1].sb_shape, i[1].sb_id)   
    # print('----------------------')
    prefetch_block_length = len(fw_prefetch_block)
    bw_prefetch_block = reverse_grouped_list(fw_prefetch_block, 3)
    bw_prefetch_block = rearrange(bw_prefetch_block) 

    # for i in bw_prefetch_block:
    #     print(len(i))
    #     if len(i) == 1:
    #         print(i[0].sb_shape, i[0].sb_id)
    #     else:
    #         print(i[0].sb_shape, i[0].sb_id, i[1].sb_shape, i[1].sb_id)    
    # print('----------------------')
     
    for sub_module in prefetch_module:
        if sub_module.__class__.__name__ == 'GPT2Attention':
            creat_block_index.append(sub_module.id)
        elif sub_module.__class__.__name__ == 'GPT2MLP':
            creat_block_index.append(sub_module.id)

    temp_index = 0
    param_block = []
    for idx in creat_block_index:
        # print(idx)
        if idx == creat_block_index[-1]:
            # print('last', temp_index, idx)
            temp_block = []
            for i in range(temp_index, idx):
                params_to_fetch = frozenset(iter_params(prefetch_module[i]))
                for param in params_to_fetch:
                    temp_block.append(param)
            param_block.append(temp_block)
            temp_block = []
            temp_index = idx
            # print('last', temp_index, len(prefetch_module))
            for i in range(temp_index, len(prefetch_module)):
                params_to_fetch = frozenset(iter_params(prefetch_module[i]))
                for param in params_to_fetch:
                    temp_block.append(param)
            param_block.append(temp_block)

            break
        temp_block = []
        # print('normal', temp_index, idx)
        for i in range(temp_index, idx):
            params_to_fetch = frozenset(iter_params(prefetch_module[i]))
            for param in params_to_fetch:
                temp_block.append(param)

        temp_index = idx
        param_block.append(temp_block)

    param_block_reversed = list(reversed(param_block)) 


    print(len(forward_hooks))

