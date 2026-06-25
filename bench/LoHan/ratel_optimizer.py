import torch
from typing import List
from torch.nn import Parameter
from torch import Tensor
from see_mem import see_memory_usage
from logger import logger
from nvtx import nvtx_wrap
from typing import Deque, Dict, Tuple
import itertools
from debug import debug_param2name_id_shape
from ratel_init import *
from math import sqrt
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.multiprocessing as mp
from nvme_ds.pipelined_optimizer_swapper import PipelinedOptimizerSwapper
from nvme_ds.partitioned_optimizer_swapper import PartitionedOptimizerSwapper
bef_event = torch.cuda.Event()
class SB_optimizer(object):
    def __init__(self, 
                 init_optimizer, 
                 is_mp,
                 is_nvme,
                 is_grad_async,
                 is_nvme_async,
                 is_nvme_rearrange,
                 config,
                 mp_list = [],
                 sub_group_size=700000000, 
                 release_grad_bucket_size=5000000, 
                 contiguous_gradients=True):
        
        def json_object_hook(d): 
            return namedtuple('X', d.keys())(*d.values())
        with open(config) as f: 
            ds_config = json.load(f, object_hook=json_object_hook)

        self.offload_optimizer_config = ds_config.zero_config.offload_optimizer
        self.offload_param_config = ds_config.zero_config.offload_param
        self.aio_config = ds_config.aio_config
        self.optimizer = init_optimizer
        self.release_grad_bucket_size = int(release_grad_bucket_size)
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.is_nvme_rearrange = is_nvme_rearrange
        self.test_group = 0
        # 返回的数据结构[{'params':[]}]
        self.init_param_groups = self._get_parameter_groups()

        # 0--False | 1--True
        self.is_mp = is_mp
        self.is_nvme = is_nvme
        self.is_nvme_pipe = 1
        self.swap_optimizer = self.is_nvme
        self.is_nvme_async = is_nvme_async
        self.offload_optimizer_fast_init = 1
        self.mp_list = mp_list
        if is_grad_async:
            self.reduce_and_partition_stream = torch.cuda.Stream()
        else:
            self.reduce_and_partition_stream = torch.cuda.default_stream()
        self.loss_scale = 4096
        self.clip_grad = 0
        self.device = 'cpu'
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors
        self.offload_optimizer = True
        self.offload_optimizer_pin_memory = True
        self.sub_group_size = sub_group_size
        self.contiguous_gradients = contiguous_gradients        
        self.inf_or_nan_tracker: Tensor = torch.zeros(1,
                                                dtype=torch.bool,
                                                device='cuda:0',
                                                requires_grad=False)

        if self.offload_optimizer:
            self.norm_for_param_grads = {}
            self.local_overflow = False
        
        self.fp16_origin_param_groups = []
        self.fp16_manage_param_groups = []
        self.sub_group_to_group_id = {}

        # 参数按sub_group，对应于flat tensor的新的view
        self.fp16_param_groups_flat = []
        self.fp16_param_groups_flat_numel = []
        self.fp32_param_groups_flat = []
        #defragmented pinned memory
        self.param_groups_fp16_flat_cpu_memory = []

        self.params_in_ipg_bucket = []
        self.next_swappable_fp32_partitioned_groups = []

        # self.params_in_nvme_and_cpu = False
        self.offload_param = True
        self.nvme = False
        # 给origin的fp16参数做一个list，管理的fp16实际参数数据做一个list
        # 创建一个大的flat tenor管理数据, 组织到fp16_param_groups_flat
        self._create_fp16_partitions_with_defragmentation(self.init_param_groups)
        self.num_fp16_subgroups = len(self.fp16_param_groups_flat)


        if self.is_nvme:
            self._configure_tensor_swapping(self.offload_optimizer_config, self.aio_config)

        # origin参数地址 -> origin参数ID
        self.param_id = {}
        # origin参数ID -> origin参数本身
        self.param_dict = {}

        count = 0
        for i, params_group in enumerate(self.fp16_origin_param_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                count = count + 1

        # 创建fp32的参数副本，从fp16_param_groups_flat扩展到fp32_param_groups_flat
        # 初始化fp32_param_groups_flat的梯度，执行一次optimizer step，初始化优化器状态
        # 初始化fp16的参数梯度，保存在__param_id_to_grad_partition，用参数id访问
        self._setup_for_real_optimizer()

        # origin参数id -> 梯度position [group_id, offset, numel]
        self.grad_position = {}
        # 组织grad的position
        self.set_grad_positions()

        # 获取完整参数，并在参数的grad_fn之后挂上hook，用于释放梯度
        self.create_reduce_and_remove_grad_hooks()

        self.bf_i = len(self.fp32_param_groups_flat) - 1

    @nvtx_wrap
    def zero_grad(self, set_to_none=False):
        """
        Zero FP16 parameter grads.
        """

        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_origin_param_groups:
            for p in group:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def set_grad_positions(self):
        for i, group in enumerate(self.fp16_origin_param_groups):
            current_offset = 0
            for param in group:
                param_id = self.get_param_id(param)
                num_elements = param.manage.ds_numel
                self.grad_position[param_id] = [int(i), int(current_offset), int(num_elements)]
                #print(f"param id {param_id} i:{i}, manage {num_elements} numel {param.numel()}")
                current_offset += num_elements
        see_memory_usage(f"After Set Grad positions")

    def _get_parameter_groups(self):
        param_groups = []
        for param_group in self.optimizer.param_groups:
            params = {"params": [p for p in param_group["params"] if p.requires_grad]}
            param_groups.append(params)
        return param_groups

    def _create_fp16_partitions_with_defragmentation(self, init_param_groups):
        # 数据结构为([])
        param_groups: List[List[Parameter]] = tuple(self._create_fp16_sub_groups(param_group["params"]) for param_group in init_param_groups)
        
        # bookkeeping related to param groups
        for param_group_idx, param_group in enumerate(param_groups):
            for sub_group in param_group:
                sub_group_idx = len(self.fp16_origin_param_groups)
                
                # fp16的原始参数组，和manage后的参数组
                self.fp16_origin_param_groups.append(sub_group)
                self.fp16_manage_param_groups.append([param.manage for param in sub_group])

                # 将每个sub_group对应到参数组ID，通常只有一个参数组（0）
                self.sub_group_to_group_id[sub_group_idx] = param_group_idx

                # 记录每个sub group中参数总数
                self.fp16_param_groups_flat_numel.append(sum(param.manage.ds_numel for param in sub_group))

        # 根据实际param的大小创建pin住的CPU memory，返回到param_groups_fp16_flat_cpu_memory
        # 考虑nvme的场景，会根据max_in_cpu来控制cpu pin住的memory(max_in_cpu or all_param)
        # 是一维的empty创建的flat tensor
        self._create_param_groups_fp16_flat_cpu_memory()

        for param_group_idx, param_group in enumerate(param_groups):
            flat_offset = 0
            for i, sub_group in enumerate(param_group):
                total_elements = sum(p.manage.ds_numel for p in sub_group)

                #Flat buffer may not be available for parameters that reside in NVME
                # 
                # print(param_group_idx)
                # print(flat_offset + total_elements, '        ', self.param_groups_fp16_flat_cpu_memory[param_group_idx].numel())

                if not self.is_nvme or flat_offset + total_elements <= self.param_groups_fp16_flat_cpu_memory[param_group_idx].numel():
                    fp16_partitioned_group_flat = self.param_groups_fp16_flat_cpu_memory[param_group_idx].narrow(0, flat_offset, total_elements)
                elif self.is_nvme:
                    fp16_partitioned_group_flat = None
                else:
                    assert False, "Either params are in nvme, or they are in CPU memory. This code path should not be triggered. Please see you max_params_in_cpu and params_in_nvme configs"

                
                self.fp16_param_groups_flat.append(fp16_partitioned_group_flat)
                flat_offset += total_elements

                # 给如果fp16_partitioned_group_flat不是none 对flat tensor进行初始化
                self._move_to_flat_buffer(sub_group,
                                            fp16_partitioned_group_flat,
                                            avoid_copy=not self.offload_param)
        
        # 如果max_in_cpu不够存所有参数
        should_create_fp16_flat_reuse_buffer = any(flattened_partition_group is None
                                                   for flattened_partition_group in self.fp16_param_groups_flat)
        if should_create_fp16_flat_reuse_buffer and self.is_nvme:
            # print('has none')
            max_partition_numel, largest_partition_numel = 0, None
            for sub_group in self.fp16_origin_param_groups:
                total_elements = sum(t.manage.ds_numel for t in sub_group)
                if total_elements > max_partition_numel:
                    largest_partition_numel = [t.ds_numel for t in sub_group]
                    max_partition_numel = total_elements

            assert len(largest_partition_numel) > 0, f'Unexpected that largest partition is empty'
            self.fp16_origin_param_groups[0][0].nvme_swapper.reserve_partitioned_swap_space(largest_partition_numel)
    def _create_fp16_sub_groups(self, params_group):
        """根据sub_group_size, 将参数分配到sub group中

        Args:
            params_group (list): 初始化参数数组
        """
        all_params = sum([param.manage.ds_numel for param in params_group])

        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= all_params:
            return [params_group]

        sub_groups = []
        sub_group = []
        local_sub_group_size = 0
        for param in params_group:

            sub_group.append(param)
            local_sub_group_size += param.manage.ds_numel
            # print(param.manage.ds_numel)
            if local_sub_group_size >= sub_group_size or id(param) == id(params_group[-1]):
                # print('---------------------', sub_group)
                sub_groups.append(sub_group)

                sub_group = []
                local_sub_group_size = 0

        return sub_groups
    
    def _create_param_groups_fp16_flat_cpu_memory(self):

        aggregate_params_count = 0

        for j, param_group in enumerate(self.init_param_groups):
            params_in_group = sum([p.manage.ds_numel for p in param_group['params']])

            flat_buffer_size = params_in_group

            if self.is_nvme and \
                aggregate_params_count + params_in_group > self.offload_param_config.max_in_cpu:

                flat_buffer_size = max(0, self.offload_param_config.max_in_cpu - aggregate_params_count)

            aggregate_params_count += params_in_group

            if flat_buffer_size > 0:
                self.param_groups_fp16_flat_cpu_memory.append(torch.empty(int(flat_buffer_size), dtype=self.dtype).pin_memory())
            else:
                self.param_groups_fp16_flat_cpu_memory.append(torch.empty(1, dtype=self.dtype))


    def _configure_tensor_swapping(self, offload_optimizer_config, aio_config):
        nvme_swap_folder = os.path.join(offload_optimizer_config.nvme_path, 'zero_stage_3')
        os.makedirs(nvme_swap_folder, exist_ok=True)

        swapper_type = PipelinedOptimizerSwapper if self.is_nvme_pipe else PartitionedOptimizerSwapper

        self.optimizer_swapper = swapper_type(swap_config=offload_optimizer_config,
                                              aio_config=aio_config,
                                              base_folder=nvme_swap_folder,
                                              optimizer=self.optimizer,
                                              largest_numel=max(self.fp16_param_groups_flat_numel),
                                              device=self.device,
                                              dtype=torch.float32,
                                              timers=None)
        
    def _move_to_flat_buffer(self, param_list, flat_buffer, avoid_copy=False):

        if flat_buffer is None:
            # this dst buffer is on NVMe, so skip this
            return

        start = 0
        for param in param_list:
            src = param.manage
            dest = flat_buffer.narrow(0, start, src.ds_numel)
            start = start + src.ds_numel
            '''if the parameter was initialized in nvme then bring it to the destination buffer directly'''
            if self.is_nvme:
                if src.nvme_status == PartitionedParamStatus.NOT_AVAILABLE:
                    # print('swap_into_buffer')
                    param.nvme_swapper.swap_into_buffer(param, dest)
                    src.data = dest.data
                else:
                    assert src.nvme_status == PartitionedParamStatus.AVAILABLE, "Partitioned Param must be available here"
                    if not avoid_copy:
                        dest.data.copy_(src.data)
                    src.data = dest.data
                param.manage.final_location = 'not-nvme'
            else:
                if not avoid_copy:
                    dest.data.copy_(src.data)
                src.data = dest.data

    def create_reduce_and_remove_grad_hooks(self):
        print(f'[Begin] Create gradient reduction hooks')
        self.grad_accs = []
        for i, param_group in enumerate(self.fp16_origin_param_groups):
            for param in param_group:
                if param.requires_grad:
                    
                    # print(param)
                    # print(f"param grad fn {param.expand_as(param).grad_fn}")
                    # print(f'next_functions {param.expand_as(param).grad_fn.next_functions[0][0]}')
                    # param.all_gather()

                    handle = param.all_gather_coalesced([param])
                    handle.wait()
                    # print(param)
                    def wrapper(param, i):
                        
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)

                    wrapper(param, i)
                    # Partition the parameter after creating the hook
                    param.partition()

        print(f'[End] Create gradient reduction hooks')
    
    @nvtx_wrap
    def reduce_ready_partitions_and_remove_grads(self, param, i):
        # print(f"Backward {debug_param2name_id_shape(param)} -------------------------- param grad fn {param.expand_as(param).grad_fn}")
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)
    
    @property
    def elements_in_ipg_bucket(self):
        return sum(p.ds_numel for p in self.params_in_ipg_bucket)
    
    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.release_grad_bucket_size
        # see_memory_usage(
        #     f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}")
    
    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]
    
    @nvtx_wrap
    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.__reduce_and_partition_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        self.reduce_and_partition_stream.synchronize()

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):

        if self.elements_in_ipg_bucket > 0 and self.elements_in_ipg_bucket + param.ds_numel > self.release_grad_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.ds_numel)

            self.__reduce_and_partition_ipg_grads()

        self.__add_grad_to_ipg_bucket(param)
    
    @nvtx_wrap
    @torch.no_grad()
    def __add_grad_to_ipg_bucket(self, param: Parameter) -> None:
        self.reduce_and_partition_stream.wait_stream(torch.cuda.default_stream())

        if self.contiguous_gradients and self.elements_in_ipg_bucket + param.grad.numel() < self.release_grad_bucket_size:
            # move the gradient to a contiguous buffer
            with torch.cuda.stream(self.reduce_and_partition_stream):
                # move the parameter's gradient to the contiguous flat buffer
                # 发生在GPU上
                new_grad_tensor = self.__ipg_bucket_flat_buffer.narrow(0, self.elements_in_ipg_bucket,
                                                                       param.grad.numel()).view_as(param.grad)
                new_grad_tensor.copy_(param.grad, non_blocking=True)
                # print(torch.cuda.current_stream())
                param.grad.record_stream(torch.cuda.current_stream())
                param.grad.data = new_grad_tensor

        self.params_in_ipg_bucket.append(param)

    def __avg_scatter_grads(self, params_to_reduce: List[Parameter]) -> List[Tensor]:
        """average gradients and scatter partitions across ranks"""

        full_grads_for_rank = [p.grad for p in params_to_reduce]

        return full_grads_for_rank
    
    def __reduce_and_partition_ipg_grads(self, safe_mode: bool = False) -> None:
        if not self.params_in_ipg_bucket:
            return

        for param in self.params_in_ipg_bucket:
            if param.grad.numel() != param.ds_numel:
                raise RuntimeError(f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter "
                                   f"gradients whose size is not same as the params")

        self.params_in_ipg_bucket.sort(key=lambda p: p.sb_id)


        assert len(set(p.sb_id for p in self.params_in_ipg_bucket)) == len(self.params_in_ipg_bucket)

        # while self.param_reduce_events and self.param_reduce_events[0].query():
        #     self.param_reduce_events.popleft()
        # if len(self.param_reduce_events) > self.max_param_reduce_events:
        #     self.param_reduce_events.popleft().synchronize()

        with torch.cuda.stream(self.reduce_and_partition_stream):
            # if safe_mode:
            #     assert_ints_same_as_other_ranks([p.sb_id for p in self.params_in_ipg_bucket])

            grad_partitions = self.__avg_scatter_grads(self.params_in_ipg_bucket)
        self.partition_grads(self.params_in_ipg_bucket, grad_partitions)

        with torch.cuda.stream(self.reduce_and_partition_stream):
            self.params_in_ipg_bucket.clear()

            # event = get_accelerator().Event()
            # event.record()
            # self.param_reduce_events.append(event)
    @nvtx_wrap
    def _constant_buffered_norm2(self, input, buffer_size=250000000):
        norm = None
        for part in input.view(-1).split(buffer_size):
            if norm is None:
                norm = part.data.double().norm(2)**2.0
            else:
                norm += part.data.double().norm(2)**2.0
        return norm**0.5
    
    @nvtx_wrap
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        offload_fp32_gradients = {}
        offload_fp32_offsets = {}
        buffers = []
        global bef_event
        for param, grad_partition in zip(params_to_release, grad_partitions):

            with torch.cuda.stream(self.reduce_and_partition_stream):

                # grad_buffer = torch.empty_like(grad_partition.view(-1), requires_grad=True)
                grad_buffer = torch.empty_like(grad_partition.view(-1))
                grad_buffer.copy_(grad_partition.view(-1).detach())

                @nvtx_wrap
                def check_inf_or_nan():
                    if hasattr(self.inf_or_nan_tracker, "logical_or_"):
                        self.inf_or_nan_tracker.logical_or_(torch.isinf(grad_buffer).any())
                        self.inf_or_nan_tracker.logical_or_(torch.isnan(grad_buffer).any())
                    else:
                        # logical_or_ not available in older versions of pytorch
                        self.inf_or_nan_tracker += torch.isinf(grad_buffer).any()
                        self.inf_or_nan_tracker += torch.isnan(grad_buffer).any()
                        self.inf_or_nan_tracker = self.inf_or_nan_tracker > 0

                check_inf_or_nan()
            # offload the gradient partition if applicable
            if self.offload_optimizer:
                now_event = torch.cuda.Event(interprocess=True)
                with torch.cuda.stream(self.reduce_and_partition_stream):
                    i, dest_offset, _ = self.grad_position[self.get_param_id(param)]
                    # print(i)

                    self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)

                    if self._swappable_optimizer_subgroup(i) and self.is_nvme and not self.is_nvme_async:
                        if not i in offload_fp32_gradients.keys():
                            offload_fp32_gradients[i] = []
                            offload_fp32_offsets[i] = []

                        offload_fp32_gradients[i].append(grad_buffer.float())
                        offload_fp32_offsets[i].append(dest_offset)
                        # print('not copy here')
                    else:
                        # print(self.fp32_param_groups_flat[i].grad.is_pinned())
                        fp32_grad_tensor = self.fp32_param_groups_flat[i].grad.narrow(
                            0, dest_offset, grad_buffer.numel()).pin_memory()

                        fp32_grad_tensor.copy_(grad_buffer, non_blocking=True)
                        now_event.record(stream=self.reduce_and_partition_stream)


                @nvtx_wrap
                def test_single():
                    self.mp_list[0].put(self.fp32_param_groups_flat[self.bf_i])
                    self.mp_list[1].put(self.fp32_param_groups_flat[self.bf_i].grad)
                    # print(self.optimizer.state[self.fp32_param_groups_flat[self.bf_i]])
                    self.mp_list[3].put(self.optimizer.state[self.fp32_param_groups_flat[self.bf_i]]['step'])
                    self.mp_list[4].put(self.optimizer.state[self.fp32_param_groups_flat[self.bf_i]]['exp_avg'])
                    self.mp_list[5].put(self.optimizer.state[self.fp32_param_groups_flat[self.bf_i]]['exp_avg_sq'])
                    self.mp_list[6].put(self.bf_i)
                    self.mp_list[2].put(1)
                    print('main process put single')
                
                
                # print(grad_buffer.size())
                # print('dest_offset', dest_offset)
                # print(dest_offset + grad_buffer.numel())
                if self.bf_i != i:
                    # test_single()
                    if self.is_nvme_async:
                        self.single_step(self.bf_i, now_event)
                        pass
                        # torch.cuda.synchronize()
                        # import time
                        # time.sleep(0.5)
                    # if self.bf_i == 1 and i == 1:
                    #     self.bf_i = 0
                    #     self.independent_gradient_partition_epilogue()
                    # print('self.bf_i:', self.bf_i, 'i:', i)
                    self.test_group = 0

                
                # elif i == 1 and dest_offset + grad_buffer.numel() == 5242880:
                    
                #     if self.is_nvme_async:
                #         self.single_step(self.bf_i, now_event)
                #         pass
                #         # torch.cuda.synchronize()
                #         # import time
                #         # time.sleep(0.5)
                #     # if self.bf_i == 1 and i == 1:
                #     #     self.bf_i = 0
                #     #     self.independent_gradient_partition_epilogue()
                #     print('self.bf_i:', self.bf_i, 'i:', i)

                #     self.test_group = 0
                
                if i == 0 and dest_offset == 0:
                    if self.is_nvme_async:
                        self.single_step(i, now_event)
                        pass
                    # print('self.bf_i:', self.bf_i, 'i:', i)
                else:
                    self.test_group += self.elements_in_ipg_bucket
                
                bef_event = now_event
                if i == 0 and dest_offset == 0:
                    self.bf_i = len(self.fp32_param_groups_flat) - 1
                else:
                    self.bf_i = i
            # free the gradient
            with torch.cuda.stream(self.reduce_and_partition_stream):
                param.grad.record_stream(torch.cuda.current_stream())
                param.grad = None
            # print(f'---------------------{param.sb_shape}---------------------')
            # print(fp32_grad_tensor)
                @nvtx_wrap
                def test_swap_out_grad():
                    if self.offload_optimizer and self.swap_optimizer and self.is_nvme and not self.is_nvme_async:
                        for i in offload_fp32_gradients.keys():
                            self.optimizer_swapper.swap_out_gradients(parameter=self.fp32_param_groups_flat[i],
                                                                    gradient_offsets=offload_fp32_offsets[i],
                                                                    gradient_tensors=offload_fp32_gradients[i])
                test_swap_out_grad()
        return buffers

    def _setup_for_real_optimizer(self):
        see_memory_usage("Before creating fp32 partitions", force=True)
        self._create_fp32_partitions()
        see_memory_usage("After creating fp32 partitions", force=True)

        # # To support pipelined optimizer swapping
        self._create_next_swappable_fp32_groups()

        see_memory_usage("Before initializing optimizer states", force=True)

        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)

        logger.info(f"optimizer state initialized")

        # IPG
        if self.contiguous_gradients:
            self.__ipg_bucket_flat_buffer: Tensor = torch.empty(self.release_grad_bucket_size,
                                                                dtype=self.dtype,
                                                                device='cuda:0')
        # grad_partitions_flat_buffer = None
        # self.__param_id_to_grad_partition: Dict[int, Tensor] = {}

        # all_params = list(itertools.chain.from_iterable(self.fp16_origin_param_groups))

        # grad_partitions_flat_buffer: Tensor = torch.zeros(sum(p.manage.ds_numel for p in all_params),
        #                                                   dtype=self.dtype,
        #                                                   device=self.device)
        # if self.offload_optimizer_pin_memory:
        #     grad_partitions_flat_buffer = grad_partitions_flat_buffer.pin_memory()

        # offset = 0
        # for param in all_params:
        #     self.__param_id_to_grad_partition[param.sb_id] = grad_partitions_flat_buffer.narrow(
        #         0, offset, param.manage.ds_numel)
        #     offset += param.manage.ds_numel

    def _get_sub_group_partitions(self, sub_group_id):
        sub_group_partitions = []
        for param, manage_param in zip(self.fp16_origin_param_groups[sub_group_id],
                                            self.fp16_manage_param_groups[sub_group_id]):
            if manage_param.nvme_status == PartitionedParamStatus.NOT_AVAILABLE:
                swap_path = param.nvme_swapper.get_path(param, True)
                sub_group_partitions.append((manage_param, param.partition_numel(), swap_path))
            else:
                sub_group_partitions.append((manage_param, manage_param.ds_numel, None))
        # for i in sub_group_partitions:
        #     print(i[1])
        return sub_group_partitions
    
    # def _swap_in_sub_group_to_flat_buffer(self, flat_buffer, sub_group_id):
    #     offset = 0
    #     elements_in_sub_group = sum([t.ds_numel for t in self.fp16_param_groups_flat[sub_group_id]])
    #     assert (flat_buffer.numel() == elements_in_sub_group)
    #     for param, partitioned_param in zip(self.fp16_groups[sub_group_id],
    #                                         self.fp16_param_groups_flat[sub_group_id]):
    #         dest = flat_buffer.narrow(0, offset, partitioned_param.ds_numel)
    #         if partitioned_param.status == PartitionedParamStatus.NOT_AVAILABLE:
    #             # print_rank_0(
    #             #     f"Swapping in {param.ds_id} with elements {param.ds_numel} and partition {param.partition_numel()}"
    #             # )
    #             param.nvme_swapper.swap_in([param], async_op=False)
    #             dest.data.copy_(partitioned_param.data)
    #             param.nvme_swapper.remove_partition_and_release_buffers([param])
    #             # print_rank_0(f"Swapping in {param.ds_id} done")
    #         else:
    #             dest.data.copy_(partitioned_param.data)
    #         offset += partitioned_param.ds_numel
    def _create_fp32_partitions(self):
        cpu_memory_usage = 0
        cpu_memory_sub_groups = 0
        nvme_memory_usage = 0
        GIGA_BYTES = (1024**3)

        num_swappable_partitions = 0
        num_swap_from_nvme_partitions = 0
        num_swap_from_cpu_partitions = 0
        swap_from_nvme_memory_usage = 0
        swap_from_cpu_memory_usage = 0
        swappable_fp32_tensors = []
        swappable_fp16_src_tensors = []
        nvme_fp16_partitions_info = []
        nvme_fp16_num_elems = []
        nvme_fp32_dest_tensors = []
        fp32_element_size = torch.tensor([], dtype=torch.float32).element_size()

        # for i, tensor in enumerate(self.fp16_param_groups_flat):
        #     print(self._swappable_optimizer_subgroup(i), self.fp16_param_groups_flat_numel[i])
        for i, tensor in enumerate(self.fp16_param_groups_flat):
            num_elements = self.fp16_param_groups_flat_numel[i]

            # a partition of the fp32 master weights that will be updated by this process
            # print(self._swappable_optimizer_subgroup(i))
            if self._swappable_optimizer_subgroup(i) and self.is_nvme:
                self.fp32_param_groups_flat.append(torch.Tensor())
                nvme_memory_usage += (fp32_element_size * num_elements)
                num_swappable_partitions += 1

                if self.is_nvme and tensor is None:
                    num_swap_from_nvme_partitions += 1
                    swap_from_nvme_memory_usage += (fp32_element_size * num_elements)
                    if self.offload_optimizer_fast_init:
                        # 获取到一个tuple，(manage参数， 元素个数，swap的path)
                        sub_group_partitions = self._get_sub_group_partitions(i)
                        nvme_fp16_partitions_info.append(sub_group_partitions)
                        nvme_fp16_num_elems.append(num_elements)
                        nvme_fp32_dest_tensors.append(self.fp32_param_groups_flat[i])
                    # else:
                    #     unpinned_fp32_buffer = torch.empty(num_elements, device=self.device, dtype=torch.float)
                    #     self._swap_in_sub_group_to_flat_buffer(unpinned_fp32_buffer, i)
                    #     self.optimizer_swapper.initialize_parameters(parameters=[self.fp32_param_groups_flat[i]],
                    #                                                  src_tensors=[unpinned_fp32_buffer])
                else:
                    num_swap_from_cpu_partitions += 1
                    swap_from_cpu_memory_usage += (fp32_element_size * num_elements)
                    swappable_fp32_tensors.append(self.fp32_param_groups_flat[i])
                    swappable_fp16_src_tensors.append(self.fp16_param_groups_flat[i])
            else:
                cpu_memory_usage += (fp32_element_size * num_elements)
                cpu_memory_sub_groups += 1
                if self.is_nvme and tensor is None:
                    unpinned_fp32_buffer = torch.empty(num_elements, device=self.device, dtype=torch.float)
                    self._swap_in_sub_group_to_flat_buffer(unpinned_fp32_buffer, i)
                    self.fp32_param_groups_flat.append(unpinned_fp32_buffer)
                else:
                    self.fp32_param_groups_flat.append(self.fp16_param_groups_flat[i].to(self.device).clone().float().detach())


            self.fp32_param_groups_flat[i].requires_grad = True  # keep this in case internal optimizer uses it

        if len(swappable_fp32_tensors) > 0 and self.is_nvme:
            # print('init 1')
            self.optimizer_swapper.initialize_parameters(parameters=swappable_fp32_tensors,
                                                         src_tensors=swappable_fp16_src_tensors)

        if len(nvme_fp32_dest_tensors) > 0 and self.is_nvme:
            # print('init 2')
            fp16_pinned_buffers = self.fp16_origin_param_groups[0][0].nvme_swapper.reserve_available_buffers()
            assert len(fp16_pinned_buffers) > 0
            self.optimizer_swapper.initialize_from_swapped_fp16_params(fp16_partitions_info=nvme_fp16_partitions_info,
                                                                       fp16_num_elems=nvme_fp16_num_elems,
                                                                       fp16_pinned_buffers=fp16_pinned_buffers,
                                                                       fp32_parameters=nvme_fp32_dest_tensors)
            self.fp16_origin_param_groups[0][0].nvme_swapper.release_reserved_buffers()
            
        cpu_memory_gigabytes = cpu_memory_usage / GIGA_BYTES
        # print(f'In-Memory FP32 Partitions: count={cpu_memory_sub_groups} size={cpu_memory_gigabytes:5.2f} GB')

        # Clear for on-the-fly population before the optimizer step
        for param_group in self.optimizer.param_groups:
            param_group['params'] = []

    def _create_next_swappable_fp32_groups(self):
        reverse_order_indices = [i for i in range(len(self.fp32_param_groups_flat))]
        reverse_order_indices.reverse()

        next_group = None
        for i in reverse_order_indices:
            self.next_swappable_fp32_partitioned_groups.append(next_group)
            if self._swappable_optimizer_subgroup(i):
                next_group = self.fp32_param_groups_flat[i]

        self.next_swappable_fp32_partitioned_groups.reverse()
    
    def _swappable_optimizer_subgroup(self, sub_group_id):
        if not self.swap_optimizer:
            return False

        return self.optimizer_swapper.swappable_tensor(None,
                                                       numel=self.fp16_param_groups_flat_numel[sub_group_id])
    @nvtx_wrap
    def _optimizer_states_and_gradient_swap_in(self, sub_group_id):
        param_length = self.fp16_param_groups_flat_numel[sub_group_id]
        fp32_param_id = id(self.fp32_param_groups_flat[sub_group_id])
        assert self._swappable_optimizer_subgroup(sub_group_id), \
            f'Parameter {fp32_param_id} of numel={param_length} is not swappable'

        # print('!!!!!!!!!! swap_in ', sub_group_id)
        see_memory_usage(f'pre-step Before swapping in optimizer tensors {sub_group_id}', force=False)
        self.optimizer_swapper.swap_in_optimizer_state(
            parameter=self.fp32_param_groups_flat[sub_group_id],
            # async_parameter=self.next_swappable_fp32_partitioned_groups[sub_group_id])
            async_parameter=None)
        see_memory_usage(f'pre-step After swapping in optimizer tensors {sub_group_id}', force=False)

    @nvtx_wrap
    def _optimizer_states_and_gradient_swap_in_new(self, sub_group_id, is_first):
        param_length = self.fp16_param_groups_flat_numel[sub_group_id]
        fp32_param_id = id(self.fp32_param_groups_flat[sub_group_id])
        assert self._swappable_optimizer_subgroup(sub_group_id), \
            f'Parameter {fp32_param_id} of numel={param_length} is not swappable'

        # print('!!!!!!!!!!swap in', sub_group_id)
        see_memory_usage(f'pre-step Before swapping in optimizer tensors {sub_group_id}', force=False)
        # print('bef swap_in_optimizer_state_new', self.fp32_param_groups_flat[sub_group_id].grad.is_pinned())
        self.optimizer_swapper.swap_in_optimizer_state_new(
            parameter=self.fp32_param_groups_flat[sub_group_id],
            async_parameter=self.next_swappable_fp32_partitioned_groups[sub_group_id],
            is_first=is_first
            )
        # print('aft swap_in_optimizer_state_new', self.fp32_param_groups_flat[sub_group_id].grad.is_pinned())
        see_memory_usage(f'pre-step After swapping in optimizer tensors {sub_group_id}', force=False)
    def _partitioned_params_swap_out(self, i):
        offset = 0
        fp32_param = self.fp32_param_groups_flat[i]
        assert fp32_param is not None, \
        f'fp32 parameters of sub_group {i} is None'

        @nvtx_wrap
        def test():
            pass
        swap_fp16_params = []
        swap_fp32_params = []
        for param, manage_param in zip(self.fp16_origin_param_groups[i], self.fp16_manage_param_groups[i]):
            src = fp32_param.narrow(0, offset, manage_param.ds_numel)
            if manage_param.nvme_status == PartitionedParamStatus.AVAILABLE:
                test()
                manage_param.data.copy_(src.data, non_blocking = True)
                test()
            else:
                swap_fp32_params.append(src)
                swap_fp16_params.append(param)
            offset += manage_param.ds_numel

        if len(swap_fp16_params):
            swap_fp16_params[0].nvme_swapper.swap_out_partitioned_params(dst_fp16_params=swap_fp16_params,
                                                                         src_fp32_params=swap_fp32_params)

    def _optimizer_states_and_gradient_swap_out(self, sub_group_id):
        param_length = self.fp16_param_groups_flat_numel[sub_group_id]
        fp32_param_id = id(self.fp32_param_groups_flat[sub_group_id])
        assert self._swappable_optimizer_subgroup(sub_group_id), \
            f'Parameter {fp32_param_id} of numel={param_length} is not swappable'

        see_memory_usage(f'post-step Before swapping out optimizer tensors {sub_group_id}', force=False)

        self.optimizer_swapper.swap_out_optimizer_state(
            parameter=self.fp32_param_groups_flat[sub_group_id],
            async_swap=self.next_swappable_fp32_partitioned_groups[sub_group_id] is not None)

        see_memory_usage(f'post-step After swapping out optimizer tensors {sub_group_id}', force=False)

        # get rid of the fp32 gradients. Not needed anymore
        if not self.is_nvme_async:
            self.fp32_param_groups_flat[sub_group_id].grad = None

    def initialize_optimizer_states(self):
        num_subgroups = len(self.fp16_origin_param_groups)

        largest_numel = max([sum([p.ds_numel for p in psg]) for psg in self.fp16_manage_param_groups])
        gradient_dtype = self.fp32_param_groups_flat[0].dtype
        gradient_buffer = torch.zeros(int(largest_numel), dtype=gradient_dtype, device=self.device, pin_memory=True)



        for i, group in enumerate(self.fp16_origin_param_groups):
            swappable_optimizer_subgroup = self._swappable_optimizer_subgroup(i)
            swappable_param_subgroup = self.fp16_param_groups_flat[i] is None
            num_elements = int(self.fp16_param_groups_flat_numel[i])

            see_memory_usage(
                f'[Begin] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}')

            # for p in [self.fp32_param_groups_flat[i]]:
            #     print('run single step')
            #     state = self.optimizer.state[p]
            #     if state:
            #         print('bef ini swap in param state_m', state['exp_avg'].size())
            #         print('bef ini swap in param state_v', state['exp_avg_sq'].size())
            #     print('bef ini swap in param param', p.size())
            #     if p.grad:
            #         print('bef ini swap in param grad', p.grad.size())

            if swappable_optimizer_subgroup and self.is_nvme:
                self._optimizer_states_and_gradient_swap_in(i)
            # if swappable_optimizer_subgroup:
            #     self._optimizer_states_and_gradient_swap_in(i, timer_names)

            # if True:
            if self.offload_optimizer and not swappable_optimizer_subgroup:
                subgroup_gradient_buffer = torch.zeros(num_elements, dtype=gradient_dtype, device=self.device)
                if self.offload_optimizer_pin_memory:
                    subgroup_gradient_buffer = subgroup_gradient_buffer.pin_memory()

                self.fp32_param_groups_flat[i].grad = subgroup_gradient_buffer
                print('!!!!!!!!!!!')
            else:
                self.fp32_param_groups_flat[i].grad = gradient_buffer.narrow(0, 0, num_elements).pin_memory()
                print('@@@@@@@@@')

            # Initialize the optimizer states with the flattended fp32 partition.



            @nvtx_wrap
            def test_put_fp32():
                self.fp32_param_groups_flat[i].share_memory_()
                self.fp32_param_groups_flat[i].grad.share_memory_()
                # self.mp_list[0].put(self.fp32_param_groups_flat[i])
                # self.mp_list[1].put(self.fp32_param_groups_flat[i].grad)

            if self.is_mp and False:
                test_put_fp32()

            self._optimizer_step(i)
            fp32_param = self.fp32_param_groups_flat[i]
            state = self.optimizer.state[fp32_param]

            if state and self.is_mp and False:
            # state['step'].share_memory_()
                @nvtx_wrap
                def test_put_state():
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
                
                test_put_state()
                # print("state['exp_avg'].is_shared()", state['exp_avg'].is_shared())
                # print("state['exp_avg_sq'].is_shared()", state['exp_avg_sq'].is_shared())
                # print(f"step = {state['step']}\nexp_avg_size = {state['exp_avg'].size()}\nexp_avg_sq_size = {state['exp_avg_sq'].size()}")

            if swappable_param_subgroup and self.is_nvme:
                self._partitioned_params_swap_out(i)
            if swappable_optimizer_subgroup and self.is_nvme:
                self._optimizer_states_and_gradient_swap_out(i)
            
            # print(state['exp_avg'].is_pinned(), state['exp_avg_sq'].is_pinned())
            see_memory_usage('1')
            print(state['exp_avg'].size(), state['exp_avg_sq'].size())
            state['exp_avg'].data  = torch.Tensor()
            state['exp_avg_sq'].data = torch.Tensor()
            print(state['exp_avg'].size(), state['exp_avg_sq'].size())
            see_memory_usage('2')

            see_memory_usage(
                f'[End] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}')


        if not self.offload_optimizer:
            for group in self.fp32_param_groups_flat:
                # print('group.grad = None')
                group.grad = None

        # for p in [self.fp32_param_groups_flat[i]]:
        #     print('run single step')
        #     state = self.optimizer.state[p]
        #     if state:
        #         print('aft reset state_m', state['exp_avg'].size())
        #         print('aft reset state_v', state['exp_avg_sq'].size())
        #     print('aft reset param', p.size())
        #     if not p.grad is None:
        #         print('aft reset grad', p.grad.size())

        # Reset steps
        return
    
    @nvtx_wrap
    def _optimizer_step(self, sub_group_id):
        param_group_id = self.sub_group_to_group_id[sub_group_id]
        fp32_param = self.fp32_param_groups_flat[sub_group_id]
        self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            # if state:
            #     # state['step'].share_memory_()
            #     @nvtx_wrap
            #     def test_put_state():
            #         state['exp_avg'].share_memory_()
            #         state['exp_avg_sq'].share_memory_()
                
            #     test_put_state()
            #     print("state['exp_avg'].is_shared()", state['exp_avg'].is_shared())
            #     print("state['exp_avg_sq'].is_shared()", state['exp_avg_sq'].is_shared())
            #     print(f"step = {state['step']}\nexp_avg_size = {state['exp_avg'].size()}\nexp_avg_sq_size = {state['exp_avg_sq'].size()}")
        self.optimizer.step()
        # see_memory_usage('BEF release')
        self.optimizer.param_groups[param_group_id]['params'] = []
        # see_memory_usage('AFT release')

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            param_id = self.get_param_id(p)
            if param_id in self.norm_for_param_grads.keys():
                param_norm = self.norm_for_param_grads[param_id]
                total_norm += param_norm.item()**2

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm
    
    def _get_norm_groups(self):
        norm_groups = []
        for i, group in enumerate(self.fp16_origin_param_groups):
            if self.offload_optimizer:
                norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.fp16_origin_param_groups[i]))
        return norm_groups
    
    def get_global_norm(self, norm_list):
        """ Compute total from a list of norms
        """
        total_norm = 0.0
        for norm in norm_list:
            total_norm += norm**2.0

        return sqrt(total_norm)

    @nvtx_wrap
    def step(self, closure=None):
        """
        Not supporting closure.
        """

        norm_groups = self._get_norm_groups()
        scaled_global_grad_norm = self.get_global_norm(norm_groups)

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale



        #update parameters one sub group at a time
        for sub_group_id, group in enumerate(self.fp16_origin_param_groups):

            #prepare optimizer states, gradients and fp32 parameters for update
            self._prepare_sub_group(sub_group_id)

            #scale the fp32 gradients
            # self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

            #apply the optimizer step on the sub group and copy fp32 parameters to fp16
            self._optimizer_step(sub_group_id)

            #put fp16 parameters in appropriate location
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)

            #release memory or swap out optimizer states of fp32 parameters
            self._release_sub_group(sub_group_id)
    
    @nvtx_wrap
    def single_step_origin(self, sub_group_id, event):


        self._prepare_sub_group(sub_group_id)

        self._optimizer_step(sub_group_id)

        #put fp16 parameters in appropriate location
        self._reassign_or_swap_out_partitioned_parameters(sub_group_id)

        #release memory or swap out optimizer states of fp32 parameters
        self._release_sub_group(sub_group_id)

    
    @nvtx_wrap
    def single_step(self, sub_group_id, event):

        if self.is_nvme_rearrange:
            # print('single_step 1', self.fp32_param_groups_flat[sub_group_id].grad.is_pinned())
            # print('single_step', sub_group_id)
            if sub_group_id == len(self.fp16_origin_param_groups) - 1:
                self._optimizer_states_and_gradient_swap_in_new(sub_group_id, True)
            else:
                self._optimizer_states_and_gradient_swap_in_new(sub_group_id, False)
            
            # event.synchronize()
            if not self.is_mp :
                self._optimizer_step(sub_group_id)
            else:
                param_group_id = self.sub_group_to_group_id[sub_group_id]
                fp32_param = self.fp32_param_groups_flat[sub_group_id]
                self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]

                @nvtx_wrap
                def transfer_mul_process():
                    if self.is_mp:
                        self.mp_list[0].put(fp32_param)
                        self.mp_list[1].put(fp32_param.grad)
                        # print(self.optimizer.state[fp32_param])
                        self.mp_list[3].put(self.optimizer.state[fp32_param]['step'])
                        self.mp_list[4].put(self.optimizer.state[fp32_param]['exp_avg'])
                        self.mp_list[5].put(self.optimizer.state[fp32_param]['exp_avg_sq'])
                        self.mp_list[6].put(self.bf_i)
                        self.mp_list[7].put(event)
                        self.mp_list[2].put(1)
                        # print('main process put single')

                transfer_mul_process()
            if sub_group_id != len(self.fp16_origin_param_groups) - 1:
                # print('realse ', sub_group_id + 1)
                self._reassign_or_swap_out_partitioned_parameters(sub_group_id + 1)
                self._release_sub_group_new(sub_group_id + 1, False)

            if sub_group_id ==  0:
                self._reassign_or_swap_out_partitioned_parameters(sub_group_id)

                self._release_sub_group_new(sub_group_id, True)

        else:
            torch.cuda.synchronize()
            self._optimizer_states_and_gradient_swap_in(sub_group_id)
            event.synchronize()
            self._optimizer_step(sub_group_id)
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            self._release_sub_group(sub_group_id)






    @nvtx_wrap
    def _prepare_sub_group(self, sub_group_id):
        # pass
        if self._swappable_optimizer_subgroup(sub_group_id):
            self._optimizer_states_and_gradient_swap_in(sub_group_id)
        # elif not self.offload_optimizer:
        #     self._prepare_fp32_grad_for_sub_group(sub_group_id)
    
    @nvtx_wrap
    def unscale_and_clip_grads(self, sub_group_id, total_norm):
        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        self.fp32_param_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)
    
    @nvtx_wrap
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.fp16_param_groups_flat[sub_group_id] is not None:
            self.fp16_param_groups_flat[sub_group_id].data.copy_(
                self.fp32_param_groups_flat[sub_group_id].data)

            #unflatten fp16 parameter subgroup
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)
    
    def _unflatten_partitioned_parameters(self, sub_group_id):
        updated_params = self.unflatten(self.fp16_param_groups_flat[sub_group_id],
                                        self.fp16_manage_param_groups[sub_group_id])

        for partitioned_param, q in zip(self.fp16_manage_param_groups[sub_group_id], updated_params):
            partitioned_param.data = q.data
    
    @nvtx_wrap
    def _release_sub_group(self, sub_group_id):
        # pass
        # get rid of the fp32 gradients. Not needed anymore
        if not self.offload_optimizer:
            self.fp32_param_groups_flat[sub_group_id].grad = None

        if self._swappable_optimizer_subgroup(sub_group_id):
            self._optimizer_states_and_gradient_swap_out(sub_group_id)
    
    @nvtx_wrap
    def _release_sub_group_new(self, sub_group_id, is_last):
        # pass
        # get rid of the fp32 gradients. Not needed anymore

        # print('@@@@@@@@@@@@swap out', sub_group_id)
        if not self.offload_optimizer:
            self.fp32_param_groups_flat[sub_group_id].grad = None

        if self._swappable_optimizer_subgroup(sub_group_id):
            self._optimizer_states_and_gradient_swap_out_new(sub_group_id, is_last=is_last)

    def _optimizer_states_and_gradient_swap_out_new(self, sub_group_id, is_last):
        param_length = self.fp16_param_groups_flat_numel[sub_group_id]
        fp32_param_id = id(self.fp32_param_groups_flat[sub_group_id])
        assert self._swappable_optimizer_subgroup(sub_group_id), \
            f'Parameter {fp32_param_id} of numel={param_length} is not swappable'

        see_memory_usage(f'post-step Before swapping out optimizer tensors {sub_group_id}', force=False)

        self.optimizer_swapper.swap_out_optimizer_state_new(
            parameter=self.fp32_param_groups_flat[sub_group_id],
            async_swap=self.next_swappable_fp32_partitioned_groups[sub_group_id] is not None,
            is_last=is_last)

        see_memory_usage(f'post-step After swapping out optimizer tensors {sub_group_id}', force=False)

        # get rid of the fp32 gradients. Not needed anymore

        # self.fp32_param_groups_flat[sub_group_id].grad = None