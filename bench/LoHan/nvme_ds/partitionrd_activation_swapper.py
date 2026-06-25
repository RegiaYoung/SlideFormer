import os
import shutil
from enum import Enum
import torch
# from deepspeed import comm as dist
from op_ds.accelerator import get_accelerator
from op_ds.ops.op_builder.async_io import AsyncIOBuilder
# from deepspeed.ops.op_builder import AsyncIOBuilder
from nvme_ds.constants import *
from nvme_ds.utils import swap_in_tensors, swap_out_tensors, MIN_AIO_BYTES, AIO_ALIGNED_BYTES, print_object, SwapBufferPool
from nvtx import nvtx_wrap

def print_rank_0(message, debug=False, force=False):
    print(message)


class PartitionedActStatus(Enum):
    # cpu
    AVAILABLE = 1

    # nvme
    NOT_AVAILABLE = 2

    # 
    INFLIGHT = 3

class AsyncPartitionedActivationSwapper(object):


    def __init__(self, ds_config, model_dtype):
        aio_op = AsyncIOBuilder().load(verbose=False)
        self.aio_handle = aio_op.aio_handle
        self.dtype = model_dtype

        #set swap buffers, create aio handles
        self._configure_aio(ds_config)

        #mapping from act id to path
        self.id_to_path = {}

        #mapping from act_id to buffer id
        self.act_id_to_buffer_id = {}

        # mapping from param_id to swap buffer
        self.act_id_to_swap_buffer = {}

        #number of elements in the act
        self.act_id_to_numel = {}

        self.pending_writes = 0
        self.pending_reads = 0

        #keep track of async swap in params and buffers
        self.inflight_acts = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

        #keep track of available params
        self.available_acts = set()
        self.available_numel = 0

        # for swapping out from partitioned fp32 params
        self.partitioned_swap_buffer = None
        self.partitioned_swap_pool = None

        self.invalid_buffer = torch.tensor(1).half()

        exclude_list = ['aio_read_handle', 'aio_write_handle', 'buffers']
        print_object(obj=self, name='AsyncPartitionedActivationSwapper', exclude_list=exclude_list)

    def available_swap_in_buffers(self):
        return len(self.available_buffer_ids)
    
    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_act
        torch_dtype_string = str(self.dtype).split(".")[1]
        self.swap_folder = os.path.join(self.swap_config.nvme_path, 'zero_stage_3', f'{torch_dtype_string}act')
        shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)

        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.aio_config = ds_config.aio_config

        # Read/Write alignment for each thread during Intra-request parallelism
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config.block_size)
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config.thread_count
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        self.elements_per_buffer = self.swap_config.buffer_size
        self.aligned_elements_per_buffer = self._io_aligned_numel(self.elements_per_buffer)
        self.param_buffer_count = self.swap_config.buffer_count

        self.available_buffer_ids = [i for i in range(self.param_buffer_count)]
        self.reserved_buffer_ids = []
        self.buffers = get_accelerator().pin_memory(
            torch.empty(int(self.aligned_elements_per_buffer * self.param_buffer_count),
                        dtype=self.dtype,
                        requires_grad=False))


        self.aio_read_handle = self.aio_handle(self.aio_config.block_size, self.aio_config.queue_depth,
                                               self.aio_config.single_submit, self.aio_config.overlap_events,
                                               self.aio_config.thread_count)

        self.aio_write_handle = self.aio_handle(self.aio_config.block_size, self.aio_config.queue_depth,
                                                self.aio_config.single_submit,
                                                self.aio_config.overlap_events, self.aio_config.thread_count)

        self.swap_out_acts = []

        #Check if partitioned param or numel in a tensor is swappable or not
    def swappable_tensor(self, act_block=None, numel=None):
        if act_block is not None:
            assert numel is None, "Both parma and numel cannot be provided"
            numel = act_block.sb_numel
        if numel is not None:
            # print(self.min_aio_bytes, numel * self.swap_element_size)
            return self.min_aio_bytes <= numel * self.swap_element_size
        assert False, "Either act_block or numel must be provided"

    def get_path(self, act_block):
        paths = self._get_swap_paths(act_block)
        return paths[0]

    def _get_swap_paths(self, act_block):
        paths = []
        act_id = act_block.id
        if act_id in self.id_to_path.keys():
            param_path = self.id_to_path[act_id]
        else:
            param_path = os.path.join(self.swap_folder, f'{act_id}_act.tensor.swp')

            self.id_to_path[act_id] = param_path
        paths.append(param_path)

        return paths

    def _get_swap_buffers(self, act_block):
        buffers = []
        act_id = act_block.id
        assert act_id in self.act_id_to_swap_buffer.keys(), \
        f'param {act_id} has not been assigned a swap buffer'
        buffers.append(self.act_id_to_swap_buffer[act_id])

        return buffers

    def _track_numel(self, act_block):

        assert act_block.manage is not None, "Partitioned tensor is None"
        self.act_id_to_numel[act_block.id] = act_block.sb_numel

    def _allocate_and_return_buffers_for_swap_in(self, act_block):
        compute_buffers = []
        swap_buffers = []
        act_id = act_block.id
        
        assert act_id in self.act_id_to_numel.keys(), f" Number of elements in param {act_id} is unknown"
        assert act_id not in self.act_id_to_buffer_id.keys(
        ), f"param {act_id} already assigned swap buffer id {self.act_id_to_buffer_id[act_id]}"
        assert act_id not in self.act_id_to_swap_buffer.keys(
        ), f"param {act_id} has already been assigned a swap buffer"

        buffer_id = self.available_buffer_ids.pop()
        self.act_id_to_buffer_id[act_id] = buffer_id
        aligned_swap_numel = self._io_aligned_numel(self.act_id_to_numel[act_id])
        swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)

        self.act_id_to_swap_buffer[act_id] = swap_buffer
        compute_buffer = swap_buffer.narrow(0, 0, self.act_id_to_numel[act_id])
        compute_buffers.append(compute_buffer)
        swap_buffers.append(swap_buffer)

        return compute_buffers, swap_buffers

    #waits for inflight nvme write to complete
    def synchronize_writes(self):
        if self.pending_writes == 0:
            return
        assert self.pending_writes == self.aio_write_handle.wait()
        self.pending_writes = 0
        self.remove_activation_and_release_buffers(self.swap_out_acts)
        self.swap_out_acts = []
    
    #waits for inflight nvme reads to complete
    def synchronize_reads(self):
        if self.pending_reads == 0:
            return

        assert self.pending_reads == self.aio_read_handle.wait()

        self.pending_reads = 0

        for act_block, swap_in_buffer in zip(self.inflight_acts, self.inflight_swap_in_buffers):

            compute_buffer = swap_in_buffer.narrow(0, 0, self.act_id_to_numel[act_block.id])
            act_block.manage.data = compute_buffer.data
            act_block.nvme_status = PartitionedActStatus.AVAILABLE

        self.available_acts.update([act_block.id for act_block in self.inflight_acts])
        self.available_numel += self.inflight_numel

        self.inflight_acts = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0
    
    @nvtx_wrap
    def remove_activation_and_release_buffers(self, act_block_list):
        
        for act_block in act_block_list:
            act_id = act_block.id
            if act_id in self.act_id_to_buffer_id.keys():

                buffer_id = self.act_id_to_buffer_id[act_id]

                assert buffer_id is not None, "Missing buffer id for releasing"

                # print(f'@@@@@@@@@param_id {param_id} REALSE buffer_id {buffer_id}')
                self.available_buffer_ids.append(buffer_id)
                del self.act_id_to_buffer_id[act_id]
                del self.act_id_to_swap_buffer[act_id]
                # print_rank_0(f"param {param.sb_id} releases buffer id {buffer_id}  ")

                if act_id in self.available_acts:
                    self.available_acts.remove(act_id)
                    self.available_numel -= self.act_id_to_numel[act_id]
            # print(f'remove {act_id}')
            act_block.manage.data = self.invalid_buffer.data
            act_block.nvme_status = PartitionedActStatus.NOT_AVAILABLE

    def _swap_out(self, act_block, async_op=True):
        
        swap_out_paths = self._get_swap_paths(act_block)
        swap_out_params = self._get_swap_buffers(act_block)
        self._track_numel(act_block)

        swap_out_tensors(self.aio_write_handle, swap_out_params, swap_out_paths)

        self.pending_writes += len(swap_out_params)
        self.swap_out_acts.append(act_block)

        if not async_op:
            self.synchronize_writes()
    
    #blocking swap out followed by releasing the memory buffers
    @nvtx_wrap
    def swap_out_and_release(self, act_block, async_op=False):
        self._swap_out(act_block, async_op=async_op)
    
    # book keeping function for inflight swap in
    def _update_inflight_swap_in(self, act_block, swap_in_buffers, inflight_numel):
        self.inflight_acts.append(act_block)
        self.inflight_swap_in_buffers.extend(swap_in_buffers)
        self.inflight_numel += inflight_numel

        # print(self.inflight_acts[0])
        # print(act_block)

        act_block.nvme_status = PartitionedActStatus.INFLIGHT

        self.pending_reads += 1
    
    #assigns an in memory buffer and swaps in from nvme
    def swap_in(self, act_block, async_op=True, swap_in_buffers=None):

        assert act_block.nvme_status == PartitionedActStatus.NOT_AVAILABLE, "Act is already available or in flight"
        swap_in_paths = self._get_swap_paths(act_block)

        if swap_in_buffers is None:
            if len(self.available_buffer_ids) < len(swap_in_paths):
                ids = act_block.id
                print_rank_0(
                    f'Not enough swap in buffers {len(self.available_buffer_ids)} for {len(swap_in_paths)} params, ids = {ids}',
                    force=True)
                print_rank_0(
                    f'Num inflight: params {len(self.inflight_acts)}, buffers {len(self.inflight_swap_in_buffers)}, numel = {self.inflight_numel}',
                    force=True)
                print_rank_0(
                    f'Num available params: count = {len(self.available_acts)}, ids = {self.available_acts}, numel = {self.available_numel}',
                    force=True)

            assert len(swap_in_paths) <= len(
                self.available_buffer_ids
            ), f"Not enough buffers {len(self.available_buffer_ids)} for swapping {len(swap_in_paths)}"
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(act_block)
            inflight_numel = sum([t.numel() for t in compute_buffers])
        else:
            inflight_numel = sum([t.numel() for t in swap_in_buffers])

        # print(self.aio_read_handle, swap_in_buffers, swap_in_paths)
        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)

        self._update_inflight_swap_in(act_block, swap_in_buffers, inflight_numel)

        if not async_op:
            self.synchronize_reads()

    # Enables swapping into buffer that is out the control of swapper. This is always synchronous
    def swap_into_buffer(self, act_block, dest_buffer):
        assert act_block.nvme_status == PartitionedActStatus.NOT_AVAILABLE, f"param {act_block.id} is already available or inflight"

        require_swap_buffer = not (dest_buffer.is_pinned() and self._is_io_aligned(dest_buffer.numel()))

        if require_swap_buffer:
            assert len(self.available_buffer_ids) > 0, f"No buffer available to swap param {act_block.id}."
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(act_block)
            inflight_numel = compute_buffers[0].numel()
        else:
            swap_in_buffers = [dest_buffer]
            inflight_numel = dest_buffer.numel()

        swap_in_paths = self._get_swap_paths(act_block)

        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)
        self._update_inflight_swap_in(act_block, swap_in_buffers, inflight_numel)
        self.synchronize_reads()

        if require_swap_buffer:
            dest_buffer.data.copy_(act_block.manage.data)
            # Release swap buffer memory assignment. Note, this will mark the parameter not available.
            self.remove_activation_and_release_buffers([act_block])
    
    #assign a buffer to a param and return the buffer
    def get_buffer(self, act_block, numel):
        act_id = act_block.id

        assert self.available_swap_in_buffers(
        ) > 0, f"No swap buffers to allocate for fp16 param {act_id} of numel = {numel}"
        assert numel < self.elements_per_buffer, f"More elements {numel} than buffer size {self.elements_per_buffer}"

        self.act_id_to_numel[act_id] = numel
        
        buffer_id = self.available_buffer_ids.pop()
        # print(f'!!!!!!!!!!!!!param_id {param_id} GET buffer_id {buffer_id}')
        self.act_id_to_buffer_id[act_id] = buffer_id
        aligned_swap_numel = self._io_aligned_numel(self.act_id_to_numel[act_id])
        swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)

        self.act_id_to_swap_buffer[act_id] = swap_buffer
        compute_buffer = swap_buffer.narrow(0, 0, self.act_id_to_numel[act_id])
        # print_rank_0(f"param {param.sb_id} is assigned swap in buffer id {buffer_id}")
        return compute_buffer

    def reserve_available_buffers(self):
        buffers = []
        for id in self.available_buffer_ids:
            buffers.append(
                self.buffers.narrow(0, int(id * self.aligned_elements_per_buffer),
                                    int(self.aligned_elements_per_buffer)))
            self.reserved_buffer_ids.append(id)

        self.available_buffer_ids = []
        return buffers

    def release_reserved_buffers(self):
        for id in self.reserved_buffer_ids:
            self.available_buffer_ids.append(id)
        self.reserved_buffer_ids = []

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def _is_io_aligned(self, numel):
        return (numel % self.numel_alignment) == 0

    def reserve_partitioned_swap_space(self, partition_num_elems):
        aligned_numel = sum([self._io_aligned_numel(numel) for numel in partition_num_elems])
        self.partitioned_swap_buffer = get_accelerator().pin_memory(
            torch.zeros(aligned_numel, device='cpu', dtype=self.dtype))
        self.partitioned_swap_pool = SwapBufferPool([self.partitioned_swap_buffer])

    @nvtx_wrap
    def swap_out_partitioned_params(self, dst_fp16_params, src_fp32_params):
        assert self.partitioned_swap_buffer is not None, f'partitioned swap buffers for fp16 params not initialized'
        assert self.partitioned_swap_pool is not None, f'partitioned swap pool for fp16 params not initialized'
        assert len(dst_fp16_params) == len(src_fp32_params), \
        f'mismatch in number of fp16 params {len(dst_fp16_params)} and fp32 params {len(src_fp32_params)}'

        fp16_swap_paths = self._get_swap_paths(dst_fp16_params, must_exist=True)
        self.synchronize_writes()
        self.partitioned_swap_pool.reset()
        for i, fp32_tensor in enumerate(src_fp32_params):
            swap_tensor, _ = self.partitioned_swap_pool.insert_tensor(fp32_tensor, fp16_swap_paths[i],
                                                                      self._io_aligned_numel(fp32_tensor.numel()))
            assert swap_tensor is not None
            dst_fp16_params[i].nvme_status = PartitionedActStatus.AVAILABLE

        self.partitioned_swap_pool.swap_out(self.aio_write_handle)
        
        
        for param in dst_fp16_params:
            param.nvme_status = PartitionedActStatus.NOT_AVAILABLE
    

