# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import threading
import queue
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import concurrent.futures
from transformers.modeling_utils import PreTrainedModel
from typing import Dict, Any, Optional
from collections import OrderedDict
from utils.timer import LayerTimer
from utils.legacy_fused_linear_cross_entropy import LegacyFusedLinearCrossEntropyLoss
from sliding_checkpoint import SlidingCheckpoint, save_on_cpu
import time
import torch.cuda.nvtx as nvtx

_compute_stream = torch.cuda.default_stream()
_h2d_stream = torch.cuda.Stream()
_d2h_stream = _h2d_stream
# _d2h_stream = torch.cuda.Stream()

# Chunked H2D overlaps fp32->bf16 CPU conversion with PCIe DMA.
H2D_CHUNK_SIZE = 32 * 1024 * 1024
D2H_CHUNK_SIZE = 32 * 1024 * 1024

def build_decoder_attention_mask(
    decoder: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_key_values=None,
):
    """Build the decoder attention mask for llama-like dense models across HF versions."""
    decoder_config = decoder.config

    legacy_update_causal_mask = getattr(decoder, "_update_causal_mask", None)
    if legacy_update_causal_mask is not None:
        try:
            return legacy_update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
                None,
            )
        except TypeError:
            return legacy_update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
            )

    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError as exc:
        raise RuntimeError(
            "Could not import `transformers.masking_utils`. "
            "Please use a transformers version that provides either "
            "`decoder._update_causal_mask` (v4.x) or `masking_utils.create_causal_mask` (v5.x)."
        ) from exc

    if position_ids is None and cache_position is not None:
        position_ids = cache_position.unsqueeze(0)

    return create_causal_mask(
        config=decoder_config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

def split_transformer_model(
    model: PreTrainedModel, 
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    **kwargs
) -> Dict[str, Any]:
    """Split the model and route tied gradients through the embedding owner."""
    # ---- embed和lm_head是否共享参数 ----
    tied_param = kwargs.pop("tied_param", None)
    tied_grad_accum = kwargs.pop("tied_grad_accum", None)
    
    # ---- Double Buffering Distribution ----
    double_buffer = kwargs.pop('double_buffer', False)
    cpu_grad_tensor = kwargs.pop("cpu_grad_tensor", None)
    
    global _d2h_stream
    _d2h_stream = torch.cuda.Stream() if double_buffer else _d2h_stream
        
    # helper to get tensor for a layer
    def get_grad_tensor(idx):
        if double_buffer:
            return cpu_grad_tensor[idx % 2]
        else:
            return cpu_grad_tensor[0]
        
    # ---- 创建所有层的ModuleList ----
    all_layers = nn.ModuleList([])
    
    # ---- 创建嵌入层 (idx = 0) ----    
    embed_layer = EmbeddingWrapper(
        layer=model.get_input_embeddings(),
        device=device,
        layer_idx=0,  # 特殊索引表示嵌入层
        offload_device=torch.device("cpu"),
        dtype=dtype,
        tied_param=tied_param,
        tied_grad_accum=tied_grad_accum,
        cpu_grad_tensor=get_grad_tensor(0),
        **kwargs
    )
    all_layers.append(embed_layer)
    decoder = model.get_decoder()
    
    # ---- 处理decoder层 (idx = 1 ~ num_layers) ----
    layers = decoder.layers
    num_layers = len(layers)
    for idx, layer in enumerate(layers, start=1):
        decoder_layer = DecoderWrapper(
            layer=layer,
            device=device,
            layer_idx=idx,
            offload_device=torch.device("cpu"),
            dtype=dtype,
            is_last_layer=(idx == num_layers),
            cpu_grad_tensor=get_grad_tensor(idx),
            **kwargs
        )
        all_layers.append(decoder_layer)
    
    # ---- 创建输出层 (idx = num_layers + 1) ----
    output_layer = OutputWrapper(
        norm_layer=decoder.norm,
        lm_head=model.get_output_embeddings(),
        device=device,
        layer_idx=num_layers + 1,  # 使用num_layers + 1作为输出层索引
        offload_device=torch.device("cpu"),
        dtype=dtype,
        skip_params={tied_param} if tied_param is not None else None,
        tied_param=tied_param,
        tied_grad_accum=tied_grad_accum,
        cpu_grad_tensor=get_grad_tensor(num_layers + 1),
        **kwargs
    )
    all_layers.append(output_layer)
    
    components = {
        'transformer_layers': all_layers,
        'rotary_emb': getattr(decoder, 'rotary_emb', None),
    }
    
    return components

class OffloadLayerWrapper(nn.Module):
    """基类：管理模型层的CPU-GPU传输和参数更新"""
    def __init__(
        self,
        layer: nn.Module,
        device: torch.device,
        layer_idx: int,
        offload_device: torch.device = torch.device("cpu"),
        optimizer_kwargs: Optional[dict] = None,
        update_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        h2d_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,  # 新增H2D专用线程池
        d2h_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,  # 新增D2H专用线程池
        gpu_cache_queue = None,
        cpu_grad_tensor = None,
        bf16_param_tensor = None,
        skip_params: Optional[set[torch.nn.Parameter]] = None,
        tied_param: Optional[torch.nn.Parameter] = None,
        tied_grad_accum: Optional[torch.Tensor] = None,
        enable_timing: bool = False,
        is_last_layer: bool = False,
        layer_optimizer = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.offload_device = offload_device
        self.layer_idx = layer_idx
        self.is_last_layer = is_last_layer
        self.gpu_cache_queue = gpu_cache_queue
        self.dtype = dtype  # 存储dtype
        
        # 确保层完全在CPU上
        self.layer = layer
        
        # CUDA流和事件
        self.compute_stream = _compute_stream
        self.h2d_stream = _h2d_stream
        self.d2h_stream = _d2h_stream
        self.compute_ready = torch.cuda.Event()
        self.compute_ready_bw = torch.cuda.Event()
        # self.h2d_ready = torch.cuda.Event()
        # self.d2h_ready = torch.cuda.Event()
        
        # 传输与更新专用线程池
        self.h2d_executor = h2d_executor
        self.d2h_executor = d2h_executor
        self.update_executor = update_executor
        if self.update_executor is None or (self.h2d_executor is None and self.d2h_executor is None):
            raise RuntimeError("No executor provided for parameter update or transfer")
        
        # 参数更新锁和事件
        self.update_finished = threading.Event()
        self.update_finished.set()
        self.update_lock = threading.Lock()
        self._h2d_future = None
        self._d2h_future = None
        self._d2h_chunk_queue = queue.Queue(maxsize=4)
        
        # ---- tie bookkeeping ----
        self.skip_params = skip_params or set()
        self.tied_param = tied_param
        self.tied_grad_accum = tied_grad_accum
        
        self._has_tie = (self.tied_param is not None and self.tied_grad_accum is not None)
        self._tied_numel = self.tied_param.numel() if self._has_tie else None
        self._is_tied_output = self._has_tie and (self.tied_param in self.skip_params)   # output: skip tied weight
        self._is_tied_owner = False  # embed: owns tied weight (managed includes tied_param)
        
        # --- param and grad views on CPU ---
        self._managed_named_params = [
            (n, p) for n, p in self.layer.named_parameters()
            if p not in self.skip_params
        ]
        self.total_size = sum(p.numel() for _, p in self._managed_named_params)
        self._cpu_params_flat = torch.empty(
            self.total_size,
            dtype=torch.float32,
            device=self.offload_device,
        )
        
        # 初始版本，每Layer创建自己的CPU梯度张量
        # self._cpu_grads_flat = torch.empty(
        #     self.total_size,
        #     dtype=self.dtype,
        #     device=self.offload_device,
        #     pin_memory=True
        # )
        
        # 使用共享的CPU梯度张量, 为原来的 1/Layer
        self._cpu_grads_flat = cpu_grad_tensor
        
        self._param_maps = OrderedDict()
        self._param_views = OrderedDict()
        self._grad_views = OrderedDict()
        self._param_to_grad_views = OrderedDict()
        
        # record tied slice in owner (embedding) so we can add accum into grad_flat
        self._tied_owner_offset = None
        
        with torch.no_grad():
            offset = 0
            for name, param in self._managed_named_params:
                shape = param.shape
                size = param.numel()
                self._param_maps[name] = (offset, shape, size)
  
                self._param_views[name] = self._cpu_params_flat[offset:offset + size].view(shape)
                self._grad_views[name] = self._cpu_grads_flat[offset:offset + size].view(shape)
                
                self._param_views[name].copy_(param.data, non_blocking=True)
                param.data = self._param_views[name]
                # param.grad = self._grad_views[name]
                param.grad = None
                
                self._param_to_grad_views[param] = self._grad_views[name]
                
                if self._has_tie and (param is self.tied_param):
                    self._is_tied_owner = True
                    self._tied_owner_offset = offset  
                
                offset += size
                
        # at this moment, tied_param.data should already be embed's fp32 view (because embed wrapper built earlier)
        self._tied_cpu_view =  self.tied_param.data if self._is_tied_output else None 
        
        # GPU缓存相关
        self._current_gpu_cache = None
        self.bf16_param_tensor = bf16_param_tensor
        
        # 优化器相关
        default_optimizer_kwargs = {
            "lr": 1e-6,
            "bias_correction": True,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "fp32_optimizer_state": True,
        }
        
        optimizer_kwargs = optimizer_kwargs or default_optimizer_kwargs
        
        # 使用共享的LayerAdam优化器或创建单独的CPUAdam
        managed_params = [param for _, param in self._managed_named_params]
        self.layer_optimizer = layer_optimizer
        if self.layer_optimizer is not None:
            # 将此层的参数注册到共享优化器
            self.layer_optimizer.add_layer_params(self.layer_idx, managed_params)
        else:
            # 如果没有提供共享优化器，则创建独立的CPUAdam
            from optimizer.hybrid_adam.cpu_adam import CPUAdam
            self.optimizer = CPUAdam(managed_params, **optimizer_kwargs)
        
        # 计时器
        self.enable_timing = enable_timing
        self.timer = LayerTimer(layer_idx) if enable_timing else None

    def to_device_async(self, is_bwd=False):
        """异步加载到GPU - 使用专用H2D线程池"""
        def _h2d_task():
            self.wait_for_update()
            nvtx.range_push(f"H2D_Layer_{self.layer_idx}")
            if self.enable_timing:
                self.h2d_stream.synchronize()
            # if prev_offload is not None:
            #     self.h2d_stream.wait_event(prev_offload)
                
            # print("H2D transfer started: ", self.layer_idx)
            with torch.cuda.stream(self.h2d_stream):
                start = time.perf_counter() if self.enable_timing else None
                
                self._current_gpu_cache = self.gpu_cache_queue.get()
                self.h2d_stream.wait_event(self._current_gpu_cache["reuse_ready"])
                if is_bwd:
                    self._current_gpu_cache['grad'].zero_()
                # self._current_gpu_cache = {
                #         'param': torch.empty_like(self._cpu_grads_flat, device=self.device),
                #         'grad': torch.empty_like(self._cpu_grads_flat, device=self.device)
                #     }
                
                with torch.no_grad():
                    # Interleave fp32->bf16 CPU conversion with PCIe DMA.
                    chunk = H2D_CHUNK_SIZE
                    src_fp32 = self._cpu_params_flat
                    staging = self.bf16_param_tensor
                    dst_gpu = self._current_gpu_cache['param']
                    for cs in range(0, self.total_size, chunk):
                        ce = min(cs + chunk, self.total_size)
                        staging[cs:ce].copy_(src_fp32[cs:ce])
                        dst_gpu[cs:ce].copy_(staging[cs:ce], non_blocking=True)
                        
                    for name, param in self._managed_named_params:
                        offset, shape, size = self._param_maps[name]
                        param.data = self._current_gpu_cache['param'][offset:offset + size].view(shape)
                        param.grad = self._current_gpu_cache['grad'][offset:offset + size].view(shape) if is_bwd else None
                        
                    # tied weight for OUTPUT only (consumer)
                    if self._is_tied_output:
                        off = self.total_size
                        n = self._tied_numel
                        tied_src = self._tied_cpu_view.view(-1)

                        for cs in range(0, n, chunk):
                            ce = min(cs + chunk, n)
                            staging[off + cs:off + ce].copy_(tied_src[cs:ce])
                            dst_gpu[off + cs:off + ce].copy_(staging[off + cs:off + ce], non_blocking=True)

                        self.tied_param.data = self._current_gpu_cache['param'][off:off + n].view(self.tied_param.shape)
                        self.tied_param.grad = self._current_gpu_cache['grad'][off:off + n].view(self.tied_param.shape) if is_bwd else None
                        
                self.h2d_stream.synchronize()    
                # self.h2d_ready.record(self.h2d_stream)
                if self.enable_timing and start is not None:
                    duration = time.perf_counter() - start
                    self.timer.record_time("h2d_transfer_bw" if is_bwd else "h2d_transfer_fw", duration)
                nvtx.range_pop()
                    
        # 使用H2D专用线程池
        self._h2d_future = self.h2d_executor.submit(_h2d_task)
    
    def wait_for_h2d(self):
        """等待GPU加载完成"""
        # print(f"Waiting for layer {self.layer_idx} h2d transfer...")
        if self._h2d_future and not self._h2d_future.done():
            # print(f"Waiting for layer {self.layer_idx} h2d transfer...")
            self._h2d_future.result()

    def to_offload_async(self, is_bwd=False, prev_update=None):
        """异步卸载到 CPU；backward 走 chunked D2H+update，forward 只恢复参数视图。"""
        def _d2h_task(is_bwd=is_bwd, prev_update=prev_update):
            if is_bwd:
                self.d2h_stream.wait_event(self.compute_ready_bw)
                if prev_update is not None and not prev_update.is_set():
                    # print(f"Waiting for layer {self.layer_idx + 1} update...")
                    prev_update.wait()
                # Drain the GPU-side wait_event before timing starts,
                # so d2h_transfer_bw only measures actual DMA time.
                if self.enable_timing:
                    self.d2h_stream.synchronize()
            else:
                self.d2h_stream.wait_event(self.compute_ready)
                
            nvtx.range_push(f"D2H_Layer_{self.layer_idx}")
            with torch.cuda.stream(self.d2h_stream):
                # print(f"Layer {self.layer_idx} d2h transfer begin...")
                start = time.perf_counter() if self.enable_timing else None
                with torch.no_grad():
                    if is_bwd:
                        gpu_grad = self._current_gpu_cache['grad']
                        chunk = D2H_CHUNK_SIZE

                        # Tied owner must merge shared grad before exposing chunks to Adam.
                        if self._is_tied_owner:
                            for cs in range(0, self.total_size, chunk):
                                ce = min(cs + chunk, self.total_size)
                                self._cpu_grads_flat[cs:ce].copy_(gpu_grad[cs:ce], non_blocking=True)

                            self.d2h_stream.synchronize()
                            self._cpu_grads_flat[
                                self._tied_owner_offset:self._tied_owner_offset + self._tied_numel
                            ].add_(self.tied_grad_accum)

                            for cs in range(0, self.total_size, chunk):
                                ce = min(cs + chunk, self.total_size)
                                self._d2h_chunk_queue.put((cs, ce, None))
                            self._d2h_chunk_queue.put(None)
                        else:
                            # Queue each chunk with a ready_event before Adam consumes it.
                            for cs in range(0, self.total_size, chunk):
                                ce = min(cs + chunk, self.total_size)
                                self._cpu_grads_flat[cs:ce].copy_(gpu_grad[cs:ce], non_blocking=True)
                                ready_event = torch.cuda.Event()
                                ready_event.record(self.d2h_stream)
                                self._d2h_chunk_queue.put((cs, ce, ready_event))

                            # output-layer tied grad offloading
                            if self._is_tied_output:
                                self.tied_grad_accum.copy_(
                                    gpu_grad[self.total_size:self.total_size + self._tied_numel],
                                    non_blocking=True,
                                )

                            # sentinel: tells _do_chunked_update that all chunks are done
                            self._d2h_chunk_queue.put(None)

                    for name, param in self._managed_named_params:
                        param.data = self._param_views[name]
                        param.grad = None
                    # Restore the shared tied Parameter to the owner's CPU view.
                    if self._is_tied_output:
                        self.tied_param.data = self._tied_cpu_view
                        self.tied_param.grad = None

                self._current_gpu_cache["reuse_ready"].record(self.d2h_stream)
                # self.d2h_ready.record(self.d2h_stream)
                self.gpu_cache_queue.put(self._current_gpu_cache)
                self._current_gpu_cache = None

                # print(f"Layer {self.layer_idx} d2h transfer finished.")
                if self.enable_timing and start is not None:
                    duration = time.perf_counter() - start
                    self.timer.record_time("d2h_transfer_bw" if is_bwd else "d2h_transfer_fw", duration)

            nvtx.range_pop()

        # 使用D2H专用线程池
        self._d2h_future = self.d2h_executor.submit(_d2h_task)


    def wait_for_d2h(self):
        """等待CPU卸载完成"""
        # print(f"Waiting for layer {self.layer_idx} d2h transfer...")
        if self._d2h_future and not self._d2h_future.done():
            # print(f"Waiting for layer {self.layer_idx} d2h transfer...")
            self._d2h_future.result()
        # with torch.cuda.stream(torch.cuda.current_stream()):
        #     self.d2h_ready.wait()

    def _do_update(self):
        """执行参数更新 (monolithic fallback, used when D2H is monolithic)"""
        self.wait_for_d2h()
        nvtx.range_push(f"Update_Layer_{self.layer_idx}")
        with self.update_lock:
            start = time.perf_counter() if self.enable_timing else None

            if self.layer_optimizer is not None:
                self.layer_optimizer.step_with_grad_views(self.layer_idx, self._param_to_grad_views)
            else:
                self.optimizer.step()

            if self.enable_timing and start is not None:
                self.timer.record_time("parameter_update", time.perf_counter() - start)

            self.update_finished.set()
        nvtx.range_pop()
        return True

    def _do_chunked_update(self):
        """Consume D2H-ready chunks as they land and run chunked Adam."""
        nvtx.range_push(f"Update_Layer_{self.layer_idx}")
        with self.update_lock:
            start = None  # set when first chunk arrives

            if self.layer_optimizer is not None:
                param_group = self.layer_optimizer.begin_chunk_step(self.layer_idx)
                while True:
                    item = self._d2h_chunk_queue.get()
                    if item is None:
                        break
                    if start is None and self.enable_timing:
                        start = time.perf_counter()
                    cs, ce, ready_event = item
                    if ready_event is not None:
                        ready_event.synchronize()
                    self.layer_optimizer.step_chunk(
                        self._cpu_params_flat[cs:ce],
                        self._cpu_grads_flat[cs:ce],
                        self.layer_optimizer.exp_avg_flat[self.layer_idx][cs:ce],
                        self.layer_optimizer.exp_avg_sq_flat[self.layer_idx][cs:ce],
                        param_group,
                    )
                self.layer_optimizer.end_chunk_step(self.layer_idx)
            else:
                # Fallback: wait for all chunks then run full optimizer step
                while True:
                    item = self._d2h_chunk_queue.get()
                    if item is None:
                        break
                    if start is None and self.enable_timing:
                        start = time.perf_counter()
                self.optimizer.step()

            if start is not None:
                self.timer.record_time("parameter_update", time.perf_counter() - start)

            self.update_finished.set()
        nvtx.range_pop()
        return True

    def update_params(self):
        """异步更新参数"""
        self.update_finished.clear()
        future = self.update_executor.submit(self._do_chunked_update)
        return future


    def wait_for_update(self):
        """等待参数更新完成"""
        if not self.update_finished.is_set():
            # print(f"Waiting for layer {self.layer_idx} update...")
            self.update_finished.wait()

    def forward(self, *args, **kwargs):
        """前向传播，需要在子类中实现"""
        raise NotImplementedError

class DecoderWrapper(OffloadLayerWrapper):
    """Decoder层的实现"""
    def __init__(self, *args, **kwargs):
        # 添加layer_cpu_tensors参数
        self.ac_offload_nvme = kwargs.pop('ac_offload_nvme', False)
        self.layer_cpu_tensors = kwargs.pop('layer_cpu_tensors', None)
        self.layer_nvme_paths = kwargs.pop('layer_nvme_paths', None)
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        position_embeddings=None,
    ):
        """Decoder层的前向传播实现"""
        nvtx.range_push(f"Forward_Layer_{self.layer_idx}")
        # self.compute_stream.wait_event(self.h2d_ready)
        with torch.cuda.stream(self.compute_stream):
            # self.wait_for_h2d()
            start = time.perf_counter() if self.enable_timing else None

            def _forward(hidden_states):
                output = self.layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_attentions=False,
                )
                return output[0] if isinstance(output, tuple) else output
            
            # 使用SlidingCheckpoint
            if self.ac_offload_nvme:
                with SlidingCheckpoint(
                    layer_idx=self.layer_idx,
                    layer_tensors=None,
                    gds_offload=True,
                    file_paths=self.layer_nvme_paths,
                    is_last_layer=self.is_last_layer,
                    device=self.device,
                    timing_recorder=self.timer,
                    enable_timing=self.enable_timing,
                ):
                    output = checkpoint(_forward, hidden_states, use_reentrant=False)
            else:
                with SlidingCheckpoint(
                    layer_idx=self.layer_idx,
                    layer_tensors=self.layer_cpu_tensors,
                    gds_offload=False,
                    file_paths=None,
                    is_last_layer=self.is_last_layer,
                    device=self.device,
                    timing_recorder=self.timer,
                    enable_timing=self.enable_timing,
                ):
                # with save_on_cpu(pin_memory=True):
                    output = checkpoint(_forward, hidden_states, use_reentrant=False)
            
            self.compute_ready.record(self.compute_stream)
            
            if self.enable_timing and start is not None:
                self.timer.record_time("forward", time.perf_counter() - start)
                
            nvtx.range_pop()
            return output

class EmbeddingWrapper(OffloadLayerWrapper):
    """输入嵌入层的实现"""
    def forward(self, input_ids):
        nvtx.range_push(f"Forward_Layer_{self.layer_idx}")
        # self.compute_stream.wait_event(self.h2d_ready)
        with torch.cuda.stream(self.compute_stream):
            # self.wait_for_h2d()
            # self.h2d_ready.synchronize()
            start = time.perf_counter() if self.enable_timing else None
            output = self.layer(input_ids)
            self.compute_ready.record(self.compute_stream)
            if self.enable_timing and start is not None:
                self.timer.record_time("forward", time.perf_counter() - start)
            nvtx.range_pop()
            return output

class OutputWrapper(OffloadLayerWrapper):
    """组合norm, lm_head和CrossEntropyLoss的输出层"""
    def __init__(self, norm_layer: nn.Module, lm_head: nn.Module, *args, **kwargs):
        # 创建一个Sequential来组合norm和lm_head
        combined_layer = nn.Sequential(norm_layer,lm_head)
        super().__init__(layer=combined_layer, *args, **kwargs)
        self.norm_layer = norm_layer
        self.lm_head = lm_head
        # The vendored legacy FLCE path is still the fastest choice for large vocab.
        # from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        # self.lce = LigerFusedLinearCrossEntropyLoss(reduction="mean", accum_dtype=torch.float32)
        self.lce = LegacyFusedLinearCrossEntropyLoss(reduction="mean")
    
    def forward(self, hidden_states, labels=None):
        nvtx.range_push(f"Forward_Layer_{self.layer_idx}")
        # self.compute_stream.wait_event(self.h2d_ready)
        with torch.cuda.stream(self.compute_stream):
            # self.wait_for_h2d()
            # self.h2d_ready.synchronize()
            start = time.perf_counter() if self.enable_timing else None
            
            hidden_states = self.norm_layer(hidden_states)

            if labels is None:
                output = self.lm_head(hidden_states)
            else:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # flatten tokens
                shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
                shift_labels = shift_labels.view(-1)

                output = self.lce(
                    self.lm_head.weight,
                    shift_hidden_states,
                    shift_labels
                )
                
            # output = self.layer(hidden_states)
            
            self.compute_ready.record(self.compute_stream)
            
            if self.enable_timing and start is not None:
                duration = time.perf_counter() - start
                self.timer.record_time("forward", duration)
            nvtx.range_pop()
            return output
        
    # def forward(self, hidden_states):
    #     nvtx.range_push(f"Forward_Layer_{self.layer_idx}")
    #     with torch.cuda.stream(self.compute_stream):
    #         # self.wait_for_h2d()
    #         # self.h2d_ready.wait()
    #         # self.h2d_stream.synchronize()
    #         start = time.perf_counter() if self.enable_timing else None
            
    #         output = self.layer(hidden_states)
            
    #         self.compute_ready.record()
            
    #         if self.enable_timing and start is not None:
    #             duration = time.perf_counter() - start
    #             self.timer.record_time("forward", duration)
            
    #         nvtx.range_pop()
    #         return output
