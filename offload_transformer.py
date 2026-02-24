import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformer_layer import split_transformer_model # 
from utils.log_mem import log_memory_stats
from utils.module_utils import _print_module_structure
import concurrent.futures
import time
import os
from collections import deque, defaultdict, OrderedDict  # 新增导入
import queue
from typing import Optional
from optimizer import LayerAdam
# from optimizer.hybrid_adam.layer_adam_nvme import LayerAdam
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
        double_buffer: bool = False, # 新增双缓冲，默认关闭，收益不高推荐显存内存充足时开启
        enable_timing: bool = True,  # 新增参数控制计时功能
        enable_memory_stats: bool = True,  # 新增参数控制内存统计功能
        optimizer_kwargs: Optional[dict] = None,  # 新增参数传递优化器配置
    ):
        
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.double_buffer = double_buffer
        
        # 保存对基础模型的引用，以便后续保存
        self.base_model = model
                
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
        self.gpu_unit_size = min(1 + gpu_cache_size, len(model.get_decoder().layers))
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
            "num_layer": len(model.get_decoder().layers) + 2,  # 包括嵌入层和输出层
            "nvme_offload_fraction": 0.0, # 0为关闭，目前是0/0.5/1三档
            "offload_dir": offload_dir,
            "prefetch": True
        }
        optimizer_kwargs = optimizer_kwargs or default_optimizer_kwargs
        self.layer_optimizer = LayerAdam(**optimizer_kwargs)

        # 保存模型配置
        self.config = model.config
        self.ac_offload_nvme = ac_offload_nvme
        self.offload_dir = offload_dir
        
        #预估内存占用
        self.param_size_bytes = self.total_param * 2
        self.grad_size_bytes = self.max_param_size * 2
        self.opt_state_bytes = self.total_param * 4 * 2
        nvme_offload_fraction = optimizer_kwargs.get("nvme_offload_fraction", 0.0) if optimizer_kwargs else 0.0
        self.opt_offload_bytes = self.opt_state_bytes * nvme_offload_fraction  
        
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
            double_buffer=self.double_buffer
        )
        
        # 模型Layer
        self.transformer_layers = components['transformer_layers']
        self.rotary_emb = components['rotary_emb']
        self._update_causal_mask = components['update_causal_mask']
        
        # 快捷访问
        self.embed_layer = self.transformer_layers[0]  # idx = -1
        self.output_layer = self.transformer_layers[-1]  # idx = num_layers
        
        # rotary_position_embeddings & causal_mask
        self.rotary_emb = components['rotary_emb']
        self._update_causal_mask = components['update_causal_mask']   
        
        # 激活值存储
        self.position_embeddings = None
        
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
        """计算模型所有层中最大的参数数量"""
        # 计算嵌入层参数数量
        total_param = 0
        
        embed_layer = model.get_input_embeddings()
        embed_size = sum(p.numel() for p in embed_layer.parameters())
        total_param += embed_size
        
        # 计算decoder层参数数量
        decoder = model.get_decoder()
        layers = decoder.layers if hasattr(decoder, 'layers') else decoder.block
        decoder_sizes = sum(p.numel() for p in layers[0].parameters())
        total_param += decoder_sizes * len(layers)
        
        # 计算输出层参数数量
        norm_layer = decoder.norm
        lm_head = model.get_output_embeddings()
        output_size = sum(p.numel() for p in norm_layer.parameters()) + sum(p.numel() for p in lm_head.parameters())
        total_param += output_size
        
        # 返回三种层中的最大值
        return max(embed_size, decoder_sizes, output_size), total_param      

    def _init_gpu_cache(self):
        """初始化GPU缓存单元"""
        if self.gpu_unit_size < 1:
            raise ValueError("gpu_unit_size must be >= 1")
        
        for _ in range(self.gpu_unit_size):
            cache_unit = {
                "param": torch.empty(self.max_param_size, dtype=self.dtype, device=self.device),
                "grad": torch.empty(self.max_param_size, dtype=self.dtype, device=self.device)
            }
            self.gpu_cache_queue.put(cache_unit)
        
    def _verify_gpu_layers(self, ward, layer_number):
        """验证GPU上的层数是否符合window_size限制"""
        gpu_layers = 0
        device_display = []  # 用于存储每一层的显示符号

        for layer in self.transformer_layers:
            layer_device = next(layer.layer.parameters()).device
            # 判断当前层是否在 GPU 上，并记录显示符号
            if layer_device == self.device:
                device_display.append("G")
                gpu_layers += 1
            else:
                device_display.append(".")

        # 一次性打印所有层的设备状态
        print("|".join(device_display))
        # 打印最终结果并换行
        # print(f" {ward} layer({layer_number}): layers_on_gpu({gpu_layers}) and window_size({self.window_size})")
        
    def _register_layer_hooks(self):
        """为每一层注册管理钩子"""
        num_layers = len(self.transformer_layers)
        
        def get_pre_forward(idx):
            def hook(module, input):
                """前向传播前预加载窗口中的下一层"""
                # 预加载下一层（如果不是最后一个窗口）
                # self._verify_gpu_layers("F", idx)
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
                # def pre_backward_func(module):           
                # self._verify_gpu_layers("B", idx)
                # 预加载前一层（如果存在）
                prev_layer_idx = idx - 1
                if prev_layer_idx >= 0:
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
                # print(f"Post-Backward: {idx}")
                # Embed的call func导致hook不准确，导致第一层在反向过程中提前释放，手动后向
                # if idx == 0:
                #     return None
                
                # 前一层更新完成判断（for only one grad tensor）
                prev_update = None
                
                check_offset = 2 if self.double_buffer else 1
                check_idx = idx + check_offset
                
                if check_idx < num_layers:
                     prev_update = self.transformer_layers[check_idx].update_finished
                elif self.double_buffer and check_idx == num_layers: 
                     # Double Buffering Special Case:
                     pass
                     
                # if idx + 1 != num_layers:
                #     prev_update = self.transformer_layers[idx + 1].update_finished
                # 所有层都要更新和卸载
                module.to_offload_async(is_bwd=True, prev_update=prev_update)
                module.update_params()
                # module.update_finished.wait() # 需要注释，用于Ablation
                # return BackwardFunction.apply(module, post_backward_func, input)
            return hook
        
        # 为所有层注册钩子
        for idx, layer in enumerate(self.transformer_layers):
            layer.register_forward_pre_hook(get_pre_forward(idx))
            layer.register_forward_hook(get_post_forward(idx))
            layer.register_full_backward_pre_hook(get_pre_backward(idx))
            layer.register_full_backward_hook(get_post_backward(idx))

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
                    for op, times in layer.timer.timings.items():
                        all_stats[op].extend(times)
            print("\nHistorical Average Layer Statistics:")
            for op, times in all_stats.items():
                avg = sum(times) / len(times)
                print(f"  {op}: {avg*1000:.2f}ms (avg over {len(times)} calls)")
        else:
            # 收集当前step的数据
            for layer in self.transformer_layers:
                if layer.timer:
                    for op, duration in layer.timer.current_step.items():
                        all_stats[op].append(duration)
            print("\nCurrent Step Layer Statistics:")
            for op, durations in all_stats.items():
                avg = sum(durations) / len(durations)
                print(f"  {op}: {avg*1000:.2f}ms (avg across {len(durations)} layers)")
    
    def _print_memory_occupancy(self):
        """打印预估内存占用"""
        
        print(f"Model Type: {self.config.model_type}")
        print(f"Estimated Memory Occupancy: {self.param_size_bytes / (1024 ** 3):.2f} GB (params) + "
              f"{self.grad_size_bytes * (2 if self.double_buffer else 1) / (1024 ** 3):.2f} GB (grad) + "
              f"{self.grad_size_bytes / (1024 ** 3):.2f} GB (convert) + "
              f"{self.activation_bytes / (1024 ** 3):.2f} GB (activations) + "
              f"{self.opt_state_bytes / (1024 ** 3):.2f} GB (optimizer state) - "
              f"{self.opt_offload_bytes / (1024 ** 3):.2f} GB (os offload)",
              f" - {self.activation_bytes / (1024 ** 3):.2f} GB (ac offload) " if self.ac_offload_nvme else ""
        )
        estimated_memory = self.param_size_bytes + self.grad_size_bytes * (2 if self.double_buffer else 1) + self.grad_size_bytes + self.activation_bytes + self.opt_state_bytes - self.opt_offload_bytes
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
            if self.config._attn_implementation == "flash_attention_2":
                # Flash Attention使用形状为[batch_size, seq_length]的掩码
                layer_cpu_tensors = [
                    torch.empty((batch_size, seq_length, hidden_size), 
                            dtype=self.dtype, pin_memory=True),  # hidden_state
                    torch.empty((batch_size, seq_length), 
                            dtype=torch.bool, pin_memory=True)   # attention_mask
                    ]
            else:
                # SDPA使用形状为[batch_size, 1, seq_length, seq_length]的掩码
                layer_cpu_tensors = [
                    torch.empty((batch_size, seq_length, hidden_size), 
                              dtype=self.dtype, pin_memory=True),  # hidden_state
                    torch.empty((batch_size, 1, seq_length, seq_length), 
                              dtype=self.dtype, pin_memory=True)   # attention_mask
                    ]
            self.layer_tensors.append(layer_cpu_tensors)
        
        # 更新每个transformer层的layer_tensors引用
        for i, layer in enumerate(self.transformer_layers):
            # 跳过嵌入层和输出层
            if 0 < i < len(self.transformer_layers) - 1:
                layer.layer_cpu_tensors = self.layer_tensors
                
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
            mask_path = os.path.join(layer_dir, f"attention_mask")
            self.layer_tensor_files.append([hidden_path, mask_path])
                    
        # 更新每个transformer层的NVMe路径引用
        for i, layer in enumerate(self.transformer_layers):
            if 0 < i < len(self.transformer_layers) - 1:
                layer.layer_nvme_paths = self.layer_tensor_files
                layer.ac_offload_nvme = self.ac_offload_nvme
        
        print(f"NVMe activation files init, for {len(self.layer_tensor_files)} decoder layers")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """模型的前向传播"""
        # 启用异常检测, 
        # torch.autograd.set_detect_anomaly(True, check_nan=True)
        
        # activation checkpointing
        if self.layer_tensors is None and self.layer_tensor_files is None:
            batch_size, seq_length = input_ids.size()
            hidden_size = self.config.hidden_size
            if self.ac_offload_nvme:
                # NVMe offload
                self._allocate_activation_tensors_on_nvme(batch_size, seq_length, hidden_size)
            else:
                # CPU offload
                self._allocate_activation_tensors_on_cpu(batch_size, seq_length, hidden_size)
            
            # count
            activation_size_per_layer = (
                batch_size * seq_length * hidden_size * 2 +  # hidden_state (float16)
                batch_size * seq_length * 1  # attention_mask (int64)/(int8)/(bool)
            )
            self.activation_bytes = (len(self.transformer_layers) - 2) * activation_size_per_layer
            self._print_memory_occupancy()
            
        # ----------------------- Forward Start -------------------------
            
        # Pre-Load Embedding and window_size-1 transformer layers
        for i in range(0, min(1, len(self.transformer_layers))):
            self.transformer_layers[i].to_device_async()

        if self.enable_memory_stats:
            print("Begin:"+log_memory_stats())
            
        # Embedding Layer
        # self.embed_layer.h2d_ready.synchronize()
        hidden_states = self.embed_layer(input_ids)
        
        causal_mask = None
        # attention_mask (keep same as huggingface)
        cache_position = torch.arange(
            0, hidden_states.shape[1], device=hidden_states.device
        )
        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, None, None)

        # position_embeddings
        if self.position_embeddings is None:
            # self.rotary_emb.to(self.device)
            position_ids = cache_position.unsqueeze(0)
            self.position_embeddings = self.rotary_emb(hidden_states, position_ids)
            # self.rotary_emb.to('cpu')
            
        # torch.cuda.empty_cache() # (gpu: -500MB)
        
        # Forward
        for idx in range(1, len(self.transformer_layers)-1):
            hidden_states = self.transformer_layers[idx](
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=self.position_embeddings
            )
            # if hidden_states.isnan().any():
            #     raise ValueError(f"NaN detected in layer {idx} during forward pass")
            
        # 4. 损失计算
        loss = None
        logits = None
        
        if labels is not None:
            
            # Norm + LM Head + CrossEntropyLoss
            loss = self.output_layer(hidden_states, labels, self.config.hidden_size)
            
            # Ablation Old One / No Fused Kernel
            # logits = logits = self.output_layer(hidden_states)
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
            
            if self.enable_memory_stats:
                print("Forward:" + log_memory_stats())
            
            # 直接调用backward，hooks会处理层的加载/卸载
            loss.backward()

            # Embed要手动后向
            # prev_update = self.transformer_layers[1].update_finished
            # self.transformer_layers[0].to_offload_async(is_bwd=True, prev_update=prev_update)
            # self.transformer_layers[0].update_params()
            
            # torch.cuda.empty_cache() # tmp
            
            if self.enable_memory_stats:
                print("Backward:"+log_memory_stats()) # (第二次回来多了3G)
                        
            # 显示当前step所有层的平均
            self.print_layer_stats(show_history=False)

            # 显示所有历史数据的平均
            # self.print_layer_stats(show_history=True)

            # 显示特定层的统计
            # self.print_layer_stats(layer_idx=0, show_history=True)  # 显示第0层的历史统计
            # self.print_layer_stats(layer_idx=0, show_history=False) # 显示第0层的当前step统计
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )
    
    def update_learning_rate(self, new_lr):
        """更新优化器的学习率"""
        self.layer_optimizer.update_learning_rate(new_lr)
    
    def save_pretrained(self, output_dir):
        """保存底层预训练模型到指定目录"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print("Going to save model to:", output_dir)
        
        # 确保所有层都已经被卸载到CPU
        print("Make sure all layers updated...")
        for i, layer in enumerate(self.transformer_layers):
            # 等待任何未完成的参数更新
            layer.wait_for_update()
        
        # 将参数从fp32转换回bf16，确保与预训练模型精度一致
        print("Converting parameters from fp32 to bf16...")
        for layer_wrapper in self.transformer_layers:
            for name, param in layer_wrapper.layer.named_parameters():
                # 获取当前参数在CPU上的fp32视图
                cpu_param = param.data
                # 创建bf16版本
                bf16_param = cpu_param.to(dtype=self.dtype)
                # 替换原始参数
                param.data = bf16_param
        
        # 保存模型
        print("saving the model...")
        self.base_model.save_pretrained(output_dir)
        
        print(f"Successfully saved the model to {output_dir}")
        return output_dir