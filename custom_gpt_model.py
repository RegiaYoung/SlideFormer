import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig
import math

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    from liger_kernel.transformers.layer_norm import LigerLayerNorm
except ModuleNotFoundError:
    LigerFusedLinearCrossEntropyLoss = None
    LigerLayerNorm = None

# 导入 flash_attn
try:
    from flash_attn import flash_attn_func
except ImportError:
    print("Flash Attention is not installed. Falling back to SDPA.")
    flash_attn_func = None
    
def zero_rotary_emb(hidden_states, position_ids):
    """返回适当形状的零张量，而不是None"""
    return None

def zero_update_causal_mask(attention_mask,hidden_states, position_ids, cache_position, *args):
    """返回适当形状的零张量，而不是None"""
    return None

# 步骤 1: 创建一个与 transformers 兼容的配置类
class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=5120,
        n_layer=40,
        n_head=80,
        n_inner=None,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        attn_implementation="flash_attention_2",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        
        # 兼容你的SlideFormer代码中的属性访问
        self.hidden_size = n_embd
        self._attn_implementation = attn_implementation
        if flash_attn_func is None and self._attn_implementation == "flash_attention_2":
            print("Flash Attention not found, automatically switching to 'sdpa'.")
            self._attn_implementation = "sdpa"

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

# 步骤 2: 构建模型的核心模块 (与 transformers 内部结构类似)
class CustomGPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.config = config
        
    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        qkv = self.c_attn(hidden_states)
        batch_size, seq_len, _ = hidden_states.size()

        # Reshape q, k, v for multi-head attention
        qkv = qkv.view(batch_size, seq_len, 3, self.n_head, self.head_dim)
        
        # MODIFIED: 使用 FlashAttention 或 SDPA
        if self.config._attn_implementation == "flash_attention_2":
            # FlashAttention expects (batch, seqlen, 3, nheads, d) or separated Q, K, V
            # Causal mask is handled internally by flash_attn_func
            # Note: flash_attn_func wants q, k, v separated. 
            query, key, value = qkv.unbind(2) # Split along the '3' dimension
            
            attn_output = flash_attn_func(
                query, key, value, dropout_p=self.dropout.p if self.training else 0.0,
                causal=True # CRITICAL: This enables causal masking
            )
        else: # Fallback to SDPA
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0) # (3, bs, n_head, seq_len, head_dim) -> unbind -> 3x(bs, n_head, seq_len, head_dim)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, 
                is_causal=True # CRITICAL: Use SDPA's built-in causal masking
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(batch_size, seq_len, self.n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return (attn_output,)

class CustomGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class CustomGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        ln_impl = LigerLayerNorm or nn.LayerNorm
        self.ln_1 = ln_impl(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CustomGPTAttention(config)
        self.ln_2 = ln_impl(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = CustomGPTMLP(config)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings, **kwargs)
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        return (hidden_states,) # 返回元tuple

# 步骤 3: 组装成完整的 PreTrainedModel
class CustomGPTModel(PreTrainedModel):
    config_class = CustomGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CustomGPTBlock(config) for _ in range(config.n_layer)])
        ln_impl = LigerLayerNorm or nn.LayerNorm
        self.ln_f = ln_impl(self.embed_dim, eps=config.layer_norm_epsilon)
        
        # 为了与 SlideFormer 的 split 函数兼容，添加这些方法
        self.decoder = self # 将自身作为 decoder
        self.layers = self.h # decoder.layers -> self.h
        self.norm = self.ln_f # decoder.norm -> self.ln_f

        self.rotary_emb = zero_rotary_emb
        self._update_causal_mask = zero_update_causal_mask

    def get_input_embeddings(self):
        return self.wte

    def get_decoder(self):
        return self # 关键！让 split 函数能找到 layers 和 norm
        
    def forward(self, input_ids, **kwargs):
        # 这是一个简化的 forward，主要为了模型结构完整。
        # SlideFormer 会逐层调用，所以这个完整的 forward 不会被直接使用。
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(0, seq_length, device=device).unsqueeze(0)
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        for block in self.h:
            hidden_states = block(hidden_states)[0]
            
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class CustomGPTForCausalLM(PreTrainedModel):
    config_class = CustomGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lce = (
            LigerFusedLinearCrossEntropyLoss(reduction="mean")
            if LigerFusedLinearCrossEntropyLoss is not None
            else None
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.transformer.get_decoder()

    def forward(self, input_ids, labels=None, **kwargs):
        
        # nomal
        # hidden_states = self.transformer(input_ids, **kwargs)
        # logits = self.lm_head(hidden_states)
        
        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        hidden_states = self.transformer(input_ids, **kwargs)
        loss = None
        logits = None
        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
            shift_labels = shift_labels.view(-1)

            if self.lce is not None:
                loss = self.lce(
                    self.lm_head.weight,
                    shift_hidden_states,
                    shift_labels,
                )
                logits = None
            else:
                logits = F.linear(shift_hidden_states, self.lm_head.weight)
                loss = F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="mean")

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)