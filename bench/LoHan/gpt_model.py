import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from new_checkpoint import checkpoint as new_checkpoint
from checkpoint import save_on_cpu, get_selective_offloading_checkpoint_modes2
import json
from collections import namedtuple
from nvme_ds.partitionrd_activation_swapper import AsyncPartitionedActivationSwapper
# from flash_attn import flash_attn_qkvpacked_func
from flash_attn.flash_attention import FlashAttention

act_stream = torch.cuda.Stream()
chp_id = [0]
chp_list = []
act_swapper = None
is_swap_and_recompute = 0

def set_training(args):
    global act_stream, chp_id, chp_list, act_swapper, is_swap_and_recompute
    for i in range(2 * args.num_layers):
        packed = torch.empty(
                (args.batch_size, args.max_seq_len, args.hidden_dim),
                dtype=torch.float16,
                pin_memory=True)
        # packed = torch.ones(
        #         1,
        #         dtype=torch.float16,
        #         pin_memory=True)
        chp_list.append(packed)

    def json_object_hook(d): 
        return namedtuple('X', d.keys())(*d.values())
    with open(args.sb_config) as f: 
        ds_config = json.load(f, object_hook=json_object_hook)

    act_swapper = AsyncPartitionedActivationSwapper(ds_config, torch.float16)
    is_swap_and_recompute = args.is_swap_and_recompute

@dataclass
class GPT2Config:
    dim: int = 5120
    hidden_dim: int = 5120
    num_heads: int = 40
    num_layers: int = 40
    dim_head: int = 128
    max_seq_len: int = 1024
    attn_pdrop: float = 0.1
    dropout: float = 0.1
    vocab_size: int = 50257
    layer_norm_epsilon: float = 1e-5

"""
{
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}
"""




class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        )

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.c_proj = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        # self.act = NewGELUActivation()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.c_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention = FlashAttention()
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        qkv = self.c_attn(x)
        
        batch_size, seq_len, _ = qkv.shape
        QKV = qkv.view(batch_size, seq_len, 3, self.config.num_heads, self.config.dim_head)
        
        # flash_attn_out = flash_attn_qkvpacked_func(QKV)
        flash_attn_out, _ = self.attention(QKV)
        out = flash_attn_out.reshape(batch_size, seq_len, self.config.hidden_dim)
        attn_out = self.c_proj(out)
        attn_out = self.resid_dropout(attn_out)

        return attn_out
    


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x, swap_list, act_pack):

        def func1(x):
            residual = x
            x = self.ln_1(x)
            x = self.attn(x) + residual
            return x

        # x = new_checkpoint(func1, x, context_fn=get_selective_offloading_checkpoint_modes2)
        # x = checkpoint(func1, x)
        with save_on_cpu(pin_memory=True, act_stream=act_stream, chp_id = chp_id, chp_list = chp_list, act_swapper=act_swapper):
            if is_swap_and_recompute:
                x = new_checkpoint(func1, x, context_fn=get_selective_offloading_checkpoint_modes2, swap_list=swap_list, act_pack=act_pack)
            else:
                x = checkpoint(func1, x)

        def func2(x):
            residual = x
            x = self.ln_2(x)
            x = self.mlp(x) + residual
            return x

        # x = new_checkpoint(func2, x, context_fn=get_selective_offloading_checkpoint_modes1)
        # x = checkpoint(func2, x)
        with save_on_cpu(pin_memory=True, act_stream=act_stream, chp_id = chp_id, chp_list = chp_list, act_swapper=act_swapper):
            if is_swap_and_recompute:
                x = new_checkpoint(func2, x, context_fn=get_selective_offloading_checkpoint_modes2, swap_list=swap_list, act_pack=act_pack)
            else:
                x = checkpoint(func2, x)
        return x

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([])
        for _ in range(config.num_layers):
            self.layers.append(GPT2Block(config))

        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x, swap_list, act_pack):

        x = self.token_emb(x)
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.pos_emb(position_ids)
        x = self.drop(x)

        for block in self.layers:
            x = block(x, swap_list, act_pack)

        x = self.ln_f(x)
        # def func_fc(x):
        x = self.fc(x)
        #     return x
        # x = checkpoint(func_fc, x)

        return x