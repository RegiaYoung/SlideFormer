"""MegaTrain single-batch overfit check — same fixed batch as overfit_check.py.

Run from bench/MegaTrain with the CUDA extensions built and cuda_pipeline on
PYTHONPATH (mt_bench.sh builds them). Mirrors overfit_check.make_fixed_batch
exactly (same SEED/seq/batch/vocab) so the curve is directly comparable.
"""
import os
import sys
import json

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS))
for _p in (_THIS, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam

MODEL_PATH = "/home/scc/models/Qwen3-1.7B"
SEQ_LEN = 128
BATCH = 2
LR = 1e-4
SEED = 1234
STEPS = 30


def make_fixed_batch(tokenizer, device):
    g = torch.Generator().manual_seed(SEED)
    vocab = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab, (BATCH, SEQ_LEN), generator=g)
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    labels = input_ids.clone()
    return (input_ids.to(device), attention_mask.to(device), labels.to(device))


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hf_model = AutoLigerKernelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    config = CPUMasterConfig(
        model_name=MODEL_PATH, dtype=torch.bfloat16, device=0,
        batch_size=BATCH, max_seq_len=SEQ_LEN, num_steps=STEPS,
        learning_rate=LR, weight_decay=0.0,
        checkpoint_interval=4, num_grad_slabs=12,
        attn_implementation="flash_attention_2", dataset_path="__dummy__",
    )
    model = CPUMasterModel(hf_model, config)
    del hf_model
    opt = DeepSpeedCPUAdam(model.get_parameters(), lr=LR, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0, adamw_mode=True)

    input_ids, attn, labels = make_fixed_batch(tok, torch.device("cuda:0"))
    losses = []
    for step in range(STEPS):
        loss_val, _, _ = model.forward_and_backward(input_ids, attn, labels)
        torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1e9)  # no clip
        opt.step()
        model._sync_params_to_gpu()
        model.zero_grad()
        opt.zero_grad()
        losses.append(float(loss_val))
        print(f"[megatrain] step {step+1:2d}/{STEPS}  loss {loss_val:.5f}")

    with open("/tmp/overfit_megatrain.json", "w") as f:
        json.dump({"mode": "megatrain", "lr": LR, "losses": losses}, f)
    print(f"\nSaved {len(losses)} losses to /tmp/overfit_megatrain.json")
    print(f"  first={losses[0]:.4f}  last={losses[-1]:.4f}  "
          f"drop={losses[0]-losses[-1]:.4f}")


if __name__ == "__main__":
    main()
