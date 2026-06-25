"""MegaTrain real-data loss curve on MathFusionQA — same fixed batches as
bench/loss_compare/loss_compare_real.py (loads the shared /tmp/mathfusion_batches.pt cache).

Run from bench/MegaTrain with CUDA extensions built + cuda_pipeline importable.
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

MODEL_PATH = "/home/scc/models/Llama-3.1-8B-Instruct"
SEQ_LEN = 1024
BATCH = 32
LR = 2e-5
WEIGHT_DECAY = 0.1
SEED = 1234
STEPS = 40
BATCH_CACHE = "/tmp/mathfusion_batches.pt"


def main():
    torch.manual_seed(SEED)
    blob = torch.load(BATCH_CACHE)
    batches = blob["batches"][:STEPS]
    assert blob["seq"] == SEQ_LEN and blob["batch"] == BATCH, "batch cache mismatch"
    print(f"Loaded {len(batches)} cached batches")

    hf_model = AutoLigerKernelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    config = CPUMasterConfig(
        model_name=MODEL_PATH, dtype=torch.bfloat16, device=0,
        batch_size=BATCH, max_seq_len=SEQ_LEN, num_steps=STEPS,
        learning_rate=LR, weight_decay=WEIGHT_DECAY,
        checkpoint_interval=4, num_grad_slabs=12,
        attn_implementation="flash_attention_2", dataset_path="__dummy__",
    )
    model = CPUMasterModel(hf_model, config)
    del hf_model
    opt = DeepSpeedCPUAdam(model.get_parameters(), lr=LR, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=WEIGHT_DECAY, adamw_mode=True)

    device = torch.device("cuda:0")
    losses = []
    for step in range(STEPS):
        b = batches[step]
        loss_val, _, _ = model.forward_and_backward(
            b["input_ids"].to(device), b["attention_mask"].to(device),
            b["labels"].to(device))
        # No grad clipping — matches DeepSpeed (gradient_clipping=0.0) and
        # SlideFormer (no clip) so the three frameworks are compared bare.
        # torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
        opt.step()
        model._sync_params_to_gpu()
        model.zero_grad()
        opt.zero_grad()
        losses.append(float(loss_val))
        print(f"[megatrain] step {step+1:2d}/{STEPS}  loss {loss_val:.5f}")

    with open("/tmp/real_megatrain.json", "w") as f:
        json.dump({"mode": "megatrain", "model": os.path.basename(MODEL_PATH),
                   "lr": LR, "seq": SEQ_LEN, "batch": BATCH, "losses": losses}, f)
    print(f"\nSaved {len(losses)} losses to /tmp/real_megatrain.json")
    print(f"  first={losses[0]:.4f}  last={losses[-1]:.4f}  drop={losses[0]-losses[-1]:.4f}")


if __name__ == "__main__":
    main()
