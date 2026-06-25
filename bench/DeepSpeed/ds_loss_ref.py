"""DeepSpeed reference loss curve on the SAME MathFusionQA batches as
SlideFormer/MegaTrain, built on the proven bench harness.

Uses ds_bench.create_ds_config and the bench model-loading path. The ZeRO-3
stage3_max_live_parameters / prefetch_bucket are set to 5e7 — verified to fit
8B + bs32 on a 24GB GPU with NO grad accumulation. Only the data source (cached
fixed batches) and per-step loss logging differ from ds_bench.py.

Run from bench/DeepSpeed:
  numactl --cpunodebind=0 --membind=0 python ds_loss_ref.py --steps 40
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse

import torch
import deepspeed
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from ds_bench import create_ds_config

MODEL_PATH = "/home/scc/models/Llama-3.1-8B-Instruct"
BATCH = 32
SEQ = 1024
LR = 2e-5
BATCH_CACHE = "/tmp/mathfusion_batches.pt"


class _Args:
    def __init__(self):
        self.batch_size = BATCH
        self.seq_len = SEQ
        self.lr = LR
        self.use_bf16 = True
        self.zero_stage = 3
        self.offload = True
        self.offload_nvme = False
        self.nvme_path = "/RAID0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", type=str, default="/tmp/real_deepspeed.json")
    cli = ap.parse_args()

    blob = torch.load(BATCH_CACHE)
    assert blob["seq"] == SEQ and blob["batch"] == BATCH, "batch cache mismatch"
    batches = blob["batches"][:cli.steps]
    print(f"Loaded {len(batches)} cached MathFusionQA batches")

    model = AutoLigerKernelForCausalLM.from_pretrained(
        MODEL_PATH, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, use_cache=False,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ds_config = create_ds_config(_Args())
    # 5e7 fits 8B + bs32 on 24GB with no grad accumulation (matches ds_bench.py).
    ds_config["zero_optimization"]["stage3_max_live_parameters"] = int(5e7)
    ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = int(5e7)

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    engine.train()
    device = engine.device

    losses = []
    for step in range(cli.steps):
        b = batches[step]
        out = engine(input_ids=b["input_ids"].to(device),
                     attention_mask=b["attention_mask"].to(device),
                     labels=b["labels"].to(device))
        loss = out.loss
        engine.backward(loss)
        engine.step()
        losses.append(float(loss.item()))
        print(f"[deepspeed] step {step+1:2d}/{cli.steps}  loss {loss.item():.5f}")

    with open(cli.out, "w") as f:
        json.dump({"mode": "deepspeed", "model": os.path.basename(MODEL_PATH),
                   "lr": LR, "seq": SEQ, "batch": BATCH, "losses": losses}, f)
    print(f"\nSaved {len(losses)} losses to {cli.out}")
    print(f"  first={losses[0]:.4f}  last={losses[-1]:.4f}  drop={losses[0]-losses[-1]:.4f}")


if __name__ == "__main__":
    main()
