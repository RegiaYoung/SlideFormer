# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import gc
import time
import torch
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.model_compat import load_text_model
from utils.metric import calculate_flops_per_batch
from utils.log_mem import log_memory_stats
from utils.datasets import DummyDataset
from offload_transformer import SlideFormerOffloader

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ModuleNotFoundError:
    AutoLigerKernelForCausalLM = None


def train_model(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    max_seq_length=1024,
    train_batch_size=64,
    num_epochs=1,
    use_bf16=True,
    use_liger: bool = True,
    attn_implementation: str = "flash_attention_2",
    ac_offload_nvme: bool = False,
    nvme_offload_fraction: float = 0.0,
    offload_dir: str = "./offload_dir",
):
    torch.cuda.empty_cache()
    gc.collect()

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using {dtype}.")

    if ac_offload_nvme or nvme_offload_fraction > 0.0:
        os.makedirs(offload_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_liger and AutoLigerKernelForCausalLM is None:
        import warnings
        warnings.warn(
            "liger-kernel is not installed; falling back to standard AutoModelForCausalLM. "
            "Install liger-kernel for better performance, or pass --no-use_liger to suppress this warning."
        )
        use_liger = False
    base_model = load_text_model(
        model_path,
        use_liger=use_liger,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer_kwargs = {
        "lr": 1e-5,
        "bias_correction": True,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "fp32_optimizer_state": True,
        "num_layer": len(base_model.get_decoder().layers) + 2,
        "nvme_offload_fraction": nvme_offload_fraction,
        "offload_dir": offload_dir,
        "prefetch": True,
    }
    model = SlideFormerOffloader(
        model=base_model,
        device=device,
        dtype=dtype,
        ac_offload_nvme=ac_offload_nvme,
        offload_dir=offload_dir,
        optimizer_kwargs=optimizer_kwargs,
    )

    train_dataset = DummyDataset(
        size=1024, tokenizer=tokenizer, max_length=max_seq_length
    )
    print(f"Dataset size: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size,
        shuffle=True, pin_memory=True, drop_last=True
    )
    model.train()

    epoch_steps = len(train_dataloader)
    total_flops = calculate_flops_per_batch(base_model, train_batch_size, max_seq_length)
    print(f"Per Batch FLOPS: {total_flops / 1e12:.2f} TFLOPS")
    tokens_per_batch = train_batch_size * max_seq_length
    log_step = 1

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        running_loss = 0

        for step, batch in enumerate(train_dataloader):
            if step == 0:
                step_start = time.perf_counter()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
            del outputs

            if step % log_step == 0:
                step_time = time.perf_counter() - step_start
                avg_loss = running_loss / log_step
                avg_tokens_per_sec = tokens_per_batch * log_step / step_time
                avg_tflops = total_flops * log_step / (10**12 * step_time)
                avg_iter_per_sec = log_step / step_time
                avg_batch_time = step_time / log_step
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{epoch_steps}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Time: {avg_batch_time:.2f}s, "
                    f"Speed: {avg_tokens_per_sec:.1f} tokens/s, "
                    f"Iter/s: {avg_iter_per_sec:.2f}, "
                    f"TFLOPS: {avg_tflops:.2f}"
                )
                running_loss = 0
                step_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s, "
            f"Average tokens/s: {tokens_per_batch * epoch_steps / epoch_time:.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlideFormer dummy run (end-to-end sanity check).")
    parser.add_argument("--model_path", type=str, default="/home/scc/models/Llama-3.1-8B-Instruct/")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_liger", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn_implementation", type=str,
        default="flash_attention_2", choices=["flash_attention_2", "sdpa"]
    )
    parser.add_argument("--ac_offload_nvme", action="store_true")
    parser.add_argument(
        "--nvme_offload_fraction",
        type=float,
        choices=[0.0, 0.5, 1.0],
        default=0.0,
    )
    parser.add_argument("--offload_dir", type=str, default="./offload_dir")
    args = parser.parse_args()

    train_model(
        model_path=args.model_path,
        max_seq_length=args.seq_len,
        train_batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_bf16=args.use_bf16,
        use_liger=args.use_liger,
        attn_implementation=args.attn_implementation,
        ac_offload_nvme=args.ac_offload_nvme,
        nvme_offload_fraction=args.nvme_offload_fraction,
        offload_dir=args.offload_dir,
    )
