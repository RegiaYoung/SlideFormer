# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
"""pytorch allocator config"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # 峰值-2G，其余无变化
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "garbage_collection_threshold:0.6" # 无明显变化
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import datetime
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.metric import calculate_flops_per_batch
from utils.log_mem import log_memory_stats
from utils.datasets import DummyDataset
from offload_transformer import SlideFormerOffloader

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ModuleNotFoundError:
    AutoLigerKernelForCausalLM = None

from torch.profiler import profile, record_function, ProfilerActivity


def train_model(
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/",
    max_seq_length=512,
    train_batch_size=4,
    num_epochs=1,
    use_bf16=True,
    use_liger: bool = True,
    attn_implementation: str = "flash_attention_2",
    ac_offload_nvme=False,
    offload_dir="./offload_dir",
):
    torch.cuda.empty_cache()
    gc.collect()

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"We are using {dtype} to train the model.")

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
    model_cls = AutoLigerKernelForCausalLM if use_liger else AutoModelForCausalLM
    base_model = model_cls.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SlideFormerOffloader(
        model=base_model,
        device=device,
        dtype=dtype,
        ac_offload_nvme=ac_offload_nvme,
        offload_dir=offload_dir,
    )

    train_dataset = DummyDataset(
        size=1024, tokenizer=tokenizer, max_length=max_seq_length
    )
    print(f"Dataset size: {len(train_dataset)}")
    sample = train_dataset[0]
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True
    )
    model.train()

    epoch_steps = len(train_dataloader)
    total_flops = calculate_flops_per_batch(base_model, train_batch_size, max_seq_length)
    print(f"Per Batch FLOPS: {total_flops / 1e12:.2f} TFLOPS")
    tokens_per_batch = train_batch_size * max_seq_length

    gc.collect()
    torch.cuda.empty_cache()

    log_step = 1

    def trace_handler(prof: torch.profiler.profile):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("./profile", exist_ok=True)
        prof.export_chrome_trace(f"./profile/trace_{timestamp}.json")
        prof.export_memory_timeline(f"./profile/mem_{timestamp}.html", device="cuda:0")

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        running_loss = 0

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # step 0: wait, step 1-3: warmup, step 4: active (record), step 5: handler fires
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=1, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            on_trace_ready=trace_handler,
        ) as prof:
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

                prof.step()
                # wait=1, warmup=3, active=1 → handler fires after step 5
                if step >= 5:
                    break

        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s, "
            f"Average tokens/s: {tokens_per_batch * len(train_dataloader) / epoch_time:.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlideFormer torch.profiler entry.")
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
        offload_dir=args.offload_dir,
    )
