# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-node full-layer data parallel training with rank 0 CPU updates."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model name or local checkpoint")
    source.add_argument("--tiny", action="store_true", help="Random tiny Llama with synthetic tokens, for smoke checks")
    parser.add_argument("--tokens", help="torch.save file containing a 2-D int64 input_ids tensor")
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences per rank")
    parser.add_argument("--seq-length", type=int, default=32, help="Synthetic sequence length for --tiny")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU threads per rank; total is ranks times this value")
    parser.add_argument("--chunk-numel", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--double-buffer", action="store_true")
    parser.add_argument("--tied", action="store_true", help="Use tied embeddings with --tiny")
    parser.add_argument("--save-model", help="Save final model weights; optimizer resume is not supported")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.cpu_threads, args.chunk_numel) < 1 or args.seq_length < 2:
        parser.error("steps, batch-size, cpu-threads and chunk-numel must be positive; seq-length must be >= 2")
    if args.model and not args.tokens:
        parser.error("--model requires --tokens; synthetic data is only used with --tiny")

    # The existing layer module creates CUDA streams at import time.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(args.cpu_threads)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("This entry point supports a single node only")
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    from offload_transformer import SlideFormerOffloader
    from shared_data_parallel import SharedDataParallel

    torch.manual_seed(1234)
    dtype = torch.bfloat16
    if args.tiny:
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=max(128, args.seq_length),
                             tie_word_embeddings=args.tied, attention_dropout=0.0)
        config._attn_implementation = "eager"
        base_model = LlamaForCausalLM(config).to(dtype=dtype)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map="cpu", attn_implementation="eager",
        )
    base_model.config.use_cache = False
    if args.tokens:
        tokens = torch.load(args.tokens, map_location="cpu", weights_only=True)
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError("--tokens must contain a 2-D int64 tensor with sequence length >= 2")
        if tokens.numel() == 0 or tokens.min() < 0 or tokens.max() >= base_model.config.vocab_size:
            raise ValueError("Token IDs must be nonempty and within the model vocabulary")
    else:
        generator = torch.Generator().manual_seed(4321)
        tokens = torch.randint(base_model.config.vocab_size,
                               (args.steps * world * args.batch_size, args.seq_length), generator=generator)
    dataset = TensorDataset(tokens)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    if not len(loader):
        raise ValueError("Dataset must provide at least one full batch per rank")

    dp = SharedDataParallel(chunk_numel=args.chunk_numel, dtype=dtype) if world > 1 else None
    model = SlideFormerOffloader(
        base_model, device=torch.device("cuda", local_rank), dtype=dtype,
        double_buffer=args.double_buffer, enable_timing=False, enable_memory_stats=False,
        auto_backward_in_forward=False, shared_dp=dp,
        optimizer_kwargs=dict(lr=args.lr, weight_decay=0.01, fp32_optimizer_state=True,
                              num_layer=len(base_model.get_decoder().layers) + 2),
    )
    model.train()
    initial = [layer._cpu_params_flat.clone() for layer in model.transformer_layers] if args.tiny else None
    # Make stochastic layers independent after constructing identical initial weights.
    torch.manual_seed(1234 + rank)
    completed, epoch = 0, 0
    while completed < args.steps:
        sampler.set_epoch(epoch)
        for (input_ids,) in loader:
            input_ids = input_ids.to(model.device)
            output = model(input_ids, labels=input_ids)
            loss = output.loss
            health = torch.tensor(int(torch.isfinite(loss)), dtype=torch.int64)
            dist.all_reduce(health, op=dist.ReduceOp.MIN)
            if not health.item():
                raise RuntimeError("Nonfinite loss on at least one rank")
            loss.backward()
            model.wait_for_completion()
            mean_loss = loss.detach().float().cpu()
            dist.all_reduce(mean_loss)
            mean_loss /= world
            completed += 1
            if rank == 0:
                print(f"step={completed} global_loss={mean_loss.item():.6f} ranks={world}", flush=True)
            if completed == args.steps:
                break
        epoch += 1
    if args.tiny:
        for before, layer in zip(initial, model.transformer_layers):
            after = layer._cpu_params_flat
            if not torch.isfinite(after).all() or torch.equal(before, after):
                raise RuntimeError(f"Layer {layer.layer_idx} has invalid or unchanged weights")
            if dp is not None and layer._dp_state.step != completed:
                raise RuntimeError("Incorrect optimizer step count")
        print(f"rank={rank} tiny_weight_checks=passed", flush=True)
    if args.save_model:
        model.save_pretrained(args.save_model)
    model.wait_for_completion()
    model.update_executor.shutdown()
    model.h2d_executor.shutdown()
    model.d2h_executor.shutdown()
    if dp is not None:
        dp.barrier()
        dp.close()
    dist.barrier()
    print(f"rank={rank} completed={completed}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
