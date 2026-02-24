#!/bin/bash


export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=64

# colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -g -x -b 16 --offload_optim_frac 1 --offload_param_frac 1

colossalai run --nproc_per_node 1 --hostfile ./hosts.txt benchmark.py -p gemini -c qwen2.5-14b -g -x -b 1 -l 1024 --offload_optim_frac 1.0 --offload_param_frac 1.0