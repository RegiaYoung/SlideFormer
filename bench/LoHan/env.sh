#!/bin/bash

# config the environment

conda create -n torch python=3.10
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# If there has different CUDA version, you should specify the CUDA version
# export CUDA_HOME=/usr/local/cuda-11.8
pip install flash-attn==1.0.4

# The following two packages are to fulfill the requirements of the ogb
pip install six==1.16.0
pip install scikit-learn


