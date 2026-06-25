# LoHan ICDE 2025 Artifact

This artifact provides a guide to replicate the primary experiments in this paper. You can follow this repository to reproduce the experimental results about LoHan's maximum trainable model sizes, batch sizes and throughput in our paper. The documentation and auto-run script mainly focus on reproducing results in Subsection V-B and you can adjust the code to reproduce results in other sections. 

## Environment Setup

### SSD Configuration

LoHan aggregates the I/O bandwidth of multiple SSDs by configuring a RAID array for efficient model states and activation offloading. Therefore, we provide a script to configure this array.

First, modify the `make_raid.sh` to meet your own needs. The script in this repo is used to configure the drives `/dev/nvme0n1` to `/dev/nvme11n1` into an array. You can adjust the line 23 to change the drives you want to set up.

After configuring the script, you can run the script to set up the RAID array. You might need a root permission to do so:

```shell
./make_raid.sh
```

### Installing the Python packages

```shell
conda create -n lohan python=3.10
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# If there are different CUDA versions, you should specify the CUDA version
# export CUDA_HOME=/usr/local/cuda-11.8
pip install flash-attn==1.0.4

# The following two packages are to fulfill the requirements of existing packages
pip install six==1.16.0
pip install scikit-learn
```

## Running LoHan

We provide a script to run LoHan. You can adjust the script to reproduce the results. 

```shell
bash run.sh
```

### Limiting the Memory Size

Experiments in Subsection V-B require adjusting the main memory capacity. Instead of manually adding and removing the machine's DRAM, you can consider pinning the main memory via huge page so that these memory spaces cannot be utilized by Ratel. 

You can use the following script (root permission required) to pin the main memory

```shell
sh -c "echo 1024 > /proc/sys/vm/nr_hugepages"
```

The **1024** means you set the **HugePages_num** to **1024**. Each HugePageSize is **2MB**. Therefore the total memory size pinned by huge page in this script is **2MB * 1024 = 2048MB**.

You can check the pinned memory by using the following command.

```shell
$ cat /proc/meminfo | grep Huge
```

For example, the following output indicates the memory pinned by huge page is 1024(HugePages_Total)*2048kB(Hugepagesize)=2GB.

```
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:    1024
HugePages_Free:     1024
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:         2097152 kB
```

### Benchmark Results

Please refer to [here](evaluation_data.md) for our raw evaluation data in our paper that might help for your reproduing. 

## Acknowledgement

Some of the code in this project is modified from the [DeepSpeed](https://github.com/microsoft/DeepSpeed) repository, we appreciate the contributions of the original repository authors.

* op_ds/accelerator/real_accelerator.py
* op_ds/ops/op_builder/all_ops.py
* op_ds/ops/op_builder/builder.py
* op_ds/ops/CPUAdam.py
* nvme_ds
