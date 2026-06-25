# Evaluation Data of LoHan

Here we provide the major evaluation results in our paper (while the others generally reuse these experimental data when evaluating LoHan). You can use this to check the correctness of your results. All the throughput data are measured in TFLOPS, which is directly output by the script. 

All the data here are produced under the testbed whose configuration is listed below.

| CPU         | Dual Intel Xeon Gold 5320 CPU               |
|-------------|---------------------------------------------|
| Main Memory | 768 GB 3200 MHz DDR4 (16 channels in total) |
| GPU         | NVIDIA GeForce RTX 4090                     |
| SSD         | 12x D7-P5510 3.84 TB SSD                    |

---

**Figure 5(a)/7(a)**: End-to-end performance, single 4090 GPU.

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| Batch Size | 8    | 16   | 32    | 64    | 128   |
|------------|------|------|-------|-------|-------|
| TFLOPS     | 42.8 | 84.3 | 143.1 | 155.8 | 153.8 |

---

**Figure 7(b)**: End-to-end performance, single 4090 GPU. 

Model Configuration:

| #Params             | #Layers | #Heads | #Hidden Dimemsion |
|---------------------|---------|--------|-------------------|
| 1.75$\times10^{11}$ | 96      | 96     | 12288             |

Result: 

| Batch Size | 8    | 16   | 32  |
|------------|------|------|-----|
| TFLOPS     | 52.6 | 86.9 | OOM |

---

**Figure 10(b)**: Throughput w.r.t. number of SSDs, single 4090 GPU. 

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| #SSDs  | 1    | 2     | 3     | 6     | 12    |
|--------|------|-------|-------|-------|-------|
| bsz=32 | 37.5 | 64.3  | 81.1  | 121.7 | 142.0 |
| bsz=48 | 53.1 | 89.7  | 121.7 | 146.3 | 153.9 |
| bsz=64 | 70.3 | 111.7 | 136.3 | 151.5 | 148.2 |

---

**Figure 11(a)**: End-to-end performance, 2x 4090 GPUs. 

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| Global Batch Size | 16   | 32    | 64    | 128   |
|-------------------|------|-------|-------|-------|
| Global TFLOPS     | 55.0 | 103.2 | 194.7 | 278.2 |

---

**Figure 11(b)**: End-to-end performance, 2x 4090 GPUs. 

Model Configuration:

| #Params          | #Layers | #Heads | #Hidden Dimemsion |
|------------------|---------|--------|-------------------|
| 7$\times10^{10}$ | 80      | 64     | 8192              |

Result: 

| Global Batch Size | 16   | 32    | 48    |
|-------------------|------|-------|-------|
| Global TFLOPS     | 64.6 | 128.8 | 183.5 |

---

**Figure 11(c)**: End-to-end performance, 4x 4090 GPUs. 

Model Configuration:

| #Params            | #Layers | #Heads | #Hidden Dimemsion |
|--------------------|---------|--------|-------------------|
| 1.3$\times10^{10}$ | 40      | 40     | 5120              |

Result: 

| Global Batch Size | 32    | 64    | 128   | 256   |
|-------------------|-------|-------|-------|-------|
| Global TFLOPS     | 106.5 | 209.7 | 358.7 | 514.4 |

---

**Figure 11(d)**: End-to-end performance, 4x 4090 GPUs. 

Model Configuration:

| #Params          | #Layers | #Heads | #Hidden Dimemsion |
|------------------|---------|--------|-------------------|
| 7$\times10^{10}$ | 80      | 64     | 8192              |

Result: 

| Global Batch Size | 32    | 64    | 96    |
|-------------------|-------|-------|-------|
| Global TFLOPS     | 124.8 | 249.7 | 348.4 |
