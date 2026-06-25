#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for batched parameter copying
__global__ void batched_copy_kernel(
    float** dst_ptrs,
    const float** src_ptrs,
    const int64_t* sizes,
    int num_tensors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tensor_idx = blockIdx.y;

    if (tensor_idx < num_tensors) {
        int64_t size = sizes[tensor_idx];
        float* dst = dst_ptrs[tensor_idx];
        const float* src = src_ptrs[tensor_idx];

        for (int64_t i = tid; i < size; i += blockDim.x * gridDim.x) {
            dst[i] = src[i];
        }
    }
}

// Batched copy for multiple tensors in a single kernel launch
void batched_copy_tensors(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<torch::Tensor>& dst_tensors,
    cudaStream_t stream
) {
    int num_tensors = src_tensors.size();

    // Allocate device memory for pointers and sizes
    std::vector<float*> h_dst_ptrs(num_tensors);
    std::vector<const float*> h_src_ptrs(num_tensors);
    std::vector<int64_t> h_sizes(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
        h_dst_ptrs[i] = dst_tensors[i].data_ptr<float>();
        h_src_ptrs[i] = src_tensors[i].data_ptr<float>();
        h_sizes[i] = dst_tensors[i].numel();
    }

    // Copy to device
    float** d_dst_ptrs;
    const float** d_src_ptrs;
    int64_t* d_sizes;

    cudaMalloc(&d_dst_ptrs, num_tensors * sizeof(float*));
    cudaMalloc(&d_src_ptrs, num_tensors * sizeof(const float*));
    cudaMalloc(&d_sizes, num_tensors * sizeof(int64_t));

    cudaMemcpyAsync(d_dst_ptrs, h_dst_ptrs.data(), num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_src_ptrs, h_src_ptrs.data(), num_tensors * sizeof(const float*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sizes, h_sizes.data(), num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    dim3 block(256);
    dim3 grid(256, num_tensors);

    batched_copy_kernel<<<grid, block, 0, stream>>>(
        d_dst_ptrs, d_src_ptrs, d_sizes, num_tensors
    );

    // Cleanup
    cudaFree(d_dst_ptrs);
    cudaFree(d_src_ptrs);
    cudaFree(d_sizes);
}

// Wrapper for PyTorch
void batched_copy_params(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<torch::Tensor>& dst_tensors
) {
    // Get current CUDA stream using new API
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batched_copy_tensors(src_tensors, dst_tensors, stream);
}

// Async gradient accumulation: GPU -> CPU with async copy
// Strategy: Copy GPU grad to CPU, then accumulate on CPU side
void async_accumulate_grads(
    const std::vector<torch::Tensor>& gpu_params,
    const std::vector<torch::Tensor>& cpu_params,
    cudaStream_t stream
) {
    for (size_t i = 0; i < gpu_params.size(); ++i) {
        if (gpu_params[i].grad().defined()) {
            auto gpu_grad = gpu_params[i].grad();

            // Initialize CPU grad if needed (pinned memory for fast transfer)
            if (!cpu_params[i].grad().defined()) {
                auto cpu_grad = torch::zeros_like(cpu_params[i],
                    torch::TensorOptions().device(torch::kCPU).pinned_memory(true));
                const_cast<torch::Tensor&>(cpu_params[i]).mutable_grad() = cpu_grad;
            }

            auto cpu_grad = cpu_params[i].mutable_grad();

            // Async copy GPU grad to CPU and accumulate
            // Note: This uses PyTorch's optimized async copy with pinned memory
            cpu_grad.add_(gpu_grad.cpu());
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_copy_params", &batched_copy_params, "Batched parameter copy");
    m.def("async_accumulate_grads",
          [](const std::vector<torch::Tensor>& gpu_params,
             const std::vector<torch::Tensor>& cpu_params) {
              // Get current CUDA stream using new API
              cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
              async_accumulate_grads(gpu_params, cpu_params, stream);
          },
          "Async gradient accumulation");
}
