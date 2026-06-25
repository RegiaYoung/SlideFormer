#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Template-based batched copy supporting multiple dtypes
void batched_copy_params(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<torch::Tensor>& dst_tensors
) {
    // Get current CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();

    // Batch copy all tensors with non_blocking=true
    // PyTorch's copy_ handles dtype conversion automatically
    for (size_t i = 0; i < src_tensors.size(); ++i) {
        dst_tensors[i].copy_(src_tensors[i], /*non_blocking=*/true);
    }
}

// Async gradient accumulation supporting multiple dtypes
void async_accumulate_grads(
    const std::vector<torch::Tensor>& gpu_params,
    const std::vector<torch::Tensor>& cpu_params
) {
    for (size_t i = 0; i < gpu_params.size(); ++i) {
        if (gpu_params[i].grad().defined()) {
            // Initialize CPU grad if needed (pinned memory)
            if (!cpu_params[i].grad().defined()) {
                auto cpu_grad = torch::zeros_like(cpu_params[i],
                    torch::TensorOptions().device(torch::kCPU).pinned_memory(true));
                const_cast<torch::Tensor&>(cpu_params[i]).mutable_grad() = cpu_grad;
            }

            auto cpu_grad = cpu_params[i].mutable_grad();
            auto gpu_grad = gpu_params[i].grad();

            // Async D2H copy and accumulate
            // PyTorch handles dtype conversion automatically
            cpu_grad.add_(gpu_grad.to(torch::kCPU, /*non_blocking=*/true));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_copy_params", &batched_copy_params,
          "Batched parameter copy with non-blocking transfers (supports all dtypes)");
    m.def("async_accumulate_grads", &async_accumulate_grads,
          "Async gradient accumulation from GPU to CPU (supports all dtypes)");
}

