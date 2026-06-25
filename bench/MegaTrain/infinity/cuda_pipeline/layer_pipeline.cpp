#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Async copy parameters from CPU to GPU buffer
void async_copy_params(
    const std::vector<torch::Tensor>& cpu_params,
    const std::vector<torch::Tensor>& gpu_params,
    c10::cuda::CUDAStream stream
) {
    c10::cuda::CUDAStreamGuard guard(stream);
    for (size_t i = 0; i < cpu_params.size(); ++i) {
        gpu_params[i].copy_(cpu_params[i], /*non_blocking=*/true);
    }
}

// Async copy gradients from GPU to CPU
void async_copy_grads(
    const std::vector<torch::Tensor>& gpu_params,
    const std::vector<torch::Tensor>& cpu_params,
    c10::cuda::CUDAStream stream
) {
    c10::cuda::CUDAStreamGuard guard(stream);
    for (size_t i = 0; i < gpu_params.size(); ++i) {
        if (gpu_params[i].grad().defined()) {
            if (!cpu_params[i].grad().defined()) {
                cpu_params[i].mutable_grad() = torch::zeros_like(cpu_params[i]);
            }
            cpu_params[i].mutable_grad().add_(gpu_params[i].grad().cpu());
            gpu_params[i].mutable_grad().reset();
        }
    }
}

// Pipeline forward pass with double buffering
torch::Tensor pipeline_forward(
    torch::Tensor hidden,
    const std::vector<std::vector<torch::Tensor>>& cpu_layers_params,
    const std::vector<std::vector<torch::Tensor>>& gpu_buffer0_params,
    const std::vector<std::vector<torch::Tensor>>& gpu_buffer1_params,
    py::object layer_forward_fn,
    torch::Tensor attention_mask,
    py::object position_embeddings,
    int64_t checkpoint_interval,
    std::vector<torch::Tensor>& checkpoints
) {
    auto compute_stream = c10::cuda::getCurrentCUDAStream();
    auto weight_stream = c10::cuda::getStreamFromPool(false, hidden.device().index());

    int num_layers = cpu_layers_params.size();

    // Preload first layer into buffer 0
    async_copy_params(cpu_layers_params[0], gpu_buffer0_params[0], weight_stream);
    weight_stream.synchronize();

    for (int i = 0; i < num_layers; ++i) {
        int buffer_idx = i % 2;
        int next_buffer_idx = (i + 1) % 2;

        // Checkpoint before layer
        if (i % checkpoint_interval == 0) {
            checkpoints.push_back(hidden.detach());
        }

        // Async prefetch next layer
        if (i + 1 < num_layers) {
            auto& next_gpu_params = (next_buffer_idx == 0) ? gpu_buffer0_params[i + 1] : gpu_buffer1_params[i + 1];
            async_copy_params(cpu_layers_params[i + 1], next_gpu_params, weight_stream);
        }

        // Compute current layer (wait for its weights)
        compute_stream.synchronize();

        // Call Python forward function
        auto result = layer_forward_fn(
            hidden,
            py::arg("attention_mask") = attention_mask,
            py::arg("position_ids") = py::none(),
            py::arg("position_embeddings") = position_embeddings,
            py::arg("use_cache") = false,
            py::arg("output_attentions") = false
        );

        if (py::isinstance<py::tuple>(result)) {
            hidden = result.cast<py::tuple>()[0].cast<torch::Tensor>();
        } else {
            hidden = result.cast<torch::Tensor>();
        }

        // Wait for next layer's weights
        if (i + 1 < num_layers) {
            compute_stream.wait_stream(weight_stream);
        }
    }

    // Final checkpoint
    checkpoints.push_back(hidden.detach());

    return hidden;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("async_copy_params", &async_copy_params, "Async copy parameters from CPU to GPU");
    m.def("async_copy_grads", &async_copy_grads, "Async copy gradients from GPU to CPU");
    m.def("pipeline_forward", &pipeline_forward, "Pipeline forward pass with double buffering");
}
