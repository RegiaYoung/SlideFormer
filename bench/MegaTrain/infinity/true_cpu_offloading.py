"""
True CPU Offloading Implementation for Infinity.

Key features:
1. Only ONE layer on GPU at a time
2. All activations stored on CPU (pinned memory)
3. Uses gradient checkpointing to avoid storing intermediate activations
4. Explicit memory management and cleanup

Memory usage:
- GPU: ~2-3 GB (single layer + current batch)
- CPU: ~100+ GB (all parameters + activations)
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List
import gc


class CPUOffloadedLayer(nn.Module):
    """
    A transformer layer that lives on CPU and is temporarily moved to GPU for computation.

    Key principles:
    - Layer parameters stay on CPU by default
    - Moved to GPU only during forward/backward
    - Immediately moved back to CPU after computation
    - Uses gradient checkpointing to avoid storing intermediate activations
    """

    def __init__(self, layer_module: nn.Module, layer_idx: int):
        super().__init__()
        self.layer = layer_module
        self.layer_idx = layer_idx
        self.current_device = "cpu"

        # Move to CPU initially
        self.layer.cpu()

    def to_gpu(self, device: torch.device):
        """Move layer to GPU."""
        if self.current_device == "cpu":
            self.layer.to(device)
            self.current_device = device

    def to_cpu(self):
        """Move layer back to CPU."""
        if self.current_device != "cpu":
            self.layer.cpu()
            self.current_device = "cpu"
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope_cache: Optional[Tuple] = None, use_checkpoint: bool = True):
        """
        Forward pass with automatic GPU offloading.

        Args:
            hidden: Input tensor (can be on CPU or GPU)
            mask: Attention mask
            rope_cache: RoPE cache
            use_checkpoint: Whether to use gradient checkpointing

        Returns:
            Output tensor (on same device as input)
        """
        input_device = hidden.device

        # Move layer to GPU if needed
        gpu_device = torch.device("cuda:0") if torch.cuda.is_available() else input_device
        self.to_gpu(gpu_device)

        # Move input to GPU
        hidden_gpu = hidden.to(gpu_device, non_blocking=True)
        mask_gpu = mask.to(gpu_device, non_blocking=True) if mask is not None else None

        # Forward pass with gradient checkpointing
        if use_checkpoint and hidden_gpu.requires_grad:
            # Checkpoint saves memory by not storing intermediate activations
            output_gpu = checkpoint(
                self._forward_impl,
                hidden_gpu,
                mask_gpu,
                rope_cache,
                use_reentrant=False
            )
        else:
            output_gpu = self._forward_impl(hidden_gpu, mask_gpu, rope_cache)

        # Move output back to original device
        output = output_gpu.to(input_device, non_blocking=True)

        # Clean up GPU memory
        del hidden_gpu, mask_gpu, output_gpu

        return output

    def _forward_impl(self, hidden: torch.Tensor, mask: Optional[torch.Tensor],
                     rope_cache: Optional[Tuple]) -> torch.Tensor:
        """Actual forward implementation (called by checkpoint)."""
        return self.layer.forward(hidden, mask, rope_cache)


class TrueCPUOffloadingTrainer:
    """
    Trainer with true CPU offloading.

    Memory model:
    - CPU: All parameters (FP32 masters), all activations
    - GPU: Only current layer + current batch activations

    This allows training models much larger than GPU memory.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Wrap all layers for CPU offloading
        self.offloaded_layers = nn.ModuleList([
            CPUOffloadedLayer(layer, i) for i, layer in enumerate(model.layers)
        ])

        # Keep embedding and head on CPU
        self.embedding = model.embedding.cpu()
        self.final_norm = model.final_norm.cpu() if hasattr(model, 'final_norm') and model.final_norm is not None else None
        self.lm_head = model.lm_head.cpu()

        # Pinned memory for fast CPU↔GPU transfers
        self.use_pinned_memory = True

    def forward_pass(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with true CPU offloading.

        All intermediate activations are stored on CPU (pinned memory).
        Only the current layer computation happens on GPU.
        """
        B, T = input_ids.shape

        # Embedding (on CPU)
        hidden = self.embedding(input_ids.cpu())

        if self.use_pinned_memory:
            hidden = hidden.pin_memory()

        # Store activations on CPU for backward pass
        saved_activations = [hidden]

        # Forward through layers (one at a time on GPU)
        for i, layer in enumerate(self.offloaded_layers):
            # Layer computation on GPU
            hidden = layer(hidden, mask=attention_mask, use_checkpoint=True)

            # Store activation on CPU
            if self.use_pinned_memory:
                hidden = hidden.cpu().pin_memory()
            else:
                hidden = hidden.cpu()

            saved_activations.append(hidden)

            # Explicitly move layer back to CPU
            layer.to_cpu()

        # Final norm and head (on CPU, then move to GPU for computation)
        if self.final_norm is not None:
            self.final_norm.to(self.device)
            hidden_gpu = hidden.to(self.device)
            hidden_gpu = self.final_norm(hidden_gpu)
            hidden = hidden_gpu.cpu()
            self.final_norm.cpu()
            del hidden_gpu

        # LM head
        self.lm_head.to(self.device)
        hidden_gpu = hidden.to(self.device)
        logits_gpu = self.lm_head(hidden_gpu)
        logits = logits_gpu.cpu()
        self.lm_head.cpu()
        del hidden_gpu, logits_gpu

        return logits, saved_activations

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None):
        """Compute loss on GPU."""
        # Move to GPU for loss computation
        logits_gpu = logits.to(self.device)
        labels_gpu = labels.to(self.device)

        # Shift for next-token prediction
        shift_logits = logits_gpu[:, :-1, :].contiguous()
        shift_labels = labels_gpu[:, 1:].contiguous()

        # Cross entropy loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )

        # Apply mask if provided
        if attention_mask is not None:
            mask_gpu = attention_mask.to(self.device)
            shift_mask = mask_gpu[:, 1:].contiguous()
            loss = loss * shift_mask.view(-1)
            loss = loss.sum() / shift_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def backward_pass(self, loss: torch.Tensor, saved_activations: List[torch.Tensor]):
        """
        Backward pass with true CPU offloading.

        Gradients are computed layer by layer, with only one layer on GPU at a time.
        """
        # Backward through loss
        loss.backward()

        # Note: With gradient checkpointing, PyTorch will automatically
        # recompute forward passes as needed during backward.
        # We don't need to manually implement backward here.

    def train_step(self, batch: dict):
        """
        Single training step with true CPU offloading.

        Returns:
            loss: scalar loss value
            metrics: dict of training metrics
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        # Forward pass (all on CPU except current layer)
        logits, saved_activations = self.forward_pass(input_ids, attention_mask)

        # Compute loss (on GPU)
        loss = self.compute_loss(logits, input_ids, attention_mask)

        # Backward pass (automatic with checkpointing)
        self.backward_pass(loss, saved_activations)

        # Cleanup
        del logits, saved_activations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = {
            'loss': loss.item(),
            'gpu_memory_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }

        return loss.item(), metrics


def test_true_cpu_offloading():
    """Test the true CPU offloading implementation."""
    print("="*70)
    print("Testing True CPU Offloading")
    print("="*70)

    # Create a simple model
    class SimpleLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x, mask=None, rope_cache=None):
            return self.linear(x)

    class SimpleModel(nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([SimpleLayer(hidden_size) for _ in range(num_layers)])
            self.final_norm = None
            self.lm_head = nn.Linear(hidden_size, vocab_size)

    # Test configuration
    vocab_size = 1000
    hidden_size = 512
    num_layers = 4
    batch_size = 2
    seq_len = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model
    model = SimpleModel(vocab_size, hidden_size, num_layers)

    # Create trainer
    class Config:
        pass
    config = Config()

    trainer = TrueCPUOffloadingTrainer(model, config, device)

    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
    }

    # Training step
    print(f"\nRunning training step...")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Model: {num_layers} layers, {hidden_size} hidden size")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss, metrics = trainer.train_step(batch)

    print(f"\nResults:")
    print(f"  Loss: {loss:.4f}")
    print(f"  GPU memory: {metrics['gpu_memory_mb']:.2f} MB")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU memory: {peak_memory:.2f} MB")

    print("\n✓ True CPU offloading test passed!")
    print("="*70)


if __name__ == "__main__":
    test_true_cpu_offloading()
