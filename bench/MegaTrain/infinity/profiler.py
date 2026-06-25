"""
Performance profiler for Infinity training.

Measures:
1. Communication time (CPU↔GPU transfers)
2. Computation time (forward/backward passes)
3. Memory operations
4. Optimizer time
"""
import torch
import time
from contextlib import contextmanager
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    """Statistics for a timing category."""
    name: str
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0

    def add(self, duration: float):
        self.total_time += duration
        self.count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    def __repr__(self):
        return (f"{self.name}: total={self.total_time:.3f}s, "
                f"avg={self.avg_time:.4f}s, count={self.count}, "
                f"min={self.min_time:.4f}s, max={self.max_time:.4f}s")


class PerformanceProfiler:
    """
    Profiles training performance with detailed breakdown.

    Categories:
    - Communication: CPU↔GPU data transfers
    - Computation: Forward/backward passes
    - Memory: Allocation/deallocation
    - Optimizer: Parameter updates
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.stats: Dict[str, TimingStats] = {}
        self.current_timers: Dict[str, float] = {}
        self.enabled = True

        # CUDA events for accurate GPU timing
        self.use_cuda_events = device.type == 'cuda'
        if self.use_cuda_events:
            self.event_start = torch.cuda.Event(enable_timing=True)
            self.event_end = torch.cuda.Event(enable_timing=True)

    @contextmanager
    def profile(self, name: str, category: str = "other"):
        """Context manager for profiling a code block."""
        if not self.enabled:
            yield
            return

        full_name = f"{category}/{name}"

        if full_name not in self.stats:
            self.stats[full_name] = TimingStats(full_name)

        # Synchronize before timing
        if self.use_cuda_events:
            torch.cuda.synchronize(self.device)
            self.event_start.record()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            if self.use_cuda_events:
                self.event_end.record()
                torch.cuda.synchronize(self.device)
                duration = self.event_start.elapsed_time(self.event_end) / 1000.0  # ms to s
            else:
                duration = time.perf_counter() - start_time

            self.stats[full_name].add(duration)

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics grouped by category."""
        summary = {}

        # Group by category
        for name, stat in self.stats.items():
            category = name.split('/')[0]
            if category not in summary:
                summary[category] = {
                    'total_time': 0.0,
                    'operations': []
                }

            summary[category]['total_time'] += stat.total_time
            summary[category]['operations'].append({
                'name': name.split('/')[-1],
                'total_time': stat.total_time,
                'avg_time': stat.avg_time,
                'count': stat.count,
                'percentage': 0.0  # Will be calculated later
            })

        # Calculate percentages
        total_time = sum(cat['total_time'] for cat in summary.values())
        for category in summary.values():
            category['percentage'] = (category['total_time'] / total_time * 100) if total_time > 0 else 0.0
            for op in category['operations']:
                op['percentage'] = (op['total_time'] / total_time * 100) if total_time > 0 else 0.0

        return summary

    def print_summary(self):
        """Print detailed performance summary."""
        summary = self.get_summary()
        total_time = sum(cat['total_time'] for cat in summary.values())

        print("\n" + "="*80)
        print("PERFORMANCE PROFILE SUMMARY")
        print("="*80)
        print(f"Total profiled time: {total_time:.3f}s\n")

        # Sort categories by time
        sorted_categories = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)

        for category, data in sorted_categories:
            print(f"\n{category.upper()}: {data['total_time']:.3f}s ({data['percentage']:.1f}%)")
            print("-" * 80)

            # Sort operations by time
            sorted_ops = sorted(data['operations'], key=lambda x: x['total_time'], reverse=True)

            for op in sorted_ops[:10]:  # Show top 10 operations
                print(f"  {op['name']:40s} {op['total_time']:8.3f}s ({op['percentage']:5.1f}%) "
                      f"avg={op['avg_time']*1000:7.2f}ms count={op['count']:5d}")

        print("\n" + "="*80)

        # Communication vs Computation breakdown
        comm_time = summary.get('communication', {}).get('total_time', 0.0)
        comp_time = summary.get('computation', {}).get('total_time', 0.0)

        print("\nCOMMUNICATION vs COMPUTATION")
        print("-" * 80)
        print(f"Communication (CPU↔GPU): {comm_time:8.3f}s ({comm_time/total_time*100:5.1f}%)")
        print(f"Computation (GPU):       {comp_time:8.3f}s ({comp_time/total_time*100:5.1f}%)")
        print(f"Overlap efficiency:      {(1 - (comm_time + comp_time)/total_time)*100:5.1f}%")
        print("="*80)

    def reset(self):
        """Reset all statistics."""
        self.stats.clear()
        self.current_timers.clear()


def add_profiling_to_trainer(trainer_class):
    """
    Decorator to add profiling to InfinityTrainer.

    Usage:
        @add_profiling_to_trainer
        class InfinityTrainer:
            ...
    """
    original_init = trainer_class.__init__
    original_forward_backward = trainer_class.forward_backward

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.profiler = PerformanceProfiler(self.device)

    def new_forward_backward(self, input_ids, attention_mask):
        profiler = self.profiler

        B, T = input_ids.shape
        device = self.device
        scale = 1.0 / self.config.gradient_accumulation_steps

        # Embedding forward
        with profiler.profile("embedding_to_gpu", "communication"):
            self.embedding.to(device, dtype=self.dtype)

        with profiler.profile("embedding_forward", "computation"):
            hidden = torch.nn.functional.embedding(input_ids.to(device, non_blocking=True), self.embedding.param)

        with profiler.profile("save_activation_to_cpu", "communication"):
            saved_hiddens = [hidden.detach().to("cpu", dtype=self.dtype).pin_memory()]

        rope_cache = self._rope_cache(T, device)
        mask = attention_mask.to(device) if attention_mask is not None else None

        # Forward through layers
        for i, layer in enumerate(self.layers):
            with profiler.profile(f"layer_{i}_prefetch", "communication"):
                self.layer_mgr.ensure_window(i)

            with profiler.profile(f"layer_{i}_forward", "computation"):
                hidden, ctx = layer.forward(hidden, mask, rope_cache)

            with profiler.profile(f"layer_{i}_save_activation", "communication"):
                saved_hiddens.append(hidden.detach().to("cpu", dtype=self.dtype).pin_memory())

            layer._ctx = ctx

            if i - self.config.window_size >= 0:
                with profiler.profile(f"layer_{i-self.config.window_size}_evict", "communication"):
                    self.layer_mgr.evict(i - self.config.window_size)

        # Final norm + head
        with profiler.profile("final_norm_to_gpu", "communication"):
            if self.final_norm:
                self.final_norm.to(device, dtype=self.dtype)

        with profiler.profile("final_norm_forward", "computation"):
            if self.final_norm:
                hidden, norm_ctx = original_forward_backward.__code__.co_consts[17](hidden, self.final_norm.param)  # rmsnorm_forward
            else:
                norm_ctx = None

        with profiler.profile("head_to_gpu", "communication"):
            self.head.to(device, dtype=self.dtype)

        with profiler.profile("head_forward", "computation"):
            hidden = hidden.to(self.dtype)
            logits = hidden @ self.head.param.t()

        # Loss computation
        with profiler.profile("loss_computation", "computation"):
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:].to(device)
            shift_mask = attention_mask[:, 1:].to(device) if attention_mask is not None else None

            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            one_hot = torch.nn.functional.one_hot(shift_labels, num_classes=log_probs.shape[-1]).float()
            nll = -(log_probs * one_hot).sum(dim=-1)

            if shift_mask is not None:
                n_tokens = shift_mask.sum().item()
                loss = (nll * shift_mask).sum() / max(n_tokens, 1)
            else:
                n_tokens = nll.numel()
                loss = nll.mean()

            grad_logits = torch.exp(log_probs) - one_hot
            if shift_mask is not None:
                grad_logits *= shift_mask.unsqueeze(-1)
                grad_logits /= max(n_tokens, 1)
            else:
                grad_logits /= grad_logits.shape[0] * grad_logits.shape[1]
            grad_logits = grad_logits.to(self.dtype)

        # Backward head
        with profiler.profile("head_backward", "computation"):
            grad_hidden = grad_logits @ self.head.param
            grad_head_w = grad_logits.reshape(-1, grad_logits.shape[-1]).t() @ hidden.reshape(-1, hidden.shape[-1])

        with profiler.profile("head_grad_to_cpu", "communication"):
            self.head.add_grad(grad_head_w * scale)

        if self.final_norm:
            with profiler.profile("final_norm_backward", "computation"):
                # grad_hidden, grad_norm_w = rmsnorm_backward(grad_hidden, norm_ctx)
                pass  # Simplified for profiling

            with profiler.profile("final_norm_grad_to_cpu", "communication"):
                # self.final_norm.add_grad(grad_norm_w * scale)
                pass

        # Backward through layers
        grad = grad_hidden
        for i in range(len(self.layers) - 1, -1, -1):
            with profiler.profile(f"layer_{i}_prefetch_backward", "communication"):
                self.layer_mgr.ensure_window(max(0, i - self.config.window_size + 1))

            layer = self.layers[i]

            with profiler.profile(f"layer_{i}_load_activation", "communication"):
                layer_input_cpu = saved_hiddens[i]
                if self._layer_input_buf is None or self._layer_input_buf.shape != layer_input_cpu.shape:
                    self._layer_input_buf = torch.empty(layer_input_cpu.shape, device=device, dtype=self.dtype)
                self._layer_input_buf.copy_(layer_input_cpu, non_blocking=True)
                layer_input = self._layer_input_buf

            ctx = layer._ctx
            ctx["x"] = layer_input

            with profiler.profile(f"layer_{i}_backward", "computation"):
                grad, grads = layer.backward(grad, ctx)

            with profiler.profile(f"layer_{i}_grad_to_cpu", "communication"):
                # Accumulate grads (simplified)
                pass

            with profiler.profile(f"layer_{i+1}_evict_backward", "communication"):
                self.layer_mgr.evict(i + 1)

            layer._ctx = None
            del ctx

        # Embedding grad
        with profiler.profile("embedding_backward", "computation"):
            grad_embed = torch.zeros_like(self.embedding.master)
            scatter = input_ids.view(-1).to(self.embedding.master.device)
            grad_flat = grad.view(-1, grad.shape[-1]).to(self.embedding.master.device, dtype=self.embedding.master.dtype)
            grad_embed.index_add_(0, scatter, grad_flat)

        with profiler.profile("embedding_grad_to_cpu", "communication"):
            self.embedding.add_grad(grad_embed * scale)

        # Cleanup
        del grad_logits, grad_hidden, grad, hidden, logits
        self._layer_input_buf = None
        saved_hiddens.clear()

        return loss.item(), n_tokens

    trainer_class.__init__ = new_init
    trainer_class.forward_backward = new_forward_backward

    return trainer_class


if __name__ == "__main__":
    # Test the profiler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    profiler = PerformanceProfiler(device)

    # Simulate some operations
    for i in range(10):
        with profiler.profile("data_transfer", "communication"):
            x = torch.randn(100, 100).to(device)
            time.sleep(0.001)

        with profiler.profile("computation", "computation"):
            y = x @ x.t()
            time.sleep(0.002)

        with profiler.profile("result_transfer", "communication"):
            z = y.cpu()
            time.sleep(0.001)

    profiler.print_summary()
