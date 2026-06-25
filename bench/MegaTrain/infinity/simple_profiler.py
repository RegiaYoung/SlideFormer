"""
Simplified profiler that can be easily integrated into the training script.

Usage:
    from infinity.simple_profiler import SimpleProfiler

    profiler = SimpleProfiler(device)

    # Profile communication
    with profiler.time("cpu_to_gpu", "comm"):
        tensor = tensor.to(device)

    # Profile computation
    with profiler.time("forward", "comp"):
        output = model(input)

    # Print summary
    profiler.print_summary()
"""
import torch
import time
from contextlib import contextmanager
from collections import defaultdict


class SimpleProfiler:
    """Lightweight profiler for measuring communication vs computation time."""

    def __init__(self, device):
        self.device = device
        self.times = defaultdict(list)
        self.categories = {}
        self.use_cuda = device.type == 'cuda'

    @contextmanager
    def time(self, name, category="other"):
        """Time a code block."""
        self.categories[name] = category

        if self.use_cuda:
            torch.cuda.synchronize(self.device)

        start = time.perf_counter()
        yield

        if self.use_cuda:
            torch.cuda.synchronize(self.device)

        duration = time.perf_counter() - start
        self.times[name].append(duration)

    def get_stats(self):
        """Get timing statistics."""
        stats = {}
        for name, times_list in self.times.items():
            stats[name] = {
                'total': sum(times_list),
                'avg': sum(times_list) / len(times_list),
                'count': len(times_list),
                'category': self.categories.get(name, 'other')
            }
        return stats

    def print_summary(self):
        """Print timing summary."""
        stats = self.get_stats()

        # Group by category
        comm_time = sum(s['total'] for s in stats.values() if s['category'] == 'comm')
        comp_time = sum(s['total'] for s in stats.values() if s['category'] == 'comp')
        other_time = sum(s['total'] for s in stats.values() if s['category'] == 'other')
        total_time = comm_time + comp_time + other_time

        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Total time: {total_time:.3f}s")
        print(f"  Communication: {comm_time:.3f}s ({comm_time/total_time*100:.1f}%)")
        print(f"  Computation:   {comp_time:.3f}s ({comp_time/total_time*100:.1f}%)")
        print(f"  Other:         {other_time:.3f}s ({other_time/total_time*100:.1f}%)")
        print("="*70)

        # Top operations
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        print("\nTop 10 operations:")
        for name, s in sorted_stats[:10]:
            cat = s['category']
            print(f"  [{cat:4s}] {name:30s} {s['total']:7.3f}s ({s['total']/total_time*100:5.1f}%) "
                  f"avg={s['avg']*1000:6.2f}ms n={s['count']}")
        print("="*70)
