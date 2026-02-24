import time
from collections import defaultdict

class LayerTimer:
    """层级操作计时器"""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.current_step = defaultdict(float)  # 存储当前步的时间
        self.timings = defaultdict(list)  # 保留历史记录功能
        
    def record_time(self, operation: str, duration: float):
        self.current_step[operation] = duration  # 记录当前步
        self.timings[operation].append(duration)  # 保存历史
        
    def get_average(self, operation: str) -> float:
        times = self.timings[operation]
        return sum(times) / len(times) if times else 0.0
        
    def print_stats(self, show_history=False):
        print(f"\nLayer {self.layer_idx} Statistics:")
        if show_history:
            # 打印历史平均
            for op, times in self.timings.items():
                avg = sum(times) / len(times)
                print(f"  {op}: {avg*1000:.2f}ms (avg over {len(times)} calls)")
        else:
            # 只打印当前步
            for op, duration in self.current_step.items():
                print(f"  {op}: {duration*1000:.2f}ms")
