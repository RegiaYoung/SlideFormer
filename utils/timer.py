# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import torch


TIMING_LABELS: Dict[str, str] = {
    "h2d_transfer_fw": "h2d transfer fw",
    "h2d_transfer_bw": "h2d transfer bw",
    "d2h_transfer_fw": "d2h transfer fw",
    "d2h_transfer_bw": "d2h transfer bw",
    "forward": "forward",
    "backward": "backward",
    "parameter_update": "parameter update",
    "ac_fwd_d2h": "ac fwd d2h",
    "ac_bwd_h2d": "ac bwd h2d",
}


def format_timing_label(operation: str) -> str:
    return TIMING_LABELS.get(operation, operation.replace("_", " "))


class LayerTimer:
    """层级操作计时器"""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self._lock = threading.Lock()
        self.current_step: DefaultDict[str, float] = defaultdict(float)
        self.timings: DefaultDict[str, List[float]] = defaultdict(list)
        self._pending_cuda_spans: DefaultDict[
            str,
            List[Tuple[torch.cuda.Event, torch.cuda.Event]],
        ] = defaultdict(list)

    def record_time(self, operation: str, duration: float):
        with self._lock:
            self.current_step[operation] = duration
            self.timings[operation].append(duration)

    def record_cuda_span(
        self,
        operation: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
    ) -> None:
        with self._lock:
            self._pending_cuda_spans[operation].append((start_event, end_event))

    def flush_cuda_spans(self, force: bool = False) -> None:
        resolved: List[Tuple[str, float]] = []
        with self._lock:
            for operation, spans in list(self._pending_cuda_spans.items()):
                ready_spans: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
                remaining_spans: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

                for start_event, end_event in spans:
                    if force:
                        end_event.synchronize()
                        ready_spans.append((start_event, end_event))
                    elif end_event.query():
                        ready_spans.append((start_event, end_event))
                    else:
                        remaining_spans.append((start_event, end_event))

                if remaining_spans:
                    self._pending_cuda_spans[operation] = remaining_spans
                else:
                    self._pending_cuda_spans.pop(operation, None)

                if ready_spans:
                    total = sum(
                        start_event.elapsed_time(end_event) / 1000.0
                        for start_event, end_event in ready_spans
                    )
                    resolved.append((operation, total))

            for operation, duration in resolved:
                self.current_step[operation] = duration
                self.timings[operation].append(duration)

    def get_average(self, operation: str) -> float:
        self.flush_cuda_spans(force=True)
        times = self.timings[operation]
        return sum(times) / len(times) if times else 0.0

    def print_stats(self, show_history=False):
        self.flush_cuda_spans(force=True)
        print(f"\nLayer {self.layer_idx} Statistics:")
        if show_history:
            # 打印历史平均
            for op, times in self.timings.items():
                avg = sum(times) / len(times)
                print(f"  {format_timing_label(op)}: {avg*1000:.2f}ms (avg over {len(times)} calls)")
        else:
            # 只打印当前步
            for op, duration in self.current_step.items():
                print(f"  {format_timing_label(op)}: {duration*1000:.2f}ms")
