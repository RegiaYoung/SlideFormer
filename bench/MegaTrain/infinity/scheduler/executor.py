"""Executor: walks DAG, issues copy/compute/evict commands."""

from typing import Dict, List, Callable, Any
from .graph import ExecutionGraph, OpNode, OpType
from ..runtime import ManagedTensor, Stream


class Executor:
    """Executes an operation graph with explicit stream control."""

    def __init__(
        self,
        tensors: Dict[int, ManagedTensor],
        streams: Dict[int, Stream],
        compute_fn: Callable[[OpNode, Dict[int, ManagedTensor]], Any]
    ):
        self.tensors = tensors
        self.streams = streams
        self.compute_fn = compute_fn  # User-provided compute dispatch

    def run(self, graph: ExecutionGraph) -> None:
        """Execute the graph in topological order."""
        completed = set()
        events: Dict[int, Any] = {}  # node_id -> cuda event

        for node in graph.topological_order():
            stream = self.streams.get(node.stream_id)

            # Wait for dependencies on different streams
            for dep_id in node.deps:
                dep_node = graph.nodes[dep_id]
                if dep_node.stream_id != node.stream_id and dep_id in events:
                    stream.wait_event(events[dep_id])

            # Execute based on op type
            if node.op_type == OpType.PREFETCH:
                self._do_prefetch(node, stream)
            elif node.op_type == OpType.COMPUTE:
                self._do_compute(node, stream)
            elif node.op_type == OpType.EVICT:
                self._do_evict(node, stream)

            # Record event for downstream dependencies
            if stream:
                events[node.id] = stream.record_event()

            completed.add(node.id)

    def _do_prefetch(self, node: OpNode, stream: Stream) -> None:
        for tid in node.tensor_ids:
            if tid in self.tensors:
                self.tensors[tid].prefetch(stream)

    def _do_compute(self, node: OpNode, stream: Stream) -> None:
        with stream:
            self.compute_fn(node, self.tensors)

    def _do_evict(self, node: OpNode, stream: Stream) -> None:
        for tid in node.tensor_ids:
            if tid in self.tensors:
                self.tensors[tid].evict(stream)
