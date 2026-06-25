"""Execution DAG: nodes = ops, edges = dependencies."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class OpType(Enum):
    PREFETCH = "prefetch"
    COMPUTE = "compute"
    EVICT = "evict"


@dataclass
class OpNode:
    """A node in the execution graph."""
    id: int
    op_type: OpType
    tensor_ids: List[int]  # Tensors involved
    stream_id: int = 0  # Which stream to execute on
    deps: List[int] = field(default_factory=list)  # Node IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionGraph:
    """DAG of operations for explicit scheduling."""

    def __init__(self):
        self.nodes: Dict[int, OpNode] = {}
        self._next_id = 0

    def add_prefetch(
        self,
        tensor_ids: List[int],
        stream_id: int = 0,
        deps: Optional[List[int]] = None
    ) -> int:
        """Add a prefetch operation."""
        return self._add_node(OpType.PREFETCH, tensor_ids, stream_id, deps)

    def add_compute(
        self,
        tensor_ids: List[int],
        stream_id: int = 0,
        deps: Optional[List[int]] = None,
        **metadata
    ) -> int:
        """Add a compute operation."""
        node_id = self._add_node(OpType.COMPUTE, tensor_ids, stream_id, deps)
        self.nodes[node_id].metadata = metadata
        return node_id

    def add_evict(
        self,
        tensor_ids: List[int],
        stream_id: int = 0,
        deps: Optional[List[int]] = None
    ) -> int:
        """Add an evict operation."""
        return self._add_node(OpType.EVICT, tensor_ids, stream_id, deps)

    def _add_node(
        self,
        op_type: OpType,
        tensor_ids: List[int],
        stream_id: int,
        deps: Optional[List[int]]
    ) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = OpNode(
            id=node_id,
            op_type=op_type,
            tensor_ids=tensor_ids,
            stream_id=stream_id,
            deps=deps or []
        )
        return node_id

    def get_ready_nodes(self, completed: set) -> List[OpNode]:
        """Get nodes whose dependencies are all satisfied."""
        ready = []
        for node in self.nodes.values():
            if node.id not in completed:
                if all(d in completed for d in node.deps):
                    ready.append(node)
        return ready

    def topological_order(self) -> List[OpNode]:
        """Return nodes in topological order."""
        completed = set()
        order = []
        while len(order) < len(self.nodes):
            ready = self.get_ready_nodes(completed)
            if not ready:
                raise RuntimeError("Cycle detected in execution graph")
            for node in ready:
                order.append(node)
                completed.add(node.id)
        return order
