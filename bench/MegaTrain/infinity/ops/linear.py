"""Manual linear forward/backward with explicit scheduling."""

import torch
from typing import Dict, Tuple
from ..runtime import ManagedTensor
from ..scheduler import ExecutionGraph, OpNode


def linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None
) -> torch.Tensor:
    """y = x @ W^T + b. No autograd."""
    out = x @ weight.t()
    if bias is not None:
        out = out + bias
    return out


def linear_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual backward for linear layer.

    Returns: (grad_x, grad_weight, grad_bias)
    """
    grad_x = grad_output @ weight
    grad_weight = grad_output.t() @ x
    grad_bias = grad_output.sum(dim=0)
    return grad_x, grad_weight, grad_bias


def schedule_linear_forward(
    graph: ExecutionGraph,
    x_id: int,
    weight_id: int,
    bias_id: int,
    stream_id: int = 0,
    prefetch_stream_id: int = 1,
    deps: list = None
) -> Tuple[int, int]:
    """Add linear forward ops to execution graph.

    Returns: (prefetch_node_id, compute_node_id)
    """
    # Prefetch weight and bias
    prefetch_id = graph.add_prefetch(
        tensor_ids=[weight_id, bias_id],
        stream_id=prefetch_stream_id,
        deps=deps
    )

    # Compute depends on prefetch
    compute_id = graph.add_compute(
        tensor_ids=[x_id, weight_id, bias_id],
        stream_id=stream_id,
        deps=[prefetch_id] + (deps or []),
        op="linear_forward"
    )

    return prefetch_id, compute_id


def schedule_linear_backward(
    graph: ExecutionGraph,
    grad_out_id: int,
    x_id: int,
    weight_id: int,
    stream_id: int = 0,
    deps: list = None
) -> int:
    """Add linear backward op to execution graph."""
    return graph.add_compute(
        tensor_ids=[grad_out_id, x_id, weight_id],
        stream_id=stream_id,
        deps=deps or [],
        op="linear_backward"
    )
