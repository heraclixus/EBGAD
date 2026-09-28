"""Synthetic graphs for controlled stress tests.

The stochastic block model used by the misspecification study: four equal
blocks, edge probability ``p_in`` inside a block and ``p_out`` between
blocks, undirected, no self loops. Randomness comes from torch's global
generator, so ``torch.manual_seed`` fixes the graph.
"""
from __future__ import annotations

import torch


def _stochastic_block_model(
    n: int,
    n_blocks: int = 4,
    p_in: float = 0.02,
    p_out: float = 0.002,
) -> torch.Tensor:
    """Return the (2, E) edge index (both directions) of an SBM on ``n`` nodes."""
    block = torch.arange(n) * n_blocks // n
    same = block[:, None] == block[None, :]
    prob = torch.where(same, torch.full((n, n), p_in), torch.full((n, n), p_out))
    upper = torch.triu(torch.rand(n, n) < prob, diagonal=1)
    src, dst = upper.nonzero(as_tuple=True)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
