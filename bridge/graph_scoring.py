"""
Graph-Aware Anomaly Score Post-Processing
==========================================
Tier 1 scoring improvements that work on top of any per-node anomaly
score or per-node output vector.  No retraining required.

1. **Local-roughness augmentation**: augments any base score with a
   graph-local consistency penalty.
2. **Spectral-band scoring**: decomposes a residual signal into graph
   frequency bands and scores per-band energy.

These are shared utilities used by both ``bridge.anomaly_score`` and
``flow.flow_anomaly``.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import to_undirected, degree


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Local-roughness computation
# ═══════════════════════════════════════════════════════════════════════════

def local_roughness(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Per-node local roughness:  avg_{j∈N(i)} ‖x[i] − x[j]‖².

    Parameters
    ----------
    x : (n, d)  node signal.
    edge_index : (2, E)  edge list.
    num_nodes : n.

    Returns
    -------
    roughness : (n,)  non-negative.
    """
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = ei
    deg = degree(src, num_nodes=num_nodes).float().clamp(min=1)
    edge_sq_diff = ((x[src] - x[dst]) ** 2).sum(dim=-1)
    rough = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    rough.scatter_add_(0, src, edge_sq_diff)
    return rough / deg


def hybrid_score(
    base_scores: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    lam: float = 1.0,
) -> torch.Tensor:
    """Augment base anomaly scores with local-roughness penalty.

    ``S_i = base_i + λ * roughness_i``

    Both components are normalised to zero mean / unit variance before
    combining so that λ controls relative weight meaningfully.

    Parameters
    ----------
    base_scores : (n,)  any per-node anomaly scores.
    x : (n, d)  the signal to measure roughness on (e.g. model
        reconstruction, velocity output, or raw features).
    edge_index : (2, E).
    num_nodes : n.
    lam : relative weight of the roughness term.

    Returns
    -------
    combined : (n,)
    """
    rough = local_roughness(x, edge_index, num_nodes)

    def _standardise(v: torch.Tensor) -> torch.Tensor:
        m = v.mean()
        s = v.std().clamp(min=1e-8)
        return (v - m) / s

    return _standardise(base_scores) + lam * _standardise(rough)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Spectral-band scoring
# ═══════════════════════════════════════════════════════════════════════════

def spectral_band_score(
    residual: torch.Tensor,
    graph_eigen,
    band: str = "high",
    tau: float = 1.0,
) -> torch.Tensor:
    """Per-node anomaly score from a specific graph frequency band.

    Projects ``residual`` into the low or high frequency band via the
    eigenspace and returns per-node squared energy in that band.

    Parameters
    ----------
    residual : (n, d)  e.g. ``x̂₁ − x₁`` or ``s_θ(t, x_t)``.
    graph_eigen : ``GraphEigen`` instance with ``.apply()`` and
        eigenvalue methods.
    band : ``"high"`` or ``"low"``.
    tau : heat-kernel scale for the band split.

    Returns
    -------
    scores : (n,)  per-node energy in the selected band.
    """
    if band == "high":
        eig = graph_eigen.K_high_eigenvalues(tau)
    elif band == "low":
        eig = graph_eigen.K_low_eigenvalues(tau)
    else:
        raise ValueError(f"Unknown band: {band!r}")

    projected = graph_eigen.apply(eig, residual)
    return (projected ** 2).sum(dim=-1)


def edge_contrast_score(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Per-node anomaly score from edge-level output contrast.

    S_i = avg_{j∈N(i)} ‖x[i] − x[j]‖²

    This is local-roughness applied to an arbitrary signal (e.g. the
    model's velocity or score output rather than raw features).

    Parameters
    ----------
    x : (n, d)  any per-node signal (velocity output, score output, etc.).
    edge_index : (2, E).
    num_nodes : n.

    Returns
    -------
    scores : (n,)
    """
    return local_roughness(x, edge_index, num_nodes)


def spectral_ratio_score(
    residual: torch.Tensor,
    graph_eigen,
    tau: float = 1.0,
) -> torch.Tensor:
    """Per-node high-to-total energy ratio in graph frequency domain.

    Anomalous nodes (with high-frequency perturbations) will have a
    higher fraction of their residual energy in the high band.

    ``S_i = ‖K_high residual[i]‖² / (‖residual[i]‖² + ε)``

    Works with ``GraphEigen``, ``GraphEigenTruncated``, or
    ``ChebyshevFilter`` (auto-detected via duck typing).

    Parameters
    ----------
    residual : (n, d).
    graph_eigen : ``GraphEigen``, ``GraphEigenTruncated``, or
        ``ChebyshevFilter`` instance.
    tau : heat-kernel scale.

    Returns
    -------
    scores : (n,)  in [0, 1].
    """
    # ChebyshevFilter has its own spectral_ratio_score method
    if hasattr(graph_eigen, "spectral_ratio_score"):
        return graph_eigen.spectral_ratio_score(residual, tau=tau)

    high_energy = spectral_band_score(residual, graph_eigen, band="high", tau=tau)
    total_energy = (residual ** 2).sum(dim=-1).clamp(min=1e-8)
    return high_energy / total_energy
