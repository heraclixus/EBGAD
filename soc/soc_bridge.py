"""
SOC Bridge: Core Eigenspace Statistics
========================================
Stochastic optimal control bridge with graph-aware reference process,
tunable terminal penalty λ, geometry blending ρ, and feature templates.

All quantities are computed **per-eigenvalue** of Q_ρ for efficiency.
Since Q_ρ = ρ(κ²I + L)^ν + (1−ρ)I is a function of L, it shares L's
eigenvectors V — one eigendecomposition gives everything.

Key quantities (per eigenvalue q_i of Q_ρ, constant schedule α):

    Φ_{s:t}   = exp(−α(t−s) q_i)                propagator
    Σ_{s:t}   = q_i⁻¹(1 − exp(−2α(t−s) q_i))   covariance increment
    D_{0,λ}   = λ⁻¹ + Σ_{0:T}                   terminal operator (full horizon)
    b_t       = Σ_{0:t}·Φ_{t:T}·D_{0,λ}⁻¹                           coeff of x_T
    a_t       = Φ_{0:t} − b_t·Φ_{0:T}                                coeff of x₀
    c_t       = (1−Φ_{0:t}) − b_t·(1−Φ_{0:T})                       coeff of m
    Σ^SI_t    = Σ_{0:t}·Σ_{t:T}·Σ_{0:T}⁻¹                          SI covariance
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, degree


# ═══════════════════════════════════════════════════════════════════════════
# Local affinity (one-class homophily)
# ═══════════════════════════════════════════════════════════════════════════

def compute_local_affinity(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Per-node local affinity: avg cosine similarity to neighbors.

    Higher affinity → more likely to be a normal (homophily-consistent)
    node, following the one-class homophily observation of TAM.

    Parameters
    ----------
    x : (n, d) node features.
    edge_index : (2, E).
    num_nodes : n.

    Returns
    -------
    affinity : (n,)  in [-1, 1], higher = more normal.
    """
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = ei[0], ei[1]
    deg = degree(src, num_nodes=num_nodes).float().clamp(min=1)

    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=1)

    aff = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    aff.scatter_add_(0, src, cos_sim)
    return aff / deg


def affinity_to_weights(
    affinity: torch.Tensor,
) -> torch.Tensor:
    """Convert per-node affinity scores to reliability weights w_i ∈ [0, 1].

    Uses min-max normalization as in the paper::

        w_i = (a_i − min_k a_k) / (max_k a_k − min_k a_k)

    Nodes with highest local affinity get w=1 (most normal);
    nodes with lowest get w=0 (most suspicious).

    Parameters
    ----------
    affinity : (n,) local affinity scores.

    Returns
    -------
    weights : (n,) in [0, 1].
    """
    a_min = affinity.min()
    a_max = affinity.max()
    denom = (a_max - a_min).clamp(min=1e-8)
    return (affinity - a_min) / denom


def compute_affinity_edge_weights(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Per-edge affinity weights for the refined Laplacian.

    Returns cosine similarity per edge, clamped to [0, 1].
    Edges connecting similar nodes (homophily) get weight close to 1;
    edges connecting dissimilar nodes get weight close to 0.

    Parameters
    ----------
    x : (n, d) node features.
    edge_index : (2, E).
    num_nodes : n.

    Returns
    -------
    edge_weights : (E,) in [0, 1].
    """
    src, dst = edge_index[0], edge_index[1]
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=1)
    return cos_sim.clamp(min=0.0)


def compute_affinity_refined_laplacian(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    variant: str = "sym",
) -> torch.Tensor:
    """Compute L̃: Laplacian of the affinity-refined graph.

    The affinity-refined adjacency is defined as::

        Ã_{ij} = A_{ij} · w_i · w_j · max(0, sim(z_i, z_j))

    where w_i are min-max-normalised local-affinity scores (node trust)
    and sim is cosine similarity (edge trust).  L̃ = D̃ − Ã.

    Parameters
    ----------
    edge_index : (2, E).
    x : (n, d) node features.
    num_nodes : n.
    variant : "sym" or "comb".

    Returns
    -------
    L_tilde : (n, n) dense Laplacian of the affinity-refined graph.
    """
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = ei[0], ei[1]

    # Per-edge cosine similarity (clamped to [0, 1])
    edge_sim = compute_affinity_edge_weights(x, ei, num_nodes)

    # Per-node affinity weights via min-max normalization
    node_aff = compute_local_affinity(x, edge_index, num_nodes)
    w = affinity_to_weights(node_aff)

    # Ã_{ij} = A_{ij} * w_i * w_j * max(0, sim(z_i, z_j))
    A_tilde = torch.zeros(num_nodes, num_nodes, device=x.device)
    A_tilde[src, dst] = edge_sim * w[src] * w[dst]
    deg = A_tilde.sum(dim=1)

    if variant == "comb":
        return torch.diag(deg) - A_tilde

    I = torch.eye(num_nodes, device=x.device)
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return I - D_inv_sqrt @ A_tilde @ D_inv_sqrt


def compute_affinity_refined_laplacian_sparse(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    variant: str = "sym",
) -> torch.Tensor:
    """Compute L̃ as a **sparse** Laplacian of the affinity-refined graph.

    Same semantics as compute_affinity_refined_laplacian but never builds
    dense n×n. Returns sparse COO tensor. Enables affinity on large graphs
    (e.g. DGraph 3.7M nodes) when used with LOBPCG.

    Parameters
    ----------
    edge_index : (2, E).
    x : (n, d) node features.
    num_nodes : n.
    variant : "sym" or "comb".

    Returns
    -------
    L_tilde : (n, n) sparse COO Laplacian.
    """
    from torch_geometric.utils import to_undirected

    ei = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = ei[0], ei[1]

    # Per-edge cosine similarity (clamped to [0, 1])
    edge_sim = compute_affinity_edge_weights(x, ei, num_nodes)

    # Per-node affinity weights via min-max normalization
    node_aff = compute_local_affinity(x, edge_index, num_nodes)
    w = affinity_to_weights(node_aff)

    # Ã_{ij} = A_{ij} * w_i * w_j * max(0, sim(z_i, z_j))
    edge_weights = edge_sim * w[src] * w[dst]

    # Degree: deg_i = sum_j Ã_{ij}
    deg = torch.zeros(num_nodes, device=x.device, dtype=edge_weights.dtype)
    deg.scatter_add_(0, src, edge_weights)
    # to_undirected may have (i,j) and (j,i); coalesce or ensure we don't double-count
    # For to_undirected, each edge appears once per direction. So we need to sum over
    # all edges incident to i. scatter_add(edge_weights, src) sums over edges where
    # src=i. For undirected (i,j) we have both (i,j) and (j,i). So we get 2*a_ij for
    # each edge? No - to_undirected adds reverse edges, so we have (i,j) and (j,i)
    # as separate entries. scatter_add(edge_weights, src) on (i,j) adds a_ij to deg_i,
    # and on (j,i) adds a_ji to deg_j. So each edge contributes once to each endpoint.
    # deg_i = sum_{j in N(i)} a_ij. Good.

    if variant == "comb":
        # L_comb = D - A_tilde: off-diag -a_ij, diag deg_i
        diag_idx = torch.arange(num_nodes, device=ei.device)
        indices = torch.cat([
            torch.stack([src, dst]),
            torch.stack([diag_idx, diag_idx]),
        ], dim=1)
        values = torch.cat([-edge_weights, deg])
        L = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        return L.coalesce()

    if variant == "sym":
        # L_sym = I - D^{-1/2} A_tilde D^{-1/2}
        # Off-diagonal: -a_ij / sqrt(deg_i * deg_j)
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        vals_off = -edge_weights * d_inv_sqrt[src] * d_inv_sqrt[dst]

        # Diagonal: (D^{-1/2} A D^{-1/2})_{ii} = A_ii/sqrt(deg_i*deg_i) = 0 (no self-loops).
        # So L_ii = 1 - 0 = 1 for all nodes.
        diag_idx = torch.arange(num_nodes, device=ei.device)
        diag_vals = torch.ones(num_nodes, device=ei.device, dtype=edge_weights.dtype)

        indices = torch.cat([
            torch.stack([src, dst]),
            torch.stack([diag_idx, diag_idx]),
        ], dim=1)
        values = torch.cat([vals_off, diag_vals])
        L = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        return L.coalesce()

    raise ValueError(f"Unknown variant={variant!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Q_ρ construction
# ═══════════════════════════════════════════════════════════════════════════

def _canonicalize_two_scale_params(
    kappa: float,
    kappa2: Optional[float],
    eta: float,
) -> Tuple[float, float, float]:
    """Canonicalise the two-scale family to avoid label switching.

    The returned ``eta`` is always the weight on the smaller cutoff.
    """
    k1 = float(kappa)
    k2 = float(kappa if kappa2 is None else kappa2)
    mix = float(eta)
    if k2 < k1:
        return k2, k1, 1.0 - mix
    return k1, k2, mix


def compute_prior_base_eigenvalues(
    lam_L: torch.Tensor,
    kappa: float = 1.0,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> torch.Tensor:
    """Base spectral precision before rho-blending.

    Parameters
    ----------
    lam_L : (k,) Laplacian eigenvalues on the modeled subspace.
    kappa, nu : primary spectral cutoff and order.
    precision_family : ``"single_scale"`` (legacy) or ``"two_scale"``.
    eta : weight on the lower-cutoff component for ``"two_scale"``.
    kappa2 : optional second cutoff for ``"two_scale"``.
    """
    if precision_family in ("single_scale", "matern", "legacy"):
        return (kappa ** 2 + lam_L).pow(nu)

    if precision_family in ("two_scale", "two_band", "mixture"):
        k1, k2, mix = _canonicalize_two_scale_params(kappa, kappa2, eta)
        q1 = (k1 ** 2 + lam_L).pow(nu)
        q2 = (k2 ** 2 + lam_L).pow(nu)
        return mix * q1 + (1.0 - mix) * q2

    raise ValueError(f"Unknown precision_family: {precision_family!r}")


def compute_modeled_subspace_mask(
    lam_L: torch.Tensor,
    mode: str = "legacy",
    tol: float = 1e-8,
) -> torch.Tensor:
    """Boolean mask for the modeled eigenspace.

    ``"legacy"`` keeps all computed modes.
    ``"drop_nullspace"`` excludes near-zero Laplacian modes.
    """
    if mode in ("legacy", "full", "all"):
        return torch.ones_like(lam_L, dtype=torch.bool)

    if mode in ("drop_nullspace", "exclude_nullspace", "project_nullspace"):
        mask = lam_L > tol
        if not torch.any(mask):
            return torch.ones_like(lam_L, dtype=torch.bool)
        return mask

    raise ValueError(f"Unknown modeled_subspace mode: {mode!r}")


def apply_modeled_subspace(
    lam_L: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    mode: str = "legacy",
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Restrict eigenspace objects to the modeled subspace."""
    mask = compute_modeled_subspace_mask(lam_L, mode=mode, tol=tol)
    lam_model = lam_L[mask]
    if V is None:
        return lam_model, None, mask
    return lam_model, V[:, mask], mask


def compute_Q_rho_eigenvalues(
    lam_L: torch.Tensor,
    rho: float = 1.0,
    kappa: float = 1.0,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
) -> torch.Tensor:
    """Eigenvalues of Q_ρ under a label-free spectral precision family.

    Legacy behavior is recovered with ``precision_family="single_scale"``.
    """
    q_base = compute_prior_base_eigenvalues(
        lam_L,
        kappa=kappa,
        nu=nu,
        precision_family=precision_family,
        eta=eta,
        kappa2=kappa2,
    )
    return rho * q_base + (1.0 - rho)


# ═══════════════════════════════════════════════════════════════════════════
# LearnableRho — jointly optimized geometry blending
# ═══════════════════════════════════════════════════════════════════════════

class LearnableRho(nn.Module):
    """Learnable rho parameter for Q_rho = rho * Q_matern + (1-rho) * I.

    Three modes:

    - ``"learned_scalar"``: one shared rho for all eigenvalues.
    - ``"learned_diag"``:   per-eigenvalue rho_i, allowing different
      graph-vs-isotropic blending per spectral mode.
    - ``"learned_mask"``:   same as diagonal but with temperature-annealed
      sigmoid that encourages binary (on/off) rho values.

    The raw parameter is stored as a logit and passed through sigmoid so
    rho stays in [0, 1].

    Parameters
    ----------
    k : number of eigenvalues (ignored for scalar mode).
    mode : ``"learned_scalar"``, ``"learned_diag"``, or ``"learned_mask"``.
    init_rho : initialization value in [0, 1].
    tau_start : initial temperature for mask annealing (mask mode only).
    tau_end : final temperature (mask mode only).
    """

    def __init__(
        self,
        k: int,
        mode: str = "learned_scalar",
        init_rho: float = 0.5,
        tau_start: float = 5.0,
        tau_end: float = 0.5,
    ):
        super().__init__()
        self.mode = mode
        self.tau_start = tau_start
        self.tau_end = tau_end

        init_logit = math.log(max(init_rho, 1e-6) / max(1.0 - init_rho, 1e-6))

        if mode == "learned_scalar":
            self.logit = nn.Parameter(torch.tensor(init_logit))
        elif mode in ("learned_diag", "learned_mask"):
            self.logit = nn.Parameter(torch.full((k,), init_logit))
        else:
            raise ValueError(f"Unknown LearnableRho mode: {mode!r}")

    def forward(
        self,
        lam_Q_base: torch.Tensor,
        progress: float = 1.0,
    ) -> torch.Tensor:
        """Compute Q_rho eigenvalues with current learned rho.

        Parameters
        ----------
        lam_Q_base : (k,) Matern eigenvalues (kappa^2 + lam_L)^nu.
        progress : training progress in [0, 1] (for mask tau annealing).

        Returns
        -------
        lam_Q_rho : (k,) eigenvalues of rho * Q_matern + (1-rho) * I.
        """
        if self.mode == "learned_mask":
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress
            rho = torch.sigmoid(self.logit / max(tau, 1e-6))
        else:
            rho = torch.sigmoid(self.logit)

        if rho.dim() == 0:
            rho = rho.expand_as(lam_Q_base)

        return rho * lam_Q_base + (1.0 - rho)

    @torch.no_grad()
    def get_rho(self, progress: float = 1.0) -> torch.Tensor:
        """Return current rho values (detached) for inspection."""
        if self.mode == "learned_mask":
            tau = self.tau_start + (self.tau_end - self.tau_start) * progress
            return torch.sigmoid(self.logit / max(tau, 1e-6))
        return torch.sigmoid(self.logit)


# ═══════════════════════════════════════════════════════════════════════════
# Template computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_smoother_eigenvalues(
    lam_L: torch.Tensor,
    gamma: float = 1.0,
    nu: float = 1.0,
) -> torch.Tensor:
    """Eigenvalues of the Matérn smoother S_{γ,ν} = (γ²I + L)^{−ν}.

    Returns
    -------
    (n,) eigenvalues of S.
    """
    return (gamma ** 2 + lam_L).pow(-nu)


def compute_template(
    x: torch.Tensor,
    V: torch.Tensor,
    lam_L: torch.Tensor,
    template_type: str = "low",
    gamma: float = 1.0,
    nu: float = 1.0,
    pi: float = 0.5,
    node_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the graph-based template m.

    Non-affinity templates ("low", "high", "mix") operate on raw features.
    Affinity templates ("affinity", "affinity_high", "affinity_mix") first
    apply node reliability weights W before the smoother, following the
    paper's formulation::

        m_low  = S_{γ,ν} W X          (low-pass, affinity-weighted)
        m_high = (I − S_{γ,ν}) W X    (high-pass, affinity-weighted)
        m_mix  = π m_low + (1−π) m_high

    V and lam_L should be from the affinity-refined Laplacian L̃ when
    using affinity templates, so that S_{γ,ν} = (γ²I + L̃)^{-ν}.

    Parameters
    ----------
    x : (n, d) node features.
    V : (n, n) or (n, k) eigenvectors of L (or L̃).
    lam_L : (n,) or (k,) eigenvalues of L (or L̃).
    template_type : "zero", "low", "high", "mix",
        "affinity" (= affinity low), "affinity_high", "affinity_mix".
    gamma : smoother scale.
    nu : smoother order.
    pi : mixing weight for "mix"/"affinity_mix".
    node_weights : (n,) reliability weights W for affinity templates.

    Returns
    -------
    m : (n, d) template.
    """
    if template_type == "zero":
        return torch.zeros_like(x)

    if template_type == "normal_mean":
        return x.mean(dim=0, keepdim=True).expand_as(x)

    s_eig = compute_smoother_eigenvalues(lam_L, gamma, nu)

    # ── Affinity-weighted templates: apply W before smoother ──
    if template_type in ("affinity", "affinity_high", "affinity_mix"):
        x_w = node_weights.unsqueeze(-1) * x if node_weights is not None else x
        x_hat = V.T @ x_w
        x_low = V @ (s_eig.unsqueeze(-1) * x_hat)
        x_high = x_w - x_low

        if template_type == "affinity":
            return x_low
        elif template_type == "affinity_high":
            return x_high
        elif template_type == "affinity_mix":
            return pi * x_low + (1.0 - pi) * x_high

    # ── Standard (non-weighted) templates ──
    x_hat = V.T @ x
    x_low = V @ (s_eig.unsqueeze(-1) * x_hat)

    if template_type == "low":
        return x_low
    elif template_type == "high":
        return x - x_low
    elif template_type == "mix":
        x_high = x - x_low
        return pi * x_low + (1.0 - pi) * x_high
    else:
        raise ValueError(f"Unknown template_type: {template_type!r}")


# ═══════════════════════════════════════════════════════════════════════════
# SOC bridge statistics (eigenspace)
# ═══════════════════════════════════════════════════════════════════════════

class SOCBridgeStats(NamedTuple):
    """Pre-computed SOC bridge quantities in eigenspace.

    All (k,) tensors are per-eigenvalue of Q_ρ.
    """
    V: torch.Tensor           # (n, n) or (n, k)
    a_t: torch.Tensor         # (k,) coefficient of x₀
    b_t: torch.Tensor         # (k,) coefficient of x_T
    c_t: torch.Tensor         # (k,) coefficient of m
    sigma_prime_t: torch.Tensor    # (k,) Σ'_t eigenvalues
    sigma_prime_t_sqrt: torch.Tensor  # (k,) √Σ'_t
    sigma_prime_t_inv: torch.Tensor   # (k,) 1/Σ'_t


def soc_bridge_statistics(
    lam_Q: torch.Tensor,
    V: torch.Tensor,
    alpha: float,
    t: float,
    T: float,
    lam_penalty: float = 10.0,
    jitter: float = 1e-6,
) -> SOCBridgeStats:
    """Pre-compute SOC bridge quantities in eigenspace.

    Implements the stochastic interpolant form from Theorem 1:

        b_t = Σ_{0:t} Φ_{t:T} D_{0,λ}⁻¹
        a_t = Φ_{0:t} − b_t Φ_{0:T}
        c_t = (I − Φ_{0:t}) − b_t (I − Φ_{0:T})
        Σ^SI_t = Σ_{0:t} Σ_{t:T} Σ_{0:T}⁻¹

    where D_{0,λ} = λ⁻¹I + Σ_{0:T} uses the full-horizon covariance.

    Parameters
    ----------
    lam_Q : (k,) eigenvalues of Q_ρ.
    V : (n, k) eigenvectors.
    alpha : diffusion rate (constant schedule).
    t : current time in (0, T).
    T : terminal time.
    lam_penalty : terminal penalty λ (larger = harder endpoint matching).
    jitter : floor for numerical stability.

    Returns
    -------
    SOCBridgeStats named-tuple.
    """
    lam_safe = lam_Q.clamp(min=jitter)

    # Propagator eigenvalues
    A_0t = torch.exp(-alpha * t * lam_Q)
    A_tT = torch.exp(-alpha * (T - t) * lam_Q)
    A_0T = torch.exp(-alpha * T * lam_Q)

    # Covariance increment eigenvalues
    Sig_0t = (1.0 - torch.exp(-2.0 * alpha * t * lam_safe)) / lam_safe
    Sig_tT = (1.0 - torch.exp(-2.0 * alpha * (T - t) * lam_safe)) / lam_safe
    Sig_0T = (1.0 - torch.exp(-2.0 * alpha * T * lam_safe)) / lam_safe

    # Terminal operator D_{0,λ} = 1/λ + Σ_{0:T}  (full-horizon, see Theorem 1)
    D_0l = 1.0 / lam_penalty + Sig_0T
    D_0l_inv = 1.0 / D_0l.clamp(min=jitter)

    # Bridge coefficients (eqs. b_t_def, a_t_def, c_t_def in the paper)
    b_t = Sig_0t * A_tT * D_0l_inv
    a_t = A_0t - b_t * A_0T
    c_t = (1.0 - A_0t) - b_t * (1.0 - A_0T)

    # Bridge covariance (stochastic interpolant): Σ' = Σ_{0:t} Σ_{t:T} / Σ_{0:T}
    sigma_prime = (Sig_0t * Sig_tT) / Sig_0T.clamp(min=jitter)
    sigma_prime_clamped = sigma_prime.clamp(min=jitter)

    return SOCBridgeStats(
        V=V,
        a_t=a_t,
        b_t=b_t,
        c_t=c_t,
        sigma_prime_t=sigma_prime,
        sigma_prime_t_sqrt=sigma_prime_clamped.sqrt(),
        sigma_prime_t_inv=1.0 / sigma_prime_clamped,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Bridge mean / sample / score target
# ═══════════════════════════════════════════════════════════════════════════

def _eig_apply(V: torch.Tensor, eig: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """V diag(eig) Vᵀ x."""
    return V @ (eig.unsqueeze(-1) * (V.T @ x))


def soc_bridge_mean(
    stats: SOCBridgeStats,
    x_0: torch.Tensor,
    x_T: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """SOC bridge conditional mean  μ_{t,λ}(x₀, x_T).

    μ = V (a_t x̂₀ + b_t x̂_T + c_t m̂)

    Parameters
    ----------
    stats : pre-computed SOCBridgeStats.
    x_0 : (n, d) prior sample.
    x_T : (n, d) data sample.
    m : (n, d) template.

    Returns
    -------
    mu_t : (n, d)
    """
    V = stats.V
    x0_hat = V.T @ x_0
    xT_hat = V.T @ x_T
    m_hat = V.T @ m

    mu_hat = (stats.a_t.unsqueeze(-1) * x0_hat
              + stats.b_t.unsqueeze(-1) * xT_hat
              + stats.c_t.unsqueeze(-1) * m_hat)
    return V @ mu_hat


def soc_bridge_sample(
    stats: SOCBridgeStats,
    x_0: torch.Tensor,
    x_T: torch.Tensor,
    m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t ~ N(μ_{t,λ}, Σ'_{t,λ}) via reparameterization.

    Returns
    -------
    x_t : (n, d)
    mu_t : (n, d)
    """
    mu_t = soc_bridge_mean(stats, x_0, x_T, m)
    eps = torch.randn_like(mu_t)
    noise = _eig_apply(stats.V, stats.sigma_prime_t_sqrt, eps)
    return mu_t + noise, mu_t


def soc_score_target(
    stats: SOCBridgeStats,
    x_t: torch.Tensor,
    mu_t: torch.Tensor,
) -> torch.Tensor:
    """Analytic score target  −(Σ'_t)^{−1} (x_t − μ_t).

    Parameters
    ----------
    stats : pre-computed SOCBridgeStats.
    x_t : (n, d) intermediate state.
    mu_t : (n, d) bridge mean.

    Returns
    -------
    target : (n, d)
    """
    residual = x_t - mu_t
    return -_eig_apply(stats.V, stats.sigma_prime_t_inv, residual)


def soc_epsilon_target(
    stats: SOCBridgeStats,
    x_t: torch.Tensor,
    mu_t: torch.Tensor,
) -> torch.Tensor:
    """ε-prediction target:  (Σ'_t)^{−1/2} (x_t − μ_t).

    The network predicts ε such that x_t = μ_t + (Σ'_t)^{1/2} ε.

    Returns
    -------
    epsilon : (n, d)
    """
    residual = x_t - mu_t
    inv_sqrt = 1.0 / stats.sigma_prime_t_sqrt
    return _eig_apply(stats.V, inv_sqrt, residual)
