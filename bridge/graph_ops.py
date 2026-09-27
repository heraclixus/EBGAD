"""
Phase 0: Graph Operators
========================
Laplacians, Matérn precision matrices, matrix-exponential–vector products,
graph-structured prior sampling, and heat-kernel graph filters.

Three API levels:

* **Legacy (dense)**: standalone functions that build full n×n matrices.
  Simple and used by tests.  Fine for n < ~2K.
* **GraphEigen (full eigenspace)**: one O(n³) eigendecomposition of L, then
  all subsequent operations are O(n²·d).  Use this for n < ~10K.
* **GraphEigenTruncated (partial eigenspace)**: O(n·k²) partial eigen via
  LOBPCG; per-step cost O(n·k·d) with k << n.  For n up to ~50K.
* **ChebyshevFilter (no eigendecomposition)**: Chebyshev polynomial
  approximation using only sparse matmuls.  For n up to ~500K.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch_geometric.utils import to_dense_adj, to_undirected


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Laplacian construction
# ═══════════════════════════════════════════════════════════════════════════

def compute_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    variant: str = "sym",
) -> torch.Tensor:
    """Compute a dense graph Laplacian from a PyG ``edge_index``.

    Parameters
    ----------
    edge_index : (2, E) long tensor
        Edge list (undirected – both directions expected).
    num_nodes : int
        Number of nodes *n*.
    variant : ``"comb"`` | ``"sym"`` | ``"rw"``
        * ``"comb"``  – combinatorial  L = D − A
        * ``"sym"``   – symmetric normalised  L = I − D^{-1/2} A D^{-1/2}
        * ``"rw"``    – random-walk  L = I − D^{-1} A

    Returns
    -------
    L : (n, n) float tensor
    """
    # Symmetrise directed graphs so the Laplacian is always valid PSD
    edge_index_sym = to_undirected(edge_index, num_nodes=num_nodes)
    A = to_dense_adj(edge_index_sym, max_num_nodes=num_nodes)[0]
    deg = A.sum(dim=1)

    if variant == "comb":
        return torch.diag(deg) - A

    I = torch.eye(num_nodes, device=A.device, dtype=A.dtype)

    if variant == "sym":
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        return I - D_inv_sqrt @ A @ D_inv_sqrt

    if variant == "rw":
        d_inv = torch.where(deg > 0, deg.pow(-1.0), torch.zeros_like(deg))
        D_inv = torch.diag(d_inv)
        return I - D_inv @ A

    raise ValueError(f"Unknown Laplacian variant: {variant!r}")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Shifted Laplacian precision   Q = (κ²I + L)^ν
# ═══════════════════════════════════════════════════════════════════════════

def shifted_laplacian_precision(
    L: torch.Tensor,
    kappa: float = 1.0,
    nu: float = 1.0,
) -> torch.Tensor:
    """Graph Matérn precision  Q = (κ²I + L)^ν  (dense).

    For ν = 1 this is simply κ²I + L.
    For general ν we use eigendecomposition of the SPD base matrix.
    """
    n = L.shape[0]
    base = kappa ** 2 * torch.eye(n, device=L.device, dtype=L.dtype) + L
    if nu == 1.0:
        return base
    eigvals, eigvecs = torch.linalg.eigh(base)
    return eigvecs @ torch.diag(eigvals.pow(nu)) @ eigvecs.T


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Matrix exponential  &  expm-multiply
# ═══════════════════════════════════════════════════════════════════════════

def matrix_exp(M: torch.Tensor) -> torch.Tensor:
    """Dense matrix exponential  exp(M).

    Wrapper around ``torch.linalg.matrix_exp`` for a single call-site that
    can later be swapped for Krylov / Chebyshev approximations.
    """
    return torch.linalg.matrix_exp(M)


def expm_multiply(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute  exp(M) @ v  (dense).

    Parameters
    ----------
    M : (n, n)
    v : (n,) or (n, d)

    Returns
    -------
    Same shape as *v*.
    """
    return matrix_exp(M) @ v


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Prior sampling   X₀ ~ N(0, Q⁻¹)
# ═══════════════════════════════════════════════════════════════════════════

def compute_Q_inv_sqrt(Q: torch.Tensor) -> torch.Tensor:
    """Compute  Q^{-1/2}  via eigendecomposition (Q must be SPD)."""
    eigvals, eigvecs = torch.linalg.eigh(Q)
    return eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T


def sample_matern_prior(
    Q: torch.Tensor,
    d: int,
    num_samples: int = 1,
    Q_inv_sqrt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample  X₀ ~ N(0, Q⁻¹)  column-wise  (Matérn prior).

    X₀ = Q^{-1/2} Ξ   where  Ξ ~ N(0, I_{n×d}).

    Parameters
    ----------
    Q : (n, n) SPD precision matrix.
    d : feature dimension.
    num_samples : batch size (1 → squeeze the batch dim).
    Q_inv_sqrt : optional pre-computed Q^{-1/2}.

    Returns
    -------
    (n, d) if ``num_samples == 1``, else (num_samples, n, d).
    """
    if Q_inv_sqrt is None:
        Q_inv_sqrt = compute_Q_inv_sqrt(Q)
    n = Q.shape[0]
    if num_samples == 1:
        xi = torch.randn(n, d, device=Q.device, dtype=Q.dtype)
        return Q_inv_sqrt @ xi
    xi = torch.randn(num_samples, n, d, device=Q.device, dtype=Q.dtype)
    return torch.einsum("ij, bjk -> bik", Q_inv_sqrt, xi)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Heat-kernel graph filters
# ═══════════════════════════════════════════════════════════════════════════

def heat_kernel_filters(
    L: torch.Tensor,
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Low-pass and high-pass heat-kernel filters.

    K_low  = exp(−τ L)
    K_high = I − exp(−τ L)

    Returns
    -------
    (K_low, K_high), each (n, n).
    """
    K_low = matrix_exp(-tau * L)
    n = L.shape[0]
    K_high = torch.eye(n, device=L.device, dtype=L.dtype) - K_low
    return K_low, K_high


def apply_graph_filters(
    L: torch.Tensor,
    x: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Graph-filter feature augmentation.

    φ(x) = [ x,  K_low x,  K_high x ]  ∈  R^{n × 3d}

    Parameters
    ----------
    L : (n, n) Laplacian.
    x : (n, d) node features.
    tau : heat-kernel scale.

    Returns
    -------
    (n, 3d) augmented features.
    """
    K_low, K_high = heat_kernel_filters(L, tau)
    return torch.cat([x, K_low @ x, K_high @ x], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Two-band prior
# ═══════════════════════════════════════════════════════════════════════════

def two_band_covariance(
    L: torch.Tensor,
    tau: float = 1.0,
    sigma_low: float = 1.0,
    sigma_high: float = 1.0,
) -> torch.Tensor:
    """Analytic two-band covariance matrix.

    Σ = σ²_low  K_low K_low^T  +  σ²_high  K_high K_high^T

    Returns
    -------
    (n, n) PSD covariance.
    """
    K_low, K_high = heat_kernel_filters(L, tau)
    return (sigma_low ** 2) * (K_low @ K_low.T) + (sigma_high ** 2) * (K_high @ K_high.T)


def sample_two_band_prior(
    L: torch.Tensor,
    d: int,
    tau: float = 1.0,
    sigma_low: float = 1.0,
    sigma_high: float = 1.0,
    num_samples: int = 1,
) -> torch.Tensor:
    """Sample from the two-band prior.

    X₀ = σ_low  K_low Ξ  +  σ_high  K_high Ξ'
    where Ξ, Ξ' ~ N(0, I_{n×d}) independently.

    Returns
    -------
    (n, d) if ``num_samples == 1``, else (num_samples, n, d).
    """
    n = L.shape[0]
    K_low, K_high = heat_kernel_filters(L, tau)
    if num_samples == 1:
        xi = torch.randn(n, d, device=L.device, dtype=L.dtype)
        xi_prime = torch.randn(n, d, device=L.device, dtype=L.dtype)
        return sigma_low * (K_low @ xi) + sigma_high * (K_high @ xi_prime)
    xi = torch.randn(num_samples, n, d, device=L.device, dtype=L.dtype)
    xi_prime = torch.randn(num_samples, n, d, device=L.device, dtype=L.dtype)
    low = sigma_low * torch.einsum("ij, bjk -> bik", K_low, xi)
    high = sigma_high * torch.einsum("ij, bjk -> bik", K_high, xi_prime)
    return low + high


# ═══════════════════════════════════════════════════════════════════════════
# 7.  GraphEigen — eigenspace precompute cache
# ═══════════════════════════════════════════════════════════════════════════

class GraphEigen:
    """One-time eigendecomposition of L; derives every other operator in O(n·d).

    All matrices used by the bridge pipeline (Q, K_low, K_high, Q^{-1/2}, …)
    are functions of L and therefore share L's eigenvectors.  This class
    computes the eigenvectors **once** and then provides fast O(n·d)
    ``apply`` methods instead of O(n²·d) dense matrix–vector products.

    Parameters
    ----------
    edge_index : (2, E) PyG edge list.
    num_nodes : n.
    variant : Laplacian type (``"sym"``, ``"comb"``, ``"rw"``).
    """

    # Threshold above which we build L and run eigh on CPU to avoid:
    # - GPU OOM / segfault from to_dense_adj (39K² ≈ 6GB)
    # - cusolver errors on large eigendecomposition
    _CPU_EIGEN_THRESHOLD = 20_000

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        variant: str = "sym",
    ):
        target_device = edge_index.device
        use_cpu_setup = num_nodes >= self._CPU_EIGEN_THRESHOLD and edge_index.is_cuda

        if use_cpu_setup:
            # Build L on CPU to avoid GPU segfault from to_dense_adj on large graphs
            ei_cpu = edge_index.cpu()
            L = compute_laplacian(ei_cpu, num_nodes, variant=variant)
            lam, V = torch.linalg.eigh(L)
            self.lam_L = lam.to(target_device)
            self.V = V.to(target_device)
            self.L = L  # keep on CPU to save GPU memory
            self.device = target_device
        else:
            L = compute_laplacian(edge_index, num_nodes, variant=variant)
            if num_nodes >= self._CPU_EIGEN_THRESHOLD and L.is_cuda:
                L_cpu = L.cpu()
                lam, V = torch.linalg.eigh(L_cpu)
                self.lam_L = lam.to(L.device)
                self.V = V.to(L.device)
            else:
                self.lam_L, self.V = torch.linalg.eigh(L)
            self.L = L
            self.device = L.device

        self.n = num_nodes
        self.dtype = L.dtype

    @classmethod
    def from_eigenpairs(
        cls,
        lam_L: torch.Tensor,
        V: torch.Tensor,
        L: Optional[torch.Tensor] = None,
    ) -> "GraphEigen":
        """Build from precomputed eigenpairs (e.g. affinity-refined Laplacian).

        Use when L is computed externally (dense eigh of affinity L̃).
        """
        obj = object.__new__(cls)
        obj.lam_L = lam_L
        obj.V = V
        obj.n = V.shape[0]
        obj.device = V.device
        obj.dtype = V.dtype
        obj.L = L
        return obj

    # ------------------------------------------------------------------ #
    #  Core apply:  V diag(f) V^T @ x   in  O(n·d)
    # ------------------------------------------------------------------ #

    def apply(self, f_lam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply  V diag(f_lam) Vᵀ  to  x  without forming the n×n matrix.

        Parameters
        ----------
        f_lam : (n,)  per-eigenvalue scalars.
        x     : (n, d)  or  (n,).

        Returns
        -------
        Same shape as *x*.
        """
        return self.V @ (f_lam.unsqueeze(-1) * (self.V.T @ x))

    # ------------------------------------------------------------------ #
    #  Derived eigenvalue vectors
    # ------------------------------------------------------------------ #

    def Q_eigenvalues(self, kappa: float = 1.0, nu: float = 1.0) -> torch.Tensor:
        """Eigenvalues of  Q = (κ²I + L)^ν."""
        return (kappa ** 2 + self.lam_L).pow(nu)

    def Q_inv_sqrt_eigenvalues(self, kappa: float = 1.0,
                               nu: float = 1.0) -> torch.Tensor:
        """Eigenvalues of  Q^{-1/2}."""
        return (kappa ** 2 + self.lam_L).pow(-nu / 2.0)

    def K_low_eigenvalues(self, tau: float = 1.0) -> torch.Tensor:
        """Eigenvalues of  exp(−τ L)."""
        return torch.exp(-tau * self.lam_L)

    def K_high_eigenvalues(self, tau: float = 1.0) -> torch.Tensor:
        """Eigenvalues of  I − exp(−τ L)."""
        return 1.0 - torch.exp(-tau * self.lam_L)

    # ------------------------------------------------------------------ #
    #  High-level helpers used by the trainer / score network
    # ------------------------------------------------------------------ #

    def filter_features(self, x: torch.Tensor,
                        tau: float = 1.0) -> torch.Tensor:
        """Graph-filter augmentation  [x, K_low x, K_high x]  in O(n·d).

        Returns
        -------
        (n, 3d)
        """
        kl = self.K_low_eigenvalues(tau)
        kh = self.K_high_eigenvalues(tau)
        return torch.cat([x, self.apply(kl, x), self.apply(kh, x)], dim=-1)

    def sample_matern_prior(self, d: int, kappa: float = 1.0,
                            nu: float = 1.0) -> torch.Tensor:
        """Sample  X₀ ~ N(0, Q⁻¹)  in O(n·d).

        Returns
        -------
        (n, d)
        """
        q_inv_sqrt_eig = self.Q_inv_sqrt_eigenvalues(kappa, nu)
        xi = torch.randn(self.n, d, device=self.device, dtype=self.dtype)
        return self.apply(q_inv_sqrt_eig, xi)

    def sample_two_band_prior(
        self, d: int, tau: float = 1.0,
        sigma_low: float = 1.0, sigma_high: float = 1.0,
    ) -> torch.Tensor:
        """Sample from two-band prior in O(n·d).

        Returns
        -------
        (n, d)
        """
        kl = self.K_low_eigenvalues(tau)
        kh = self.K_high_eigenvalues(tau)
        xi = torch.randn(self.n, d, device=self.device, dtype=self.dtype)
        xi2 = torch.randn(self.n, d, device=self.device, dtype=self.dtype)
        return sigma_low * self.apply(kl, xi) + sigma_high * self.apply(kh, xi2)

    def bridge_eigen_cache(self, kappa: float = 1.0,
                           nu: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Q_eigenvalues, V)`` for ``bridge_statistics(eigen_cache=…)``.

        This avoids a separate ``eigh(Q)`` call.
        """
        return self.Q_eigenvalues(kappa, nu), self.V


# ═══════════════════════════════════════════════════════════════════════════
# 8.  Sparse Laplacian construction
# ═══════════════════════════════════════════════════════════════════════════

def compute_sparse_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    variant: str = "sym",
) -> torch.Tensor:
    """Compute a **sparse** graph Laplacian from a PyG ``edge_index``.

    Returns a ``torch.sparse_coo_tensor``.  Supports ``"sym"`` and ``"comb"``
    variants.  Used by :class:`GraphEigenTruncated` and
    :class:`ChebyshevFilter` to avoid ever materialising an n×n dense matrix.

    Parameters
    ----------
    edge_index : (2, E) long tensor.
    num_nodes : n.
    variant : ``"comb"`` or ``"sym"``.

    Returns
    -------
    L_sparse : (n, n) sparse COO tensor.
    """
    from torch_geometric.utils import degree

    ei = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = ei[0], ei[1]
    deg = degree(src, num_nodes=num_nodes).float()

    if variant == "comb":
        # Off-diagonal: -1 for each edge
        vals_off = -torch.ones(src.shape[0], device=ei.device, dtype=torch.float)
        # Diagonal: degree
        diag_idx = torch.arange(num_nodes, device=ei.device)
        indices = torch.cat([
            torch.stack([src, dst]),
            torch.stack([diag_idx, diag_idx]),
        ], dim=1)
        values = torch.cat([vals_off, deg])
        L = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        return L.coalesce()

    if variant == "sym":
        # L_sym = I - D^{-1/2} A D^{-1/2}
        # Off-diagonal entries: -1/sqrt(d_i * d_j) for each edge (i,j)
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5),
                                  torch.zeros_like(deg))
        vals_off = -(d_inv_sqrt[src] * d_inv_sqrt[dst])
        # Diagonal: 1 (for nodes with degree > 0)
        diag_idx = torch.arange(num_nodes, device=ei.device)
        diag_vals = torch.where(deg > 0, torch.ones_like(deg),
                                 torch.zeros_like(deg))
        indices = torch.cat([
            torch.stack([src, dst]),
            torch.stack([diag_idx, diag_idx]),
        ], dim=1)
        values = torch.cat([vals_off, diag_vals])
        L = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        return L.coalesce()

    raise ValueError(f"Sparse Laplacian not implemented for variant={variant!r}")


# ═══════════════════════════════════════════════════════════════════════════
# 9.  GraphEigenTruncated — partial eigendecomposition via LOBPCG
# ═══════════════════════════════════════════════════════════════════════════

class GraphEigenTruncated:
    """Truncated eigendecomposition of L using LOBPCG.

    Stores only the k smallest eigenpairs (V_k ∈ R^{n×k}, λ_k ∈ R^k).
    All operations become O(n·k·d) instead of O(n²·d).

    The ``apply`` method computes the **rank-k approximation**:
    ``V_k diag(f(λ_k)) V_k^T @ x``.  For heat-kernel filters this is
    accurate because exp(−τ λ) decays exponentially — high eigenvalues
    contribute negligibly.

    Provides the **same API** as :class:`GraphEigen` so it can be used
    as a drop-in replacement.

    Parameters
    ----------
    edge_index : (2, E) PyG edge list.
    num_nodes : n.
    k : number of eigenpairs to compute (default: min(200, n)).
    variant : Laplacian type (``"sym"`` or ``"comb"``).
    """

    @classmethod
    def from_sparse_laplacian(
        cls,
        L_sparse: torch.Tensor,
        k: int,
        seed: int = 0,
    ) -> "GraphEigenTruncated":
        """Build from a pre-computed sparse Laplacian (e.g. affinity-refined).

        Use this for weighted Laplacians that cannot be built from
        edge_index alone (e.g. compute_affinity_refined_laplacian_sparse).

        ``seed`` fixes the LOBPCG initial block. Before 2026-09 this was an
        unseeded ``torch.randn``; the eigenbasis (and every downstream
        truncated-spectrum score) then varied run to run.
        """
        num_nodes = L_sparse.shape[0]
        k = min(k, num_nodes)
        device = L_sparse.device

        gen = torch.Generator(device=device).manual_seed(int(seed))
        X0 = torch.randn(num_nodes, k, generator=gen, device=device, dtype=L_sparse.dtype)
        lam, V = torch.lobpcg(L_sparse, k=k, X=X0, largest=False, niter=300)
        sort_idx = lam.argsort()
        lam_L = lam[sort_idx]
        V_sorted = V[:, sort_idx]

        obj = object.__new__(cls)
        obj.n = num_nodes
        obj.k = k
        obj.L = L_sparse
        obj.lam_L = lam_L
        obj.V = V_sorted
        obj.device = V_sorted.device
        obj.dtype = V_sorted.dtype
        return obj

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        k: int = 200,
        variant: str = "sym",
        seed: int = 0,
    ):
        k = min(k, num_nodes)
        self.n = num_nodes
        self.k = k

        # Use dense eigh for small graphs, LOBPCG for large ones
        if num_nodes <= 2 * k:
            L = compute_laplacian(edge_index, num_nodes, variant=variant)
            lam_all, V_all = torch.linalg.eigh(L)
            self.lam_L = lam_all[:k]
            self.V = V_all[:, :k]
            self.L = L
        else:
            L_sparse = compute_sparse_laplacian(edge_index, num_nodes,
                                                 variant=variant)
            self.L = L_sparse
            # Seeded initial block: see from_sparse_laplacian.
            gen = torch.Generator(device=edge_index.device).manual_seed(int(seed))
            X0 = torch.randn(num_nodes, k, generator=gen, device=edge_index.device)
            lam, V = torch.lobpcg(L_sparse, k=k, X=X0, largest=False,
                                   niter=300)
            sort_idx = lam.argsort()
            self.lam_L = lam[sort_idx]
            self.V = V[:, sort_idx]

        self.device = self.V.device
        self.dtype = self.V.dtype

    # -- Core apply: V_k diag(f) V_k^T @ x  in O(n·k·d) --

    def apply(self, f_lam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply rank-k approximation  V_k diag(f_lam) V_k^T @ x."""
        return self.V @ (f_lam.unsqueeze(-1) * (self.V.T @ x))

    # -- Derived eigenvalue vectors (same API as GraphEigen) --

    def Q_eigenvalues(self, kappa: float = 1.0, nu: float = 1.0) -> torch.Tensor:
        return (kappa ** 2 + self.lam_L).pow(nu)

    def Q_inv_sqrt_eigenvalues(self, kappa: float = 1.0,
                               nu: float = 1.0) -> torch.Tensor:
        return (kappa ** 2 + self.lam_L).pow(-nu / 2.0)

    def K_low_eigenvalues(self, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(-tau * self.lam_L)

    def K_high_eigenvalues(self, tau: float = 1.0) -> torch.Tensor:
        return 1.0 - torch.exp(-tau * self.lam_L)

    # -- High-level helpers (same API as GraphEigen) --

    def filter_features(self, x: torch.Tensor,
                        tau: float = 1.0) -> torch.Tensor:
        """Graph-filter augmentation  [x, K_low_approx x, K_high_approx x].

        Note: K_high x = x - K_low x, so the rank-k approximation of K_high
        is exact as long as we compute it via the identity:
        K_high_approx x = x - K_low_approx x.
        """
        kl = self.K_low_eigenvalues(tau)
        x_low = self.apply(kl, x)
        x_high = x - x_low
        return torch.cat([x, x_low, x_high], dim=-1)

    def sample_matern_prior(self, d: int, kappa: float = 1.0,
                            nu: float = 1.0) -> torch.Tensor:
        """Sample X₀ ~ N(0, Q⁻¹) using rank-k approximation."""
        q_inv_sqrt_eig = self.Q_inv_sqrt_eigenvalues(kappa, nu)
        xi = torch.randn(self.n, d, device=self.device, dtype=self.dtype)
        return self.apply(q_inv_sqrt_eig, xi)

    def sample_two_band_prior(
        self, d: int, tau: float = 1.0,
        sigma_low: float = 1.0, sigma_high: float = 1.0,
    ) -> torch.Tensor:
        kl = self.K_low_eigenvalues(tau)
        xi = torch.randn(self.n, d, device=self.device, dtype=self.dtype)
        xi2 = torch.randn(self.n, d, device=self.device, dtype=self.dtype)
        x_low = self.apply(kl, xi)
        x_high = xi2 - self.apply(kl, xi2)  # (I - K_low) xi2
        return sigma_low * x_low + sigma_high * x_high

    def bridge_eigen_cache(self, kappa: float = 1.0,
                           nu: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Q_eigenvalues_k, V_k)`` for bridge_statistics."""
        return self.Q_eigenvalues(kappa, nu), self.V


# ═══════════════════════════════════════════════════════════════════════════
# 10.  ChebyshevFilter — no eigendecomposition, sparse matmuls only
# ═══════════════════════════════════════════════════════════════════════════

class ChebyshevFilter:
    """Chebyshev polynomial approximation of graph filters.

    Computes  exp(−τ L) x  ≈  Σ_{j=0}^{P} c_j T_j(L̃) x  using only
    sparse matrix–vector products, where L̃ = 2L/λ_max − I  is the
    rescaled Laplacian.

    No eigendecomposition is needed.  Per-step cost: O(P · E · d) where
    E = number of edges.

    Parameters
    ----------
    edge_index : (2, E) PyG edge list.
    num_nodes : n.
    order : Chebyshev polynomial order P.
    variant : Laplacian type.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        order: int = 15,
        variant: str = "sym",
    ):
        self.n = num_nodes
        self.order = order
        self.L_sparse = compute_sparse_laplacian(edge_index, num_nodes,
                                                  variant=variant)
        self.device = edge_index.device
        self.dtype = torch.float32

        # Estimate spectral radius via power iteration
        self.lam_max = self._estimate_spectral_radius(n_iter=50)

        # Rescaled Laplacian: L_tilde = 2L / lam_max - I
        scale = 2.0 / max(self.lam_max, 1e-6)
        n = num_nodes
        diag_idx = torch.arange(n, device=self.device)
        I_sparse = torch.sparse_coo_tensor(
            torch.stack([diag_idx, diag_idx]),
            torch.ones(n, device=self.device),
            (n, n),
        ).coalesce()
        self.L_tilde = (scale * self.L_sparse - I_sparse).coalesce()

    def _estimate_spectral_radius(self, n_iter: int = 50) -> float:
        """Estimate λ_max of L via power iteration."""
        x = torch.randn(self.n, 1, device=self.device)
        for _ in range(n_iter):
            x = torch.sparse.mm(self.L_sparse, x)
            norm = x.norm()
            if norm > 0:
                x = x / norm
        lam = (torch.sparse.mm(self.L_sparse, x) * x).sum().item()
        return max(lam, 1e-6)

    def _chebyshev_coeffs(self, tau: float) -> torch.Tensor:
        """Compute Chebyshev coefficients for exp(-tau * x) on [-1, 1].

        Uses the discrete cosine transform formula:
        c_j = (2/P) Σ_{k=0}^{P-1} f(cos(π(k+0.5)/P)) cos(πj(k+0.5)/P)
        with c_0 halved.
        """
        P = self.order
        # Map from [-1,1] to [0, lam_max]: x = lam_max*(s+1)/2
        # f(s) = exp(-tau * lam_max * (s+1)/2)
        k = torch.arange(P, dtype=torch.float64, device=self.device)
        nodes = torch.cos(math.pi * (k + 0.5) / P)  # Chebyshev nodes in [-1,1]
        lam_vals = self.lam_max * (nodes + 1.0) / 2.0  # map to [0, lam_max]
        f_vals = torch.exp(-tau * lam_vals)

        j = torch.arange(P, dtype=torch.float64, device=self.device)
        # c_j = (2/P) sum_k f(node_k) * cos(pi * j * (k+0.5) / P)
        args = math.pi * j.unsqueeze(1) * (k.unsqueeze(0) + 0.5) / P
        coeffs = (2.0 / P) * (f_vals.unsqueeze(0) * args.cos()).sum(dim=1)
        coeffs[0] = coeffs[0] / 2.0
        return coeffs.float()

    def apply_filter(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Compute  exp(−τ L) x  via Chebyshev approximation.

        Uses the Chebyshev recurrence:  T_0 = x,  T_1 = L̃ x,
        T_{j+1} = 2 L̃ T_j − T_{j-1}.

        Parameters
        ----------
        x : (n, d).
        tau : heat-kernel scale.

        Returns
        -------
        result : (n, d).
        """
        coeffs = self._chebyshev_coeffs(tau)
        P = len(coeffs)

        T_prev = x                                         # T_0(L̃) x = x
        T_curr = torch.sparse.mm(self.L_tilde, x)          # T_1(L̃) x = L̃ x
        result = coeffs[0] * T_prev
        if P > 1:
            result = result + coeffs[1] * T_curr
        for j in range(2, P):
            T_next = 2.0 * torch.sparse.mm(self.L_tilde, T_curr) - T_prev
            result = result + coeffs[j] * T_next
            T_prev = T_curr
            T_curr = T_next
        return result

    def filter_features(self, x: torch.Tensor,
                        tau: float = 1.0) -> torch.Tensor:
        """Graph-filter augmentation  [x, K_low x, K_high x]  via Chebyshev.

        Returns
        -------
        (n, 3d)
        """
        x_low = self.apply_filter(x, tau=tau)
        x_high = x - x_low
        return torch.cat([x, x_low, x_high], dim=-1)

    def spectral_high_energy(self, residual: torch.Tensor,
                             tau: float = 1.0) -> torch.Tensor:
        """Per-node high-frequency energy of a residual, without eigenvalues.

        Computes  ‖(I − exp(−τL)) r‖²  per node via Chebyshev.

        Parameters
        ----------
        residual : (n, d).
        tau : heat-kernel scale.

        Returns
        -------
        (n,)  per-node energy.
        """
        r_low = self.apply_filter(residual, tau=tau)
        r_high = residual - r_low
        return (r_high ** 2).sum(dim=-1)

    def spectral_ratio_score(self, residual: torch.Tensor,
                             tau: float = 1.0) -> torch.Tensor:
        """Spectral ratio score via Chebyshev (no eigenvalues needed).

        S_i = ‖K_high r‖² / (‖r‖² + ε)
        """
        high_energy = self.spectral_high_energy(residual, tau=tau)
        total_energy = (residual ** 2).sum(dim=-1).clamp(min=1e-8)
        return high_energy / total_energy
