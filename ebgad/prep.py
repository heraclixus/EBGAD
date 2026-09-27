"""Preprocessing: features, spectrum, template, residual.

Mirrors the non-neural part of ``soc.soc_trainer.SOCTrainer._setup`` so that
the paper's equilibrium numbers reproduce, but keeps everything in plain
numpy float64 after the eigendecomposition and caches eigenpairs on disk.

Order of operations (as in the code that produced the paper's tables):
    raw x -> [PCA on raw x] -> z-score -> spectrum -> template -> residual
The appendix states PCA after z-scoring; ``pca_order="paper"`` gives that.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from bridge.encoder import PCAEncoder
from bridge.graph_ops import (
    GraphEigenTruncated,
    compute_laplacian,
    compute_sparse_laplacian,
)
from soc.soc_bridge import (
    affinity_to_weights,
    apply_modeled_subspace,
    compute_affinity_refined_laplacian,
    compute_affinity_refined_laplacian_sparse,
    compute_local_affinity,
    compute_template,
)

LARGE_N = 20_000          # >= this: truncated LOBPCG spectrum (as in _setup)
AFFINITY_DENSE_N = 8_000  # affinity-refined Laplacian: dense eigh below this
BLOCK_MAX_SIZE = 5_000    # block-diagonal exact spectrum when every component is at most this large
FULL_BLOCK_MAX_N = 30_000 # ... and the dense n x n basis fits in memory (n^2 * 8 bytes)


def largest_component(L_sparse: torch.Tensor, n: int) -> int:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    L = L_sparse.coalesce()
    idx = L.indices().numpy()
    A = sp.csr_matrix((np.ones(idx.shape[1]), (idx[0], idx[1])), shape=(n, n))
    _, labels = connected_components(A, directed=False)
    return int(np.bincount(labels).max())


def auto_truncation(n: int) -> Optional[int]:
    """Paper's auto-k rule (SOCTrainer._setup)."""
    if n >= 1_000_000:
        return 128
    if n >= 500_000:
        return 200
    if n >= 100_000:
        return 300
    if n >= LARGE_N:
        return 500
    return None


def preprocess_features(
    x_raw: torch.Tensor,
    edge_index: torch.Tensor,
    pca: Optional[int] = None,
    pca_order: str = "code",
) -> torch.Tensor:
    """z-scored (optionally PCA-projected) features, float32 torch."""
    x = x_raw.float()
    use_pca = pca is not None and x.shape[1] > int(pca)
    if use_pca and pca_order == "code":
        enc = PCAEncoder(x.shape[1], int(pca))
        enc.fit(x)
        x = enc.encode(x, edge_index)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-8)  # unbiased, as torch default
    x = (x - mean) / std
    if use_pca and pca_order == "paper":
        enc = PCAEncoder(x.shape[1], int(pca))
        enc.fit(x)
        x = enc.encode(x, edge_index)
    return x


@dataclass
class Spectrum:
    lam: np.ndarray            # (k,) eigenvalues of the selected Laplacian
    V: np.ndarray              # (n, k) eigenvectors (float64)
    L_sparse: torch.Tensor     # sparse (n, n) Laplacian of the selected operator
    graph: str
    k: int
    truncated: bool


def _sparse_diag(L: torch.Tensor) -> torch.Tensor:
    L = L.coalesce()
    idx = L.indices()
    vals = L.values()
    mask = idx[0] == idx[1]
    diag = torch.zeros(L.shape[0], dtype=vals.dtype)
    diag[idx[0][mask]] = vals[mask]
    return diag


def block_diagonal_eigh(L_sparse: torch.Tensor, n: int, max_block: int = 5000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Full spectrum of a Laplacian with many small connected components.

    The Laplacian is block-diagonal over components, so its eigenpairs are
    the union of the components' eigenpairs. Returns (lam, V) with V dense
    n x n (memory n^2 * 8 bytes; caller checks n), or None if some component
    exceeds ``max_block`` nodes. YelpChi's R-U-R graph (23,831 nodes,
    7,308 components, largest 47) is the motivating case: its 500 smallest
    eigenvalues are all nullspace, so any truncated spectrum is degenerate.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    L = L_sparse.coalesce()
    idx = L.indices().numpy()
    vals = L.values().numpy().astype(np.float64)
    Ls = sp.csr_matrix((vals, (idx[0], idx[1])), shape=(n, n))
    ncomp, labels = connected_components(Ls, directed=False)
    sizes = np.bincount(labels)
    if sizes.max() > max_block:
        return None
    lam = np.empty(n)
    V = np.zeros((n, n))
    col = 0
    for c in range(ncomp):
        nodes = np.where(labels == c)[0]
        block = Ls[nodes][:, nodes].toarray()
        w, U = np.linalg.eigh(0.5 * (block + block.T))
        lam[col:col + len(nodes)] = w
        V[np.ix_(nodes, np.arange(col, col + len(nodes)))] = U
        col += len(nodes)
    return lam, V


def compute_spectrum(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n: int,
    graph: str = "original",
    k: Optional[int] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    cache_tag: str = "",
    force_full: bool = False,
) -> Spectrum:
    """Eigenpairs of the selected operator's symmetric-normalized Laplacian.

    ``graph="original"`` uses the input edge set; ``graph="affinity"`` uses
    the affinity-refined reweighting of the same edge set (Appendix B).
    Full dense eigh below LARGE_N nodes, seeded LOBPCG above, unless
    ``force_full`` (then dense, via the block-diagonal route when the graph
    is a union of small components; n^2 * 8 bytes of memory).
    """
    if graph == "original":
        L_sparse = compute_sparse_laplacian(edge_index, n, variant="sym")
    elif graph == "affinity":
        L_sparse = compute_affinity_refined_laplacian_sparse(edge_index, x, n, variant="sym")
    else:
        raise ValueError(f"unknown graph operator {graph!r}")

    # Structural rule (no per-dataset choice): a graph that is a union of small
    # components gets its exact full spectrum per component whenever the dense
    # n x n eigenbasis fits in memory; a truncated basis on such a graph is
    # pure nullspace (YelpChi R-U-R: 7,308 components, largest 47).
    if k is None and not force_full and n >= LARGE_N and n <= FULL_BLOCK_MAX_N:
        if largest_component(L_sparse, n) <= BLOCK_MAX_SIZE:
            force_full = True
    k_eff = None if force_full else (k if k is not None else auto_truncation(n))
    truncated = k_eff is not None
    key = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        h = hashlib.md5(f"{cache_tag}|{graph}|{k_eff}|{seed}|{n}".encode()).hexdigest()[:10]
        key = os.path.join(cache_dir, f"spec_{cache_tag}_{graph}_k{k_eff}_s{seed}_{h}.npz")

    if key and os.path.exists(key):
        z = np.load(key)
        return Spectrum(z["lam"], z["V"], L_sparse, graph, int(z["V"].shape[1]), truncated)

    # Dense paths run eigh in float64 (the paper's SOCTrainer path used float32,
    # which leaves ~1e-3 eigenvector error; exact nulls and the permutation
    # gate want better). Truncated paths keep the seeded LOBPCG in float32.
    if graph == "original":
        if not truncated:
            bd = block_diagonal_eigh(L_sparse, n) if (force_full or n >= LARGE_N) else None
            if bd is not None:
                lam, V = (torch.from_numpy(bd[0]), torch.from_numpy(bd[1]))
            else:
                L_dense = compute_laplacian(edge_index, n, variant="sym").double()
                lam, V = torch.linalg.eigh(L_dense)
                del L_dense
        else:
            ge = GraphEigenTruncated(edge_index, n, k=k_eff, variant="sym", seed=seed)
            lam, V = ge.lam_L, ge.V
    else:
        if n < AFFINITY_DENSE_N and not truncated:
            L_dense = compute_affinity_refined_laplacian(edge_index, x, n, variant="sym").double()
            lam, V = torch.linalg.eigh(L_dense)
            del L_dense
        else:
            ge = GraphEigenTruncated.from_sparse_laplacian(
                L_sparse, k=k_eff or auto_truncation(max(n, LARGE_N)) or 500, seed=seed,
            )
            lam, V = ge.lam_L, ge.V

    lam_np = lam.detach().cpu().numpy().astype(np.float64)
    V_np = V.detach().cpu().numpy().astype(np.float64)
    if truncated:
        lam_np, V_np = rayleigh_ritz_refine(L_sparse, V_np)
    order = np.argsort(lam_np)
    lam_np, V_np = lam_np[order], V_np[:, order]
    if key:
        np.savez(key, lam=lam_np, V=V_np)
    return Spectrum(lam_np, V_np, L_sparse, graph, int(V_np.shape[1]), truncated)


def rayleigh_ritz_refine(L_sparse: torch.Tensor, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize a LOBPCG basis in float64 and re-diagonalize L on it.

    torch.lobpcg returns a float32 basis whose orthonormality error (~1e-4)
    is enough to bias the exact chi-squared nulls when d is large (Questions,
    d = 301: KS 0.04 against the simulated null). Ritz vectors on the
    orthonormalized span fix this at O(n k^2 + k^3) cost.
    """
    W, _ = np.linalg.qr(V)
    L = L_sparse.coalesce().double()
    LW = torch.sparse.mm(L, torch.from_numpy(W)).numpy()
    H = W.T @ LW
    H = 0.5 * (H + H.T)
    lam, U = np.linalg.eigh(H)
    return lam, W @ U


@dataclass
class Prepared:
    name: str
    gamma: float
    template: str
    graph: str
    subspace: str
    n: int
    d: int
    k: int                    # modeled modes
    truncated: bool           # spectrum is a LOBPCG truncation (k << n)
    x: np.ndarray             # (n, d) processed features
    m: np.ndarray             # (n, d) template
    delta: np.ndarray         # (n, d) residual x - m
    lam: np.ndarray           # (k,) modeled eigenvalues
    V: np.ndarray             # (n, k) modeled eigenvectors
    Xi: np.ndarray            # (k, d) = V^T delta
    M_hat: np.ndarray         # (k, d) = V^T m
    S: np.ndarray             # (k,) spectral energies of delta
    lev: np.ndarray           # (n,) leverage h_i = sum_j V_ij^2
    delta_perp_sq: np.ndarray # (n,) ||delta_i||^2 - ||(V V^T delta)_i||^2
    L_sparse: torch.Tensor
    L_diag: np.ndarray        # (n,) diagonal of the sparse Laplacian
    edge_index: torch.Tensor


def prepare(
    name: str,
    data,
    gamma: float,
    template: str = "low",
    graph: str = "original",
    pca: Optional[int] = None,
    k: Optional[int] = None,
    subspace: str = "drop_nullspace",
    seed: int = 0,
    cache_dir: Optional[str] = None,
    pca_order: str = "code",
    _spectrum_cache: Optional[Dict[Tuple, Spectrum]] = None,
    force_full: bool = False,
) -> Prepared:
    """Build the residual field for one template bandwidth gamma."""
    x = preprocess_features(data.x, data.edge_index, pca=pca, pca_order=pca_order)
    n, d = x.shape
    edge_index = data.edge_index

    skey = (name, graph, k, seed, pca, pca_order, force_full)
    spec = None
    if _spectrum_cache is not None:
        spec = _spectrum_cache.get(skey)
    if spec is None:
        spec = compute_spectrum(
            x, edge_index, n, graph=graph, k=k, seed=seed,
            cache_dir=cache_dir, cache_tag=f"{name}_pca{pca}_{pca_order}_f64",
            force_full=force_full,
        )
        if _spectrum_cache is not None:
            _spectrum_cache[skey] = spec

    lam_all_t = torch.from_numpy(spec.lam).double()
    V_all_t = torch.from_numpy(spec.V).double()
    x64 = x.double()

    node_weights = None
    if template.startswith("affinity"):
        aff = compute_local_affinity(x, edge_index, n)
        node_weights = affinity_to_weights(aff).double()
    x_np = x64.numpy()
    if template.endswith("_unit"):
        nw = node_weights.numpy() if node_weights is not None else None
        m_np = unit_gain_template(x_np, spec.V, spec.lam, float(gamma), nw)
    else:
        m_t = compute_template(
            x64, V_all_t, lam_all_t, template_type=template, gamma=float(gamma), nu=1.0,
            node_weights=node_weights,
        )
        m_np = m_t.numpy().astype(np.float64)

    lam_mod_t, V_mod_t, _mask = apply_modeled_subspace(lam_all_t, V_all_t, mode=subspace)
    lam_mod = lam_mod_t.numpy().astype(np.float64)
    V_mod = V_mod_t.numpy().astype(np.float64)
    delta = x_np - m_np
    Xi = V_mod.T @ delta
    M_hat = V_mod.T @ m_np
    S = (Xi ** 2).sum(axis=1)
    V2 = V_mod ** 2
    lev = V2.sum(axis=1)
    proj_sq = ((V_mod @ Xi) ** 2).sum(axis=1)
    delta_perp_sq = np.maximum((delta ** 2).sum(axis=1) - proj_sq, 0.0)
    L_diag = _sparse_diag(spec.L_sparse).numpy().astype(np.float64)

    return Prepared(
        name=name, gamma=float(gamma), template=template, graph=graph, subspace=subspace,
        n=int(n), d=int(d), k=int(V_mod.shape[1]), truncated=bool(spec.truncated),
        x=x_np, m=m_np, delta=delta, lam=lam_mod, V=V_mod, Xi=Xi, M_hat=M_hat, S=S,
        lev=lev, delta_perp_sq=delta_perp_sq,
        L_sparse=spec.L_sparse, L_diag=L_diag, edge_index=edge_index,
    )


def unit_gain_template(x: np.ndarray, V: np.ndarray, lam: np.ndarray, gamma: float,
                       node_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit-DC-gain Matern template m = V diag(gamma^2 / (gamma^2 + lam)) V^T (W x).

    Gains lie in (0, 1], so the residual operator T = I - S has eigenvalues
    lam / (gamma^2 + lam) in [0, 1): singular only on the Laplacian
    nullspace, which is exactly the unmodeled subspace. This makes the
    change-of-variables likelihood of x (with the Jacobian) well defined
    for every gamma, so gamma can be EB-fitted like rho and kappa
. The paper's unnormalized template has gains
    1 / (gamma^2 + lam) that exceed one and a singular T at
    lam = 1 - gamma^2 for gamma < 1.
    """
    xw = x * node_weights[:, None] if node_weights is not None else x
    g = gamma ** 2 / (gamma ** 2 + np.maximum(lam, 0.0))
    return V @ (g[:, None] * (V.T @ xw))


def permute_data(data, perm: np.ndarray):
    """Relabel nodes by ``perm`` (new index i holds old node perm[i])."""
    from copy import copy
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    inv_t = torch.from_numpy(inv).long()
    out = copy(data)
    out.x = data.x[torch.from_numpy(perm).long()]
    out.y = data.y[torch.from_numpy(perm).long()]
    out.edge_index = inv_t[data.edge_index]
    return out
