"""Edge leg of the EB smoothing model: is a node more or less similar to its neighbors than normal nodes are?

For an edge (i, j) the observed similarity is the cosine r_ij = <x_i, x_j> / (|x_i| |x_j|) of the two nodes'
standardized feature vectors over the d columns.  Under the model the d column pairs are i.i.d. bivariate
Gaussian; in the population-mean limit (the template the fit chooses on Elliptic and DGraph) the two vectors
are independent and isotropic, so sqrt(d - 1) r_ij / sqrt(1 - r_ij^2) is t_{d-1} exactly and r_ij is
independent of |x_i| and |x_j| (the local leg).  In general the pairs are correlated (rho_ij); as the
log-normal scale prior does for s_i, the population of edges calibrates the leg: Fisher's z_ij = atanh r_ij
is centered and scaled by the robust location and spread over all edges, and the node statistic
e_i = sum_{j ~ i} z~_ij / sqrt(deg_i) is N(0, 1) for a node whose edges are typical.  Readings: DEV = -e_i
(less similar to its neighbors than normal: injected cliques, heterophilous edges), CAM = +e_i (more similar:
camouflaged fraud clusters), TWO-SIDED = |e_i|.  Cost O(|E| d), no solve, invariant to relabeling, to
per-feature affine maps (standardization) and to per-node scaling.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def undirected_edges(edge_index: np.ndarray, n: int):
    """Unique undirected edges (a < b), self loops removed."""
    ei = np.asarray(edge_index)
    a = np.minimum(ei[0], ei[1]).astype(np.int64); b = np.maximum(ei[0], ei[1]).astype(np.int64)
    m = a != b
    key = np.unique(a[m] * n + b[m])
    return key // n, key % n


def edge_leg(x: np.ndarray, edge_index: np.ndarray, chunk: int = 2_000_000) -> dict:
    """Node statistic e (n,), with the population calibration constants and per-node degrees."""
    x = np.asarray(x, dtype=np.float64); n, d = x.shape
    ia, ib = undirected_edges(edge_index, n)
    deg = (np.bincount(ia, minlength=n) + np.bincount(ib, minlength=n)).astype(np.float64)
    xu = x / (np.sqrt((x * x).sum(1)) + 1e-12)[:, None]
    r = np.empty(len(ia))
    for s in range(0, len(ia), chunk):
        r[s:s + chunk] = (xu[ia[s:s + chunk]] * xu[ib[s:s + chunk]]).sum(1)
    r = np.clip(r, -0.999, 0.999)
    z = np.arctanh(r)
    med = float(np.median(z)); mad = float(1.4826 * np.median(np.abs(z - med))) + 1e-12
    zs = (z - med) / mad
    acc = np.zeros(n); np.add.at(acc, ia, zs); np.add.at(acc, ib, zs)
    e = acc / np.sqrt(np.maximum(deg, 1.0))
    e[deg == 0] = 0.0
    return {"e": e, "deg": deg, "z_med": med, "z_mad": mad, "r_med": float(np.median(r)), "n_edges": int(len(ia))}


def edge_surprises(e: np.ndarray) -> dict:
    """-log p in each reading (DEV: lower tail of e, CAM: upper tail, TWO-SIDED)."""
    dev = -norm.logcdf(e); cam = -norm.logsf(e); two = -np.log(np.clip(2 * norm.sf(np.abs(e)), 1e-300, 1.0))
    return {"DEV": dev, "CAM": cam, "TWO-SIDED": two}
