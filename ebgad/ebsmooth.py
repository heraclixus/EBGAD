"""Matrix-free EB smoothing template.

Model, per feature column: x = f + eps with f ~ N(0, (a (kappa^2 I + L))^{-1})
and eps ~ N(0, sigma^2 I), L the symmetric-normalized Laplacian. Marginally
x ~ N(0, v(L)) with v(lam) = sigma^2 + 1/(a (kappa^2 + lam)). The template
is the posterior mean S x = x - sigma^2 v(L)^{-1} x (Wiener gain
g2 / (g2 + kappa^2 + lam), g2 = 1/(a sigma^2): the paper's unit-gain
low-pass template with its horizon fitted by empirical Bayes), the residual
delta = sigma^2 v(L)^{-1} x has null covariance sigma^2 (I - S) =
sigma^4 v(L)^{-1}, and the node statistic
    s_i = ||delta_i||^2 / (d tau_i^2),  tau_i^2 = sigma^4 [v(L)^{-1}]_ii,
has an exact chi^2_d / d null per node under the model.

Everything is computed without an eigendecomposition:
    v(L)^{-1} = a K B^{-1},  K = kappa^2 I + L,  B = I + sigma^2 a K  (SPD, commuting),
so v(L)^{-1} x needs one block conjugate-gradient solve, the log-determinant
tr log v(L) uses stochastic Lanczos quadrature with Ritz nodes and weights
computed once and reused for every hyperparameter value, and the diagonal
[v(L)^{-1}]_ii uses a Hutchinson estimate through the same solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import minimize

from bridge.graph_ops import compute_sparse_laplacian


def laplacian_csr(edge_index: torch.Tensor, n: int, variant: str = "sym") -> sp.csr_matrix:
    """Symmetric-normalized ("sym", spectrum in [0, 2]) or combinatorial ("comb", D - A) Laplacian."""
    L = compute_sparse_laplacian(edge_index, n, variant=variant).coalesce()
    idx = L.indices().numpy(); val = L.values().double().numpy()
    return sp.csr_matrix((val, (idx[0], idx[1])), shape=(n, n))


def rcm_order(L: sp.csr_matrix) -> np.ndarray:
    """Reverse Cuthill-McKee node order (cache-friendly sparse products; results are permutation-invariant)."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    return np.ascontiguousarray(reverse_cuthill_mckee(L.tocsr(), symmetric_mode=True)).astype(np.int64)


def spectral_bound(L: sp.csr_matrix) -> float:
    """Gershgorin upper bound on the largest eigenvalue of L."""
    return float(np.abs(L).sum(axis=1).max())


class _SparseOp:
    """L @ Y with a scipy CSR product (small graphs) or a multithreaded torch CSR product (large)."""

    def __init__(self, L: sp.csr_matrix, backend: str = "auto"):
        import sys
        self.csr = L.tocsr()
        self.shape = L.shape
        n = L.shape[0]
        # torch only pays off for multi-column products on very large graphs; on macOS the
        # torch CSR product of a single vector was found to return NaN intermittently
        # (torch 2.3.1, Elliptic), so single vectors always go through scipy there.
        self.use_torch = backend == "torch" or (backend == "auto" and n >= 1_000_000)
        self.torch_1d = self.use_torch and sys.platform.startswith("linux")
        if self.use_torch:
            self.t = torch.sparse_csr_tensor(torch.from_numpy(self.csr.indptr.astype(np.int64)),
                                             torch.from_numpy(self.csr.indices.astype(np.int64)),
                                             torch.from_numpy(self.csr.data.astype(np.float64)), size=self.shape)

    def __matmul__(self, Y):
        if self.use_torch and (Y.ndim > 1 or self.torch_1d):
            Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float64))
            return (self.t @ Yt).numpy().copy()
        return self.csr @ Y

    def matmul_t(self, Yt: torch.Tensor) -> torch.Tensor:
        return self.t @ Yt

    def diagonal(self):
        return self.csr.diagonal()


# ----------------------------------------------------------------------------- solvers

def block_cg(matvec: Callable[[np.ndarray], np.ndarray], X: np.ndarray, tol: float = 1e-8,
             maxiter: int = 500, X0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Solve B Y = X column-wise with CG (B SPD through ``matvec``)."""
    Y = np.zeros_like(X) if X0 is None else X0.copy()
    R = X - (matvec(Y) if X0 is not None else 0.0)
    P = R.copy()
    rr = (R * R).sum(axis=0)
    bnorm = np.sqrt((X * X).sum(axis=0)) + 1e-300
    it = 0
    for it in range(1, maxiter + 1):
        AP = matvec(P)
        pap = (P * AP).sum(axis=0)
        alpha = np.where(pap > 0, rr / np.where(pap > 0, pap, 1.0), 0.0)     # converged columns stop moving
        Y += alpha * P
        R -= alpha * AP
        rr_new = (R * R).sum(axis=0)
        if np.all(np.sqrt(rr_new) <= tol * bnorm):
            break
        beta = np.where(rr > 0, rr_new / np.where(rr > 0, rr, 1.0), 0.0)
        P = R + beta * P
        rr = rr_new
    return Y, it


def block_cg_torch(Lop: "_SparseOp", c1: float, c2: float, X: np.ndarray, tol: float = 1e-8,
                   maxiter: int = 500, X0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Solve (c1 I + c2 L) Y = X column-wise with CG, all vector work in multithreaded torch."""
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
    mv = lambda Yt: c1 * Yt + c2 * Lop.matmul_t(Yt)
    if X0 is None:
        Y = torch.zeros_like(Xt); R = Xt.clone()
    else:
        Y = torch.from_numpy(np.ascontiguousarray(X0, dtype=np.float64)).clone(); R = Xt - mv(Y)
    P = R.clone()
    rr = (R * R).sum(dim=0)
    bnorm = torch.sqrt((Xt * Xt).sum(dim=0)) + 1e-300
    it = 0
    for it in range(1, maxiter + 1):
        AP = mv(P)
        pap = (P * AP).sum(dim=0)
        alpha = torch.where(pap > 0, rr / torch.where(pap > 0, pap, torch.ones_like(pap)), torch.zeros_like(pap))
        Y.add_(alpha * P)
        R.sub_(alpha * AP)
        rr_new = (R * R).sum(dim=0)
        if bool((torch.sqrt(rr_new) <= tol * bnorm).all()):
            break
        beta = torch.where(rr > 0, rr_new / torch.where(rr > 0, rr, torch.ones_like(rr)), torch.zeros_like(rr))
        P = R + beta * P
        rr = rr_new
    return Y.numpy(), it


def lanczos_quadrature(L, n_probes: int = 20, m: int = 80, seed: int = 0,
                       reorth_limit_bytes: float = 2e9) -> Tuple[np.ndarray, np.ndarray]:
    """Ritz nodes and weights for tr f(L) ~ (n / P) sum_p sum_k w_pk f(theta_pk)."""
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    reorth = n * m * 8 <= reorth_limit_bytes
    nodes = np.zeros((n_probes, m)); weights = np.zeros((n_probes, m))
    for p in range(n_probes):
        z = rng.choice([-1.0, 1.0], size=n); z /= np.linalg.norm(z)
        Q = np.zeros((m + 1, n)) if reorth else None
        alpha = np.zeros(m); beta = np.zeros(m)
        q_prev = np.zeros(n); q = z
        for j in range(m):
            if reorth:
                Q[j] = q
            w = L @ q
            alpha[j] = q @ w
            w -= alpha[j] * q + (beta[j - 1] * q_prev if j > 0 else 0.0)
            if reorth:
                w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-12 * max(1.0, abs(alpha[j])):
                alpha, beta = alpha[:j + 1], beta[:j + 1]
                break
            q_prev, q = q, w / beta[j]
        k = alpha.size
        T = np.diag(alpha) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
        if not np.isfinite(T).all():
            raise FloatingPointError("Lanczos produced non-finite recurrence coefficients (probe %d)" % p)
        theta, U = np.linalg.eigh(T)
        nodes[p, :k] = theta; weights[p, :k] = U[0] ** 2
    return nodes, weights


def chebyshev_band_coefficients(edges: np.ndarray, lam_max: float, degree: int) -> np.ndarray:
    """Jackson-damped Chebyshev coefficients (T, degree+1) of the indicators of [edges[b], edges[b+1]) on [0, lam_max]."""
    T = len(edges) - 1
    K = degree
    k = np.arange(K + 1)
    jackson = ((K - k + 1) * np.cos(np.pi * k / (K + 1)) + np.sin(np.pi * k / (K + 1)) / np.tan(np.pi / (K + 1))) / (K + 1)
    C = np.zeros((T, K + 1))
    for b in range(T):
        ya, yb = (2 * edges[b] - lam_max) / lam_max, (2 * edges[b + 1] - lam_max) / lam_max
        th_a, th_b = np.arccos(np.clip(ya, -1, 1)), np.arccos(np.clip(yb, -1, 1))   # th_a >= th_b
        C[b, 0] = (th_a - th_b) / np.pi
        C[b, 1:] = 2.0 / (np.pi * k[1:]) * (np.sin(k[1:] * th_a) - np.sin(k[1:] * th_b))
    return C * jackson[None, :]


def apply_band_filters(L, lam_max: float, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(T, n, d) = h_b(L) X for the Chebyshev filters with coefficients C, one recurrence pass."""
    T, Kp1 = C.shape
    scale = 2.0 / lam_max
    Lt = lambda Y: scale * (L @ Y) - Y            # L mapped to [-1, 1]
    out = np.zeros((T,) + X.shape)
    T0, T1 = X, Lt(X)
    out += C[:, 0][:, None, None] * T0[None]
    out += C[:, 1][:, None, None] * T1[None]
    for k in range(2, Kp1):
        T0, T1 = T1, 2.0 * Lt(T1) - T0
        out += C[:, k][:, None, None] * T1[None]
    return out


class AMGPreconditioner:
    """Multi-column V-cycle on a plain-aggregation hierarchy of L + delta I, with weighted-Jacobi
    smoothing (every operation is a sparse-times-dense product, so d right-hand sides cost about
    one). Used to precondition B = (1 + c kappa^2) I + c L when the smoothing strength c is large:
    B / c = L + delta_c I with delta_c = (1 + c kappa^2) / c, so one hierarchy built for a small
    delta serves every strongly smoothed evaluation."""

    def __init__(self, L: sp.csr_matrix, delta: float = 1e-3, max_coarse: int = 500, presmooth: int = 2,
                 postsmooth: int = 2, omega: float = 2.0 / 3.0, backend: str = "auto"):
        import pyamg
        from pyamg.util.linalg import approximate_spectral_radius
        A = (L + delta * sp.eye(L.shape[0], format="csr")).tocsr()
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", smooth=None, max_coarse=max_coarse)
        self.levels = []
        for lv in ml.levels:
            A_l = lv.A.tocsr()
            dinv = 1.0 / np.maximum(A_l.diagonal(), 1e-12)
            # weighted Jacobi weight scaled by the spectral radius of D^{-1} A
            rho = float(approximate_spectral_radius(sp.diags(dinv) @ A_l))
            has_P = hasattr(lv, "P")
            self.levels.append({"A": _SparseOp(A_l, backend), "dinv": dinv, "w": omega / max(rho, 1e-12),
                                "P": _SparseOp(lv.P.tocsr(), backend) if has_P else None,
                                "PT": _SparseOp(lv.P.T.tocsr(), backend) if has_P else None})
        A_c = self.levels[-1]["A"].csr.toarray()
        self.coarse_inv = np.linalg.pinv(A_c)
        self.pre, self.post = presmooth, postsmooth
        self.complexity = float(ml.operator_complexity())
        self.n_levels = len(self.levels)

    def _smooth(self, lv, X, B, sweeps):
        A, dinv, w = lv["A"], lv["dinv"], lv["w"]
        for _ in range(sweeps):
            X = X + w * (dinv[:, None] * (B - A @ X))
        return X

    def _vcycle(self, i, B):
        lv = self.levels[i]
        if i == self.n_levels - 1:
            return self.coarse_inv @ B
        X = self._smooth(lv, np.zeros_like(B), B, self.pre)
        R = B - lv["A"] @ X
        Rc = lv["PT"] @ R
        Xc = self._vcycle(i + 1, Rc)
        X = X + lv["P"] @ Xc
        return self._smooth(lv, X, B, self.post)

    def __call__(self, R: np.ndarray) -> np.ndarray:
        return self._vcycle(0, R)


def block_pcg(matvec: Callable[[np.ndarray], np.ndarray], X: np.ndarray, precond: Callable[[np.ndarray], np.ndarray],
              tol: float = 1e-8, maxiter: int = 500, X0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Preconditioned CG, columns independent (numpy)."""
    Y = np.zeros_like(X) if X0 is None else X0.copy()
    R = X - (matvec(Y) if X0 is not None else 0.0)
    Z = precond(R); P = Z.copy()
    rz = (R * Z).sum(axis=0)
    bnorm = np.sqrt((X * X).sum(axis=0)) + 1e-300
    it = 0
    for it in range(1, maxiter + 1):
        AP = matvec(P)
        pap = (P * AP).sum(axis=0)
        alpha = np.where(pap > 0, rz / np.where(pap > 0, pap, 1.0), 0.0)
        Y += alpha * P
        R -= alpha * AP
        if np.all(np.sqrt((R * R).sum(axis=0)) <= tol * bnorm):
            break
        Z = precond(R)
        rz_new = (R * Z).sum(axis=0)
        beta = np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
        P = Z + beta * P
        rz = rz_new
    return Y, it


# ----------------------------------------------------------------------------- model

@dataclass
class EBSmoothFit:
    sigma2: float
    a: float
    kappa2: float
    nll: float
    n_eval: int

    @property
    def g2(self) -> float:
        return 1.0 / (self.a * self.sigma2)

    converged: bool = True
    message: str = ""
    column_scales: bool = False

    def as_dict(self) -> Dict:
        return {"sigma2": self.sigma2, "a": self.a, "kappa2": self.kappa2, "g2": self.g2,
                "nll": self.nll, "n_eval": self.n_eval, "converged": self.converged, "message": self.message,
                "column_scales": self.column_scales}


class EBSmoother:
    """EB signal-plus-noise smoother on a graph, matrix-free."""

    def __init__(self, L: sp.csr_matrix, x: np.ndarray, n_probes: int = 20, m: int = 80,
                 seed: int = 0, cg_tol: float = 1e-7, verbose: bool = False, backend: str = "auto",
                 lam_max: Optional[float] = None):
        self.verbose = verbose
        self.L = _SparseOp(L, backend); self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.n, self.d = x.shape
        # Spectral range for the Chebyshev filters and the quadrature clipping. The symmetric-
        # normalized Laplacian has spectrum in [0, 2] (its Gershgorin bound is loose on hubs:
        # 23.7 on DGraph); pass lam_max explicitly for other operators (comb: Gershgorin bound).
        self.lam_max = float(lam_max) if lam_max is not None else 2.0
        self.nodes, self.weights = lanczos_quadrature(self.L, n_probes=n_probes, m=m, seed=seed)
        self.n_probes = n_probes
        self.cg_tol = cg_tol
        self._y_cache: Optional[np.ndarray] = None
        self._fitting = False
        self.n_eval = 0

    # tr f(L) by the stored quadrature
    def trace_f(self, f: Callable[[np.ndarray], np.ndarray]) -> float:
        th = np.clip(self.nodes, 0.0, self.lam_max)
        return float(self.n / self.n_probes * np.sum(self.weights * f(th)))

    def _B_matvec(self, sigma2: float, a: float, kappa2: float):
        c = sigma2 * a
        def mv(Y):
            return (1.0 + c * kappa2) * Y + c * (self.L @ Y)
        return mv

    def solve_B(self, X: np.ndarray, sigma2: float, a: float, kappa2: float,
                X0: Optional[np.ndarray] = None) -> np.ndarray:
        return self._solve_B_its(X, sigma2, a, kappa2, X0)[0]

    cg_maxiter: int = 500        # scoring budget; the fit uses cg_maxiter_fit for exploratory evaluations
    cg_maxiter_fit: int = 500     # exact solves during the fit: an iteration cap makes f and its gradient inconsistent
                                  # at far probes and L-BFGS-B's line search then fails at the initial point

    def _precond(self):
        if getattr(self, "_amg", None) is None:
            self._amg = AMGPreconditioner(self.L.csr, backend="torch" if self.L.use_torch else "scipy")
        return self._amg

    def _solve_B_its(self, X, sigma2, a, kappa2, X0=None, maxiter=None):
        c = sigma2 * a
        mi = maxiter or self.cg_maxiter
        if c >= self.C_PCG:
            return block_pcg(self._B_matvec(sigma2, a, kappa2), X, self._precond(), tol=self.cg_tol, X0=X0, maxiter=mi)
        if self.L.use_torch:
            return block_cg_torch(self.L, 1.0 + c * kappa2, c, X, tol=self.cg_tol, X0=X0, maxiter=mi)
        return block_cg(self._B_matvec(sigma2, a, kappa2), X, tol=self.cg_tol, X0=X0, maxiter=mi)

    def nll_grad(self, sigma2: float, a: float, kappa2: float) -> Tuple[float, np.ndarray]:
        """NLL = d tr log v(L) + sum_f x_f^T v(L)^{-1} x_f (constants dropped) and its gradient
        in (log sigma2, log a, log kappa2).

        With K = kappa2 I + L, B = I + sigma2 a K, v^{-1} = a K B^{-1}:
          d v^{-1}/d sigma2 = -(v^{-1})^2,  d v^{-1}/d a = K B^{-2},  d v^{-1}/d kappa2 = a B^{-2},
        so the quadratic term's gradient needs only y = B^{-1} x and u = v^{-1} x = a K y, and the
        trace terms are spectral sums over the stored Lanczos nodes.
        """
        self.n_eval += 1
        v = lambda lam: sigma2 + 1.0 / (a * (kappa2 + lam))
        term1 = self.d * self.trace_f(lambda lam: np.log(v(lam)))
        import time as _t
        t0 = _t.time()
        Y, its = self._solve_B_its(self.x, sigma2, a, kappa2, X0=self._y_cache,
                                   maxiter=self.cg_maxiter_fit if self._fitting else None)
        self._y_cache = Y
        if self.verbose:
            print(f"    nll eval {self.n_eval}: sigma2={sigma2:.4g} a={a:.4g} kappa2={kappa2:.4g}  cg iters={its}  {_t.time() - t0:.1f}s", flush=True)
        KY = kappa2 * Y + self.L @ Y
        U = a * KY                                   # v(L)^{-1} x
        term2 = float((self.x * U).sum())
        # trace derivatives: d/d sigma2 = d tr v^{-1}; d/d a = -(d/a) tr B^{-1}; d/d kappa2 = -d tr(B^{-1} K^{-1})
        Binv = lambda lam: 1.0 / (1.0 + sigma2 * a * (kappa2 + lam))
        g_s2 = self.d * self.trace_f(lambda lam: 1.0 / v(lam)) - float((U * U).sum())
        g_a = -(self.d / a) * self.trace_f(Binv) + float((Y * KY).sum())
        g_k2 = -self.d * self.trace_f(lambda lam: Binv(lam) / (kappa2 + lam)) + a * float((Y * Y).sum())
        grad = np.array([sigma2 * g_s2, a * g_a, kappa2 * g_k2])
        return term1 + term2, grad

    def nll(self, sigma2: float, a: float, kappa2: float) -> float:
        return self.nll_grad(sigma2, a, kappa2)[0]

    C_MAX = 1e5       # smoothing strength c = sigma^2 a (gain 1/(1 + c(kappa^2 + lam))); solves above C_PCG are
    C_PCG = 100.0     # preconditioned by a plain-aggregation AMG V-cycle, so the cost stays bounded in c

    def fit_column_scales(self, n_rounds: int = 3, n_starts: int = 3) -> "EBSmoothFit":
        """EB column scales: x_f ~ N(0, c_f v_theta(L)), columns are replicates up to a scale.

        For fixed theta the maximum-likelihood scale of column f is c_f = x_f^T v^{-1} x_f / n,
        which the block solve of the likelihood already provides (a x_f^T K y_f). The fit
        alternates theta | c (the usual fit on X / sqrt(c)) and c | theta; the returned fit and
        the scores are on the rescaled features, stored back in self.x (self.col_scales keeps c).
        """
        x0 = self.x.copy()
        c = np.ones(self.d)
        fit = None
        for _ in range(n_rounds):
            self.x = np.ascontiguousarray(x0 / np.sqrt(c)[None, :])
            fit = self.fit(n_starts=n_starts)
            Y = self.solve_B(self.x, fit.sigma2, fit.a, fit.kappa2)
            KY = fit.kappa2 * Y + self.L @ Y
            quad = fit.a * (self.x * KY).sum(axis=0)          # x_f^T v^{-1} x_f on the rescaled columns
            c = c * quad / self.n                              # update: absorb the residual scale
            c = np.maximum(c, 1e-8)
        self.x = np.ascontiguousarray(x0 / np.sqrt(c)[None, :])
        self.col_scales = c
        fit.column_scales = True
        return fit

    def fit(self, n_starts: int = 3, fixed_sigma2: Optional[float] = None, cg_tol_fit: float = 1e-5) -> EBSmoothFit:
        """Maximize the marginal likelihood over (log sigma^2, log c, log kappa^2), c = sigma^2 a.

        c is the smoothing strength of the template (gain 1 / (1 + c (kappa^2 + lam))); beyond
        c ~ 300 the template is the component mean on every mode with lam > 0.01 and the
        likelihood is flat, while the system B = I + c K has condition number ~ 2c, so c is
        capped at C_MAX. With ``fixed_sigma2`` only (c, kappa^2) move (likelihood profile).
        The CG tolerance is loosened during the fit (``cg_tol_fit``); scoring uses ``cg_tol``.
        """
        tot = float((self.x * self.x).sum() / (self.n * self.d))
        inits = [(0.5, 3.0, 1e-2), (0.9, 0.3, 0.3), (0.2, 30.0, 1e-3), (0.7, 1.0, 1.0),
                 (0.95, 0.1, 3.0), (0.3, 10.0, 0.1)][:n_starts]
        best = None
        scale = 1.0 / (self.n * self.d)
        box = [(-14.0, 4.0), (-8.0, float(np.log(self.C_MAX))), (-14.0, 6.0)]   # log sigma2, log c, log kappa2
        tol_save, self.cg_tol = self.cg_tol, cg_tol_fit
        self._fitting = True

        def unpack(p):
            s2, c, k2 = np.exp(p)
            return s2, c / s2, k2

        def value_grad(p):
            s2, a, k2 = unpack(p)
            v, g = self.nll_grad(s2, a, k2)           # g in (log s2 | a fixed, log a, log k2)
            if not (np.isfinite(v) and np.isfinite(g).all()):
                raise FloatingPointError("non-finite marginal likelihood at sigma2=%g a=%g kappa2=%g" % (s2, a, k2))
            g_c = np.array([g[0] - g[1], g[1], g[2]])  # chain rule: log a = log c - log s2
            return v * scale, g_c * scale

        try:
            for s2f, af, k2 in inits:
                self._y_cache = None
                s2_0, a_0 = tot * s2f, af / tot
                if fixed_sigma2 is None:
                    p0 = np.log([s2_0, min(s2_0 * a_0, self.C_MAX / 2), k2])
                    r = minimize(value_grad, p0, jac=True, method="L-BFGS-B", bounds=box,
                                 options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-5, "maxls": 20})
                else:
                    def f(q):
                        v, g = value_grad(np.concatenate([[np.log(fixed_sigma2)], q]))
                        return v, g[1:]
                    p0 = np.log([min(fixed_sigma2 * a_0, self.C_MAX / 2), k2])
                    r = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=box[1:],
                                 options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-5, "maxls": 20})
                    r.x = np.concatenate([[np.log(fixed_sigma2)], r.x])
                if best is None or r.fun < best.fun:
                    best = r
        finally:
            self.cg_tol = tol_save
            self._fitting = False
        s2, a, k2 = unpack(best.x)
        self._y_cache = None
        fit = EBSmoothFit(float(s2), float(a), float(k2), float(best.fun / scale), self.n_eval)
        fit.converged = bool(best.nit >= 1)
        fit.message = str(best.message)
        return fit

    def spectral_edges(self, n_bands: int) -> np.ndarray:
        """Band edges with equal spectral mass (mode counts), from the stored Lanczos quadrature."""
        th = np.clip(self.nodes, 0.0, self.lam_max).ravel(); w = self.weights.ravel()
        order = np.argsort(th); th, w = th[order], w[order] / w.sum()
        cdf = np.cumsum(w)
        inner = [float(th[np.searchsorted(cdf, q)]) for q in np.arange(1, n_bands) / n_bands]
        return np.array([0.0] + inner + [self.lam_max])

    def band_profile(self, fit: EBSmoothFit, res: Dict[str, np.ndarray], n_bands: int = 4,
                     degree: int = 50, n_hutch: int = 128, seed: int = 2):
        """Relaxation-band energies of the residual and their null scales, matrix-free.

        Bands partition the spectrum of L into n_bands intervals of equal spectral mass
        (the v3 bands: equal mode counts by relaxation rate); h_b(L) is the Jackson-damped
        Chebyshev approximation of the interval indicator. E_ib = ||(h_b(L) delta)_i||^2
        with null scale sig2_ib = sigma^4 [h_b(L)^2 v(L)^{-1}]_ii (Hutchinson), so that
        E_ib / sig2_ib ~ chi^2_d under the model, approximately independent across bands.
        Returns (E, sig2, names) for surprise.surprise_from_bands.
        """
        s2, a, k2 = fit.sigma2, fit.a, fit.kappa2
        edges = self.spectral_edges(n_bands)
        C = chebyshev_band_coefficients(edges, self.lam_max, degree)
        # Work in the whitened scale (divide by sigma^4): E and sig2 both carry sigma^4, and the
        # absolute leverage threshold in surprise.py expects O(1) null scales.
        HD = apply_band_filters(self.L, self.lam_max, C, res["delta"] / s2)      # (T, n, d)
        E = (HD * HD).sum(axis=2).T                                             # (n, T)
        rng = np.random.default_rng(seed)
        Z = rng.choice([-1.0, 1.0], size=(self.n, n_hutch))
        YZ = self.solve_B(Z, s2, a, k2)
        MZ = a * (k2 * YZ + self.L @ YZ)                                         # v^{-1} Z
        HM = apply_band_filters(self.L, self.lam_max, C, MZ)                    # h_b v^{-1} Z
        sig2 = np.zeros((self.n, len(edges) - 1))
        for b in range(sig2.shape[1]):
            HHM = apply_band_filters(self.L, self.lam_max, C[b:b + 1], HM[b])[0]  # h_b^2 v^{-1} Z
            sig2[:, b] = (Z * HHM).mean(axis=1)                                   # [h_b^2 v^{-1}]_ii
        # a Hutchinson estimate at or below zero means the node has no leverage on the band
        sig2 = np.where(sig2 > 0, sig2, 0.0)
        names = ["band%d[%.2g,%.2g)" % (b + 1, edges[b], edges[b + 1]) for b in range(sig2.shape[1])]
        return E, sig2, names

    def regional_scale(self, fit: EBSmoothFit, res: Dict[str, np.ndarray], n_hutch: int = 128, seed: int = 3) -> Dict[str, np.ndarray]:
        """The second leg of the relaxation: the template against the population.

        m = S x is Gaussian under the model with Cov(m) = S v(L) S per feature, so
        g_i = ||m_i||^2 / (d Var(m_i)) ~ chi^2_d / d exactly. Var(m_i) = [S v(L) S]_ii by
        Hutchinson: S z = z - sigma^2 a K B^{-1} z (one B-solve), v(L) w = sigma^2 w + K^{-1} w / a
        (one K-solve, AMG-preconditioned since kappa^2 is small).
        """
        s2, a, k2 = fit.sigma2, fit.a, fit.kappa2
        Kmv = lambda Y: k2 * Y + self.L @ Y
        m = res["template"]
        rng = np.random.default_rng(seed)
        Z = rng.choice([-1.0, 1.0], size=(self.n, n_hutch))
        def S_apply(Y):
            return Y - s2 * a * Kmv(self.solve_B(Y, s2, a, k2))
        SZ = S_apply(Z)
        KinvSZ, _ = block_pcg(Kmv, SZ, self._precond(), tol=self.cg_tol, maxiter=self.cg_maxiter)
        vSZ = s2 * SZ + KinvSZ / a
        SvSZ = S_apply(vSZ)
        var_m = np.maximum((Z * SvSZ).mean(axis=1), 1e-12)
        g = (m * m).sum(axis=1) / (self.d * var_m)
        return {"g": g, "var_m": var_m}

    def residual_and_scale(self, fit: EBSmoothFit, n_hutch: int = 128, seed: int = 1,
                           shape: bool = True) -> Dict[str, np.ndarray]:
        """Scale and shape statistics of the residual under the fitted model.

        delta = sigma^2 v(L)^{-1} x with null covariance Sigma = sigma^4 v(L)^{-1} per feature.
        SCALE: s_i = ||delta_i||^2 / (d tau2_i), tau2_i = Sigma_ii (Hutchinson), exact chi^2_d/d null.
        SHAPE: r_i = ||(K delta)_i||^2 / ||delta_i||^2, K = kappa^2 I + L: the conditional roughness of the
        residual relative to its energy (the t = 0 ratio member of the dissipation family). Under the
        null ((K delta)_i, delta_i) is bivariate Gaussian per feature with variances [K^2 Sigma]_ii,
        Sigma_ii and covariance [K Sigma]_ii (all functions of L commute), estimated with the same
        probes; the exact ratio null is nulls.ratio_tails (F(d, d) on the 2x2 eigenvalues).
        """
        s2, a, k2 = fit.sigma2, fit.a, fit.kappa2
        Kmv = lambda Y: k2 * Y + self.L @ Y
        Y = self.solve_B(self.x, s2, a, k2)
        delta = s2 * a * Kmv(Y)
        rng = np.random.default_rng(seed)
        Z = rng.choice([-1.0, 1.0], size=(self.n, n_hutch))
        YZ = self.solve_B(Z, s2, a, k2)
        MZ = a * Kmv(YZ)                            # v(L)^{-1} Z
        diag_vinv = np.clip((Z * MZ).mean(axis=1), 1e-12, 1.0 / s2)   # [v^{-1}]_ii in (0, 1/sigma^2)
        S_ii = 1.0 - s2 * diag_vinv
        tau2 = s2 ** 2 * diag_vinv
        s = (delta * delta).sum(axis=1) / (self.d * tau2)
        out = {"delta": delta, "tau2": tau2, "S_ii": S_ii, "s": s, "template": self.x - delta}
        if shape:
            KMZ = Kmv(MZ); K2MZ = Kmv(KMZ)
            var_u = s2 ** 2 * np.maximum((Z * K2MZ).mean(axis=1), 1e-300)     # [K^2 Sigma]_ii
            cov_ud = s2 ** 2 * (Z * KMZ).mean(axis=1)                          # [K Sigma]_ii
            u = Kmv(delta)
            r = (u * u).sum(axis=1) / np.maximum((delta * delta).sum(axis=1), 1e-300)
            # keep the estimated 2x2 covariance positive semi-definite
            cov_ud = np.clip(cov_ud, -np.sqrt(var_u * tau2), np.sqrt(var_u * tau2))
            out.update({"r": r, "r_var_u": var_u, "r_var_d": tau2, "r_cov": cov_ud})
        return out
