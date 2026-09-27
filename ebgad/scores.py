"""Transport bank statistics and the conditional GMRF residual.

Conventions (rank-equivalent to the paper's, which carry a factor 1/2):
    energy_i(Gamma, lam_c) = || sum_j sqrt(c_j) V_ij xi_j ||^2
    ratio_i               = energy_i / ||delta_i||^2
    J* = energy at (inf, inf),  R = ratio at (inf, inf)
    P_i = ||(Q delta)_i||^2 / Q_ii          (conditional residual, full Q)

``start="template"`` (recommended): the GOU starts at the
template, the residual is delta = x - m at every horizon and the horizon only
reweights modes. ``start="origin"`` is the paper's X_0 = 0 convention:
delta_Gamma = delta + V diag(e^{-Gamma q}) V^T m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PAPER_HORIZONS: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf]
PAPER_LAMS: List[float] = [0.1, 1.0, 10.0, 100.0, np.inf]


def fmt(v: float) -> str:
    return "inf" if np.isinf(v) else ("%g" % v)


def bank_weights(q: np.ndarray, Gamma: float, lam_c: float) -> np.ndarray:
    """c_j(Gamma, lam_c) = (1/lam_c + (1 - e^{-2 Gamma q_j}) / q_j)^{-1}."""
    q_safe = np.maximum(q, 1e-12)
    if np.isinf(Gamma):
        sig = 1.0 / q_safe
    else:
        sig = -np.expm1(-2.0 * float(Gamma) * q_safe) / q_safe
    if np.isinf(lam_c):
        return 1.0 / np.maximum(sig, 1e-300)
    return 1.0 / (1.0 / float(lam_c) + sig)


def bank_configs(horizons: Sequence[float] = PAPER_HORIZONS,
                 lams: Sequence[float] = PAPER_LAMS) -> List[Tuple[float, float]]:
    cfgs = [(np.inf, np.inf)]
    for G in horizons:
        for L in lams:
            if np.isinf(G) and np.isinf(L):
                continue
            cfgs.append((float(G), float(L)))
    return cfgs


@dataclass
class BankMember:
    name: str
    kind: str                 # "energy" | "ratio" | "cond"
    Gamma: float
    lam_c: float
    score: np.ndarray         # (n,)
    # null parameters (see nulls.py)
    sigma2: Optional[np.ndarray] = None   # var of each coordinate of u_i
    tau2: Optional[np.ndarray] = None     # var of each coordinate of the denominator vector
    cov: Optional[np.ndarray] = None      # cov(u, denominator) per coordinate
    mu2: Optional[np.ndarray] = None      # ||mean(u_i)||^2 (origin start only)

    def null_mean(self, d: int) -> np.ndarray:
        """Per-node expectation of the statistic under the fitted prior.

        energy: d sigma2_i (+ mu2_i); ratio: sigma2_i / tau2_i (ratio of
        means, exact as d grows); cond: d.
        """
        if self.kind == "energy":
            m = float(d) * self.sigma2
            return m + self.mu2 if self.mu2 is not None else m
        if self.kind == "ratio":
            return self.sigma2 / np.maximum(self.tau2, 1e-300)
        return np.full(self.score.shape[0], float(d))


def effective_coefficients(prep, q: np.ndarray, Gamma: float, start: str):
    """Mode coefficients Xi_eff (k, d) and the residual field the ratio divides by."""
    if start == "template" or np.isinf(Gamma):
        return prep.Xi, prep.delta, None
    shift = np.exp(-float(Gamma) * np.maximum(q, 1e-12))[:, None] * prep.M_hat  # (k, d)
    Xi_eff = prep.Xi + shift
    delta_eff = prep.delta + prep.V @ shift
    return Xi_eff, delta_eff, shift


def rough_variance(prep) -> float:
    """Isotropic variance of the unretained residual component."""
    n_perp = max(prep.n - prep.k, 1)
    return float(prep.delta_perp_sq.sum() / (prep.d * n_perp))


@dataclass
class WeightSpec:
    """One quadratic statistic: mode weights c_j, with energy and ratio names."""
    ename: str
    rname: str
    c: np.ndarray
    Gamma: float = np.inf      # only meaningful for the (Gamma, lam_c) family
    lam_c: float = np.inf


def bank_specs(q: np.ndarray, configs: Sequence[Tuple[float, float]]) -> List[WeightSpec]:
    """The paper's (Gamma, lam_c) transport bank as weight specs (J*, R at inf, inf)."""
    specs = []
    for (G, Lc) in configs:
        tag = "%s,%s" % (fmt(G), fmt(Lc))
        eq = np.isinf(G) and np.isinf(Lc)
        specs.append(WeightSpec("J*" if eq else "C@%s" % tag, "R" if eq else "CR@%s" % tag,
                                bank_weights(q, G, Lc), G, Lc))
    return specs


PROFILE_FRACTIONS: List[float] = [0.5, 0.2, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-5]


def profile_specs(q: np.ndarray, fractions: Sequence[float] = PROFILE_FRACTIONS) -> List[WeightSpec]:
    """Dissipation-rate profile of the GOU relaxation.

    Along the mean relaxation from the observation toward the template,
    delta_t = e^{-tQ} delta, the potential energy E(t) = 1/2 sum_j q_j
    e^{-2 t q_j} xi_j^2 decays at rate -dE/dt = sum_j q_j^2 e^{-2 t q_j}
    xi_j^2. Its nodewise decomposition is the statistic with mode weights
        c_j(t) = q_j^2 e^{-2 t q_j}.
    t = 0 gives ||(Q delta)_i||^2, the numerator of the conditional residual
    P; t -> inf concentrates on the slowest (smoothest) modes; the global
    time integral is the transport energy J*. This is the GOU's native
    low-pass family: the (Gamma, lam_c) bank only reweights toward rough
    modes, so a signal in smooth modes (Weibo) is out of its reach.

    Times are set from the fitted rates, not a fixed grid: t(r) is the time
    at which the roughest mode's weight has fallen to fraction r of the
    smoothest mode's, e^{-2 t (q_max - q_min)} = r. Weights are normalized
    by e^{-2 t q_min} (common to all modes) to avoid underflow.
    """
    q_safe = np.maximum(q, 1e-12)
    spread = float(q_safe.max() - q_safe.min())
    specs = [WeightSpec("D@0", "DR@0", q_safe ** 2)]
    if spread < 1e-9:
        return specs
    for r in fractions:
        t = -np.log(r) / (2.0 * spread)
        c = q_safe ** 2 * np.exp(-2.0 * t * (q_safe - q_safe.min()))
        specs.append(WeightSpec("D@%g" % r, "DR@%g" % r, c))
    return specs


def compute_members(
    prep, q: np.ndarray,
    specs: Sequence[WeightSpec],
    start: str = "template",
    rough: str = "isotropic",
    kinds: Sequence[str] = ("energy", "ratio"),
) -> List[BankMember]:
    """Energy and ratio statistics for each weight spec, with per-node null parameters."""
    V = prep.V
    V2 = V * V
    q_safe = np.maximum(q, 1e-12)
    inv_q = 1.0 / q_safe
    tau2_ret = V2 @ inv_q                       # var of retained part of delta_i coords
    rv = rough_variance(prep) if rough == "isotropic" else 0.0
    tau2 = tau2_ret + rv * (1.0 - prep.lev)     # + isotropic unretained part
    members: List[BankMember] = []
    for spec in specs:
        c = spec.c
        Xi_eff, delta_eff, shift = effective_coefficients(prep, q, spec.Gamma, start)
        u = V @ (np.sqrt(c)[:, None] * Xi_eff)  # (n, d)
        energy = (u * u).sum(axis=1)
        sigma2 = V2 @ (c * inv_q)
        cov = V2 @ (np.sqrt(c) * inv_q)
        mu2 = None
        if shift is not None:
            mu = V @ (np.sqrt(c)[:, None] * shift)
            mu2 = (mu * mu).sum(axis=1)
        if "energy" in kinds:
            members.append(BankMember(spec.ename, "energy", spec.Gamma, spec.lam_c, energy,
                                      sigma2=sigma2, mu2=mu2))
        if "ratio" in kinds:
            if rough == "drop":
                den = ((V @ Xi_eff) ** 2).sum(axis=1)
            else:
                den = (delta_eff * delta_eff).sum(axis=1)
            den = np.maximum(den, 1e-12)
            members.append(BankMember(spec.rname, "ratio", spec.Gamma, spec.lam_c, energy / den,
                                      sigma2=sigma2, tau2=tau2, cov=cov, mu2=mu2))
    return members


def compute_bank(prep, q, configs, start="template", rough="isotropic", kinds=("energy", "ratio")):
    """The paper's (Gamma, lam_c) bank (kept for the ablation)."""
    return compute_members(prep, q, bank_specs(q, configs), start=start, rough=rough, kinds=kinds)


def node_tau2(prep, q: np.ndarray, rough: str = "isotropic", block: int = 200_000) -> np.ndarray:
    """Base-model variance of a coordinate of delta_i (retained modes plus the isotropic unretained part).

    Blockwise over nodes so that V*V is never materialized (DGraph: V is 3.8 GB)."""
    inv_q = 1.0 / np.maximum(q, 1e-12)
    tau2 = np.empty(prep.n)
    for s in range(0, prep.n, block):
        Vb = prep.V[s:s + block]
        tau2[s:s + block] = (Vb * Vb) @ inv_q
    if rough == "isotropic":
        tau2 = tau2 + rough_variance(prep) * (1.0 - prep.lev)
    return tau2


def normalized_conditional_residual(prep, rho: float, kappa: float, tau2: np.ndarray) -> BankMember:
    """NP_i = ||(Q delta)_i||^2 / (Q_ii ||delta_i||^2): the t = 0 member of the ratio
    profile computed exactly on the full sparse graph (no truncation).

    On truncated spectra (Elliptic k = 300 of 203k, DGraph k = 128 of 1.2M)
    the eigenbasis members cannot represent neighborhood disagreement; this
    member can, at O(|E| d). Null per coordinate: var(u) = 1 (u = (Q delta)_i
    / sqrt(Q_ii)), var(delta_i) = (Q^{-1})_ii, approximated by tau2, and
    cov(u, delta_i) = 1 / sqrt(Q_ii) since Cov(Q delta, delta) = I.
    """
    delta_t = torch.from_numpy(prep.delta).float()
    L = prep.L_sparse.coalesce().float()
    Q_delta = rho * (kappa ** 2 * delta_t + torch.sparse.mm(L, delta_t)) + (1.0 - rho) * delta_t
    Q_ii = rho * (kappa ** 2 + prep.L_diag) + (1.0 - rho)
    num = (Q_delta.double() ** 2).sum(dim=1).numpy() / np.maximum(Q_ii, 1e-12)
    den = np.maximum((prep.delta ** 2).sum(axis=1), 1e-12)
    return BankMember("NP", "ratio", np.inf, np.inf, num / den,
                      sigma2=np.ones(prep.n), tau2=np.maximum(tau2, 1.0 / np.maximum(Q_ii, 1e-12)),
                      cov=1.0 / np.sqrt(np.maximum(Q_ii, 1e-12)))


def conditional_residual(prep, rho: float, kappa: float) -> BankMember:
    """P_i = ||(Q delta)_i||^2 / Q_ii with the full sparse Q_rho (nu = 1).

    Under the full-spectrum fitted model (Q delta)_i ~ N(0, Q_ii I_D), so
    P_i ~ chi^2_D exactly. Cost O(|E| d); no eigendecomposition.
    """
    delta_t = torch.from_numpy(prep.delta).float()
    L = prep.L_sparse.coalesce().float()
    L_delta = torch.sparse.mm(L, delta_t)
    Q_delta = rho * (kappa ** 2 * delta_t + L_delta) + (1.0 - rho) * delta_t
    Q_ii = rho * (kappa ** 2 + prep.L_diag) + (1.0 - rho)
    P = (Q_delta.double() ** 2).sum(dim=1).numpy() / np.maximum(Q_ii, 1e-12)
    return BankMember("P", "cond", np.inf, np.inf, P, sigma2=np.ones(prep.n))
