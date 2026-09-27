"""Label-free selection: empirical null + upper-tail non-null mass.

For each candidate statistic s_i with per-node null mean m_i under the
fitted prior (scores.BankMember.null_mean) and exact tail probabilities:

  1. Selection scale: w_i = log(s_i / m_i), the leverage-standardized log
     statistic. Under the model w is log chi^2_D (nearly Gaussian) up to a
     constant; under per-node scale heterogeneity it is that plus log s_i,
     still unimodal. This is the scale on which Efron's empirical-null
     assumptions hold. (The probit of the plug-in p-value does not: on
     Weibo the bulk sits at z = -15 where the probit compresses shape.)
  2. Empirical null N(delta0, sigma0^2) and null proportion pi0 by Efron's
     central matching: a quadratic fit to log f(w) on the central 50
     percent of the w's (Efron 2004, JASA; the locfdr default).
  3. pi1_upper = mass of f(w) above pi0 f0(w) for w > delta0: estimated
     non-null mass in the anomalous direction. One-sided.
  4. Select argmax pi1_upper. Node score = 1 - local fdr = 1 - pi0 f0 / f.

Calibration diagnostic: z_i = Phi^{-1}(1 - p_i) from the exact tails; its
bulk center and scale (delta0_z, sigma0_z) say how far the plug-in null is
from describing the bulk (0 and 1 when the model is right).

Densities are binned (histogram + Gaussian smoothing), O(n).

Why not Storey at lambda = 0.5 with a median-centered null: with anomalies
in the right tail exactly half the nodes sit below the median, so the
estimator returns pi0 = 1 identically (observed on Weibo, 2026-09-17).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

P_MIN = 1e-300

# Numerical-degeneracy guard (implementation default, not a selection cutoff):
# a candidate whose leverage-standardized log statistic varies by less than
# this interquartile range across nodes is a constant up to numerical noise
# (e.g. ratio members whose endpoint tolerance dominates every mode weight,
# so the ratio collapses to a constant times the projection share). Their
# density is a spike at histogram resolution and the central-matching fit
# on it is meaningless (Reddit, 2026-09-17: fake pi1 = 0.49 at AUROC 51.8).
MIN_LOG_IQR = 0.01


def z_from_tails(sf: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    sf = np.clip(np.asarray(sf, dtype=np.float64), P_MIN, 1.0)
    cdf = np.clip(np.asarray(cdf, dtype=np.float64), P_MIN, 1.0)
    return np.where(sf < 0.5, norm.isf(sf), norm.ppf(cdf))


def binned_density(w: np.ndarray, n_grid: int = 1024, bw: Optional[float] = None,
                   lower_winsor: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    """Histogram density smoothed with a Gaussian kernel (Silverman bandwidth).

    The left tail is winsorized at the ``lower_winsor`` percentile: nodes
    with (numerically) zero statistic would otherwise stretch the grid and
    swamp the bulk, and the left tail is not where anomalies live.
    """
    wf = w[np.isfinite(w)]
    lo_clip = np.percentile(wf, lower_winsor)
    wf = np.maximum(wf, lo_clip)
    n = wf.size
    sd = float(wf.std())
    iqr = float(np.subtract(*np.percentile(wf, [75, 25])))
    if bw is None:
        spread = min(sd, iqr / 1.34) if iqr > 0 else sd
        bw = 0.9 * max(spread, 1e-6) * n ** (-0.2)
    lo, hi = float(wf.min()) - 3 * bw, float(wf.max()) + 3 * bw
    grid = np.linspace(lo, hi, n_grid)
    step = grid[1] - grid[0]
    counts, _ = np.histogram(wf, bins=n_grid, range=(lo, hi))
    dens = counts / (n * step)
    half = int(np.ceil(4 * bw / step))
    kern = norm.pdf(np.arange(-half, half + 1) * step, 0.0, bw) * step
    dens = np.convolve(dens, kern, mode="same")
    dens = dens / max(np.trapezoid(dens, grid), 1e-300)
    return grid, dens, bw


def robust_center_scale(w: np.ndarray) -> Tuple[float, float]:
    wf = w[np.isfinite(w)]
    d0 = float(np.median(wf))
    s0 = max(float(1.4826 * np.median(np.abs(wf - d0))), 1e-3)
    return d0, s0


def efron_central_matching(w: np.ndarray, grid: np.ndarray, dens: np.ndarray,
                           pct: float = 0.25) -> Tuple[float, float, float]:
    """(delta0, sigma0, pi0) from a quadratic fit to log f on the central w's."""
    wf = w[np.isfinite(w)]
    lo, hi = np.percentile(wf, [100 * pct, 100 * (1 - pct)])
    sel = (grid >= lo) & (grid <= hi) & (dens > 0)
    d0_fb, s0_fb = robust_center_scale(w)
    if sel.sum() < 5:
        return d0_fb, s0_fb, 1.0
    coef = np.polyfit(grid[sel], np.log(dens[sel]), 2, w=np.sqrt(dens[sel]))
    c, b, a = coef
    if not np.isfinite(c) or c >= 0:
        return d0_fb, s0_fb, 1.0
    sigma0 = float(np.sqrt(-1.0 / (2.0 * c)))
    delta0 = float(-b / (2.0 * c))
    log_pi0 = a + np.log(sigma0 * np.sqrt(2 * np.pi)) + delta0 ** 2 / (2 * sigma0 ** 2)
    pi0 = float(np.clip(np.exp(log_pi0), 1e-6, 1.0))
    return delta0, max(sigma0, 1e-3), pi0


def upper_excess_mass(grid: np.ndarray, dens: np.ndarray, delta0: float, sigma0: float, pi0: float) -> float:
    f0 = pi0 * norm.pdf(grid, delta0, sigma0)
    excess = np.maximum(dens - f0, 0.0)
    excess[grid <= delta0] = 0.0
    return float(np.trapezoid(excess, grid))


def mirror_null(grid: np.ndarray, dens: np.ndarray) -> Tuple[float, np.ndarray]:
    """Symmetric empirical null: f0(w) = f(2 m - w) reflected about the mode m.

    Assumption-light alternative to the Gaussian central matching: null
    nodes are taken to be symmetric about the mode on the log scale and
    anomalies one-sided (right). Heavy-tailed but symmetric bulks are
    handled exactly; a left-skewed bulk (log chi-squared at small d) makes
    the estimate conservative. Returns (mode, f0 on the grid).
    """
    m = float(grid[int(np.argmax(dens))])
    f0 = np.interp(2.0 * m - grid, grid, dens, left=0.0, right=0.0)
    return m, f0


def mirror_excess_mass(grid: np.ndarray, dens: np.ndarray) -> Tuple[float, float]:
    m, f0 = mirror_null(grid, dens)
    excess = np.maximum(dens - f0, 0.0)
    excess[grid <= m] = 0.0
    return float(np.trapezoid(excess, grid)), m


@dataclass
class Candidate:
    name: str
    stat: np.ndarray                     # raw statistic (large = anomalous)
    null_mean: np.ndarray                # per-node null expectation
    sf: Optional[np.ndarray] = None      # exact upper-tail probabilities
    cdf: Optional[np.ndarray] = None     # exact lower-tail probabilities
    w: np.ndarray = field(init=False)
    delta0: float = field(init=False)
    sigma0: float = field(init=False)
    pi0: float = field(init=False)
    pi1_gauss: float = field(init=False) # upper excess over Efron's Gaussian null (diagnostic)
    mode: float = field(init=False)
    pi1: float = field(init=False)       # upper excess over the mirrored null (selection key)
    delta0_z: float = field(init=False)  # calibration diagnostics on the probit scale
    sigma0_z: float = field(init=False)
    degenerate: bool = field(init=False)
    _grid: np.ndarray = field(init=False, repr=False)
    _dens: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        s = np.asarray(self.stat, dtype=np.float64)
        m = np.maximum(np.asarray(self.null_mean, dtype=np.float64), 1e-300)
        self.w = np.log(np.maximum(s, 1e-300) / m)
        wf = self.w[np.isfinite(self.w)]
        iqr = float(np.subtract(*np.percentile(wf, [75, 25]))) if wf.size else 0.0
        self.degenerate = bool(wf.size < 10 or iqr < MIN_LOG_IQR)
        self.delta0_z, self.sigma0_z = (float("nan"), float("nan"))
        if self.sf is not None and self.cdf is not None and not self.degenerate:
            z = z_from_tails(self.sf, self.cdf)
            self.delta0_z, self.sigma0_z = robust_center_scale(z)
        if self.degenerate:
            self.delta0, self.sigma0, self.pi0, self.pi1_gauss = 0.0, 1.0, 1.0, 0.0
            self.mode, self.pi1 = 0.0, 0.0
            self._grid, self._dens = np.zeros(2), np.zeros(2)
            return
        self._grid, self._dens, _ = binned_density(self.w)
        self.delta0, self.sigma0, self.pi0 = efron_central_matching(self.w, self._grid, self._dens)
        self.pi1_gauss = upper_excess_mass(self._grid, self._dens, self.delta0, self.sigma0, self.pi0)
        self.pi1, self.mode = mirror_excess_mass(self._grid, self._dens)

    def local_fdr(self) -> np.ndarray:
        """Mirrored-null local fdr: f(2m - w) / f(w) for w above the mode, 1 below."""
        if self.degenerate:
            return np.ones_like(self.w)
        wf = np.nan_to_num(self.w, nan=self.mode, posinf=self._grid[-1], neginf=self._grid[0])
        f = np.interp(wf, self._grid, self._dens)
        f0 = np.interp(2.0 * self.mode - wf, self._grid, self._dens, left=0.0, right=0.0)
        fdr = np.clip(f0 / np.maximum(f, 1e-300), 0.0, 1.0)
        fdr[wf <= self.mode] = 1.0
        return fdr

    def fdr_score(self) -> np.ndarray:
        return 1.0 - self.local_fdr()

    def summary(self) -> Dict[str, float]:
        return {"pi1": self.pi1, "mode": self.mode, "pi1_gauss": self.pi1_gauss, "pi0": self.pi0,
                "delta0": self.delta0, "sigma0": self.sigma0,
                "delta0_z": self.delta0_z, "sigma0_z": self.sigma0_z, "degenerate": self.degenerate}


def scan_score(cands: Dict[str, "Candidate"], names: Optional[Sequence[str]] = None,
               top_k: int = 1) -> np.ndarray:
    """Selection-free node score: max over profile members of the bulk-standardized log statistic.

    Each member's w = log(stat / null mean) is standardized by its own bulk
    (median, MAD), so members are comparable; a node is then scored by its
    most surprising member (top_k = 1, the scan statistic) or by the mean
    of its top_k members. A sparse signal (Weibo: 20 smooth modes among
    8,000) is diluted in the total energy J* but stands out in one member,
    which is why the scan exists in multi-resolution testing.
    """
    use = [c for n, c in cands.items() if not c.degenerate and (names is None or n in names)]
    if not use:
        return None
    Z = []
    for c in use:
        d0, s0 = robust_center_scale(c.w)
        Z.append((c.w - d0) / s0)
    Z = np.nan_to_num(np.vstack(Z), nan=0.0, posinf=0.0, neginf=0.0)
    if top_k <= 1:
        return Z.max(axis=0)
    k = min(top_k, Z.shape[0])
    return np.sort(Z, axis=0)[-k:].mean(axis=0)


def select(pool: Dict[str, dict]) -> tuple:
    """Return (selected name, {name: Candidate}).

    ``pool`` maps a statistic name to a dict with keys stat, null_mean and
    optionally sf, cdf.
    """
    cands = {name: Candidate(name, **val) for name, val in pool.items()}
    valid = [c for c in cands.values() if not c.degenerate]
    if not valid:
        return None, cands
    best = max(valid, key=lambda c: c.pi1)
    return best.name, cands
