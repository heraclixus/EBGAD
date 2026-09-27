"""
SOC Bridge Anomaly Scoring
============================
Per-node anomaly scores from a trained SOC bridge score network.

Scoring methods:

1. **magnitude** — ‖s_θ(t, x_t)‖² averaged over MC samples.
2. **spectral_ratio** — high-frequency energy fraction of model residuals.
3. **magnitude+rough** — score magnitude + graph-local roughness penalty.
4. **control_energy** — per-node SOC optimal control cost.
   Measures how much control effort each node requires to steer the
   GOU reference process from the template m to the data x_T, weighted
   by the graph precision Q_ρ.  Anomalous nodes whose features can't
   be predicted from the graph-structured template incur high cost.
5. **directional** — projects s_θ onto the direction from the data mean to
   each node; captures *which direction* the score points, not just its
   magnitude.  Better for fraud where anomalies are shifted in a consistent
   direction rather than being random outliers.
6. **projected** — projects s_θ onto a fixed anomaly direction vector
   (e.g. normal-class centroid → overall mean), combining directional
   sensitivity with magnitude.
7. **multiscale_ce** — evaluates control_energy across a grid of (rho, lambda)
   values at scoring time and aggregates.  Different regimes detect different
   anomaly types; no retraining required.
8. **multiscale_ce+stats** — multiscale_ce augmented with graph-topology
   statistics (local affinity + degree anomaly).

Reuses ``bridge.graph_scoring`` for Tier 1 post-hoc scoring.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from soc.soc_bridge import (
    soc_bridge_statistics, soc_bridge_sample, compute_Q_rho_eigenvalues,
    compute_local_affinity,
)
from bridge.graph_scoring import spectral_ratio_score, hybrid_score
from torch_geometric.utils import degree


def _compute_delta(
    x_test: torch.Tensor,
    template: torch.Tensor,
    V: torch.Tensor,
    lam_Q: torch.Tensor,
    alpha: float,
    T: float,
    prior_mean_mode: str = "zero",
) -> torch.Tensor:
    """Compute template residual delta = x - predictive_mean.

    Args:
        prior_mean_mode: "stationary" uses delta = x - m (stationary prior,
            predictive mean = template). "zero" uses delta = x - (I-Phi)m
            (zero-mean prior, propagator-dampened template).
    """
    if prior_mean_mode == "stationary":
        return x_test - template
    else:
        A_0T = torch.exp(-alpha * T * lam_Q)
        return x_test - V @ ((1.0 - A_0T).unsqueeze(-1) * (V.T @ template))


# ═══════════════════════════════════════════════════════════════════════════
# Score magnitude
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_magnitude_anomaly(
    score_net: torch.nn.Module,
    x_test: torch.Tensor,
    lam_Q: torch.Tensor,
    V: torch.Tensor,
    template: torch.Tensor,
    alpha: float,
    T: float,
    sample_prior_fn,
    lam_penalty: float = 10.0,
    K: int = 10,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Per-node anomaly scores via score magnitude under the SOC bridge."""
    score_net.eval()
    n = x_test.shape[0]
    accum = torch.zeros(n, device=x_test.device, dtype=x_test.dtype)

    eps_t = 1e-4 * T
    for _ in range(K):
        x_0 = sample_prior_fn()
        t = eps_t + (T - 2.0 * eps_t) * torch.rand(1, device=x_test.device).item()

        stats = soc_bridge_statistics(
            lam_Q, V, alpha, t, T,
            lam_penalty=lam_penalty, jitter=jitter,
        )
        x_t, _ = soc_bridge_sample(stats, x_0, x_test, template)
        pred = score_net(t, x_t)
        accum += (pred ** 2).sum(dim=-1)

    return accum / K


# ═══════════════════════════════════════════════════════════════════════════
# Reverse path energy
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def path_energy_anomaly(
    score_net: torch.nn.Module,
    x_test: torch.Tensor,
    lam_Q: torch.Tensor,
    V: torch.Tensor,
    template: torch.Tensor,
    alpha: float,
    T: float,
    sample_prior_fn,
    lam_penalty: float = 10.0,
    K: int = 20,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Reverse path energy anomaly score.

    Integrates the squared correction along the reverse probability flow ODE:

        E_θ(x_T) = E_{x_0} [ ∫_0^T  (1/2g²) ‖v_θ(x_t,t)‖² dt ]

    where v_θ = g u*_t − g² s_θ  is the total correction beyond the
    reference drift, and x_t follows the reverse ODE from x_T.

    With w = 1/(2g²), the path energy decomposes as:
        E_θ = J* (control energy) + M_θ (score mag.) − C_θ (cross term)
    and reduces to J* in the linear-Gaussian limit (Proposition 2).

    Parameters
    ----------
    K : number of time steps for ODE discretization.
    """
    import math

    score_net.eval()
    n, d = x_test.shape
    device = x_test.device
    lam_safe = lam_Q.clamp(min=jitter)

    eps_t = 1e-4 * T
    # Uniform time grid from T → 0 (reverse direction)
    ts = torch.linspace(T - eps_t, eps_t, K, device=device)
    dt = (T - 2 * eps_t) / K

    n_mc = 3  # Monte Carlo samples over x_0
    accum = torch.zeros(n, device=device, dtype=x_test.dtype)

    for _ in range(n_mc):
        x_0 = sample_prior_fn()
        x_t = x_test.clone()  # start from x_T

        energy = torch.zeros(n, device=device, dtype=x_test.dtype)

        for k in range(K):
            t_k = ts[k].item()
            g_k = math.sqrt(2.0 * alpha)  # constant schedule: g = sqrt(2α)

            # --- u*_t(x_t) in eigenspace (Theorem 1) ---
            A_tT = torch.exp(-alpha * (T - t_k) * lam_Q)            # Φ_{t:T}
            Sig_tT = (1.0 - torch.exp(-2.0 * alpha * (T - t_k) * lam_safe)) / lam_safe
            D_tl_inv = 1.0 / (1.0 / lam_penalty + Sig_tT).clamp(min=jitter)

            # target residual: x_T − Φ_{t:T} x_t − (I − Φ_{t:T}) m
            xt_hat = V.T @ x_t        # (k_eig, d)
            xT_hat = V.T @ x_test
            m_hat  = V.T @ template
            residual_hat = xT_hat - A_tT.unsqueeze(-1) * xt_hat - (1.0 - A_tT).unsqueeze(-1) * m_hat

            # u*_t = g Φ_{t:T} D_{t,λ}⁻¹ residual   (in eigenspace)
            u_star_hat = g_k * A_tT.unsqueeze(-1) * D_tl_inv.unsqueeze(-1) * residual_hat
            u_star = V @ u_star_hat  # (n, d)

            # --- s_θ(x_t, t) ---
            s_theta = score_net(t_k, x_t)  # (n, d)

            # --- v_θ = g u* − g² s_θ ---
            v_theta = g_k * u_star - g_k ** 2 * s_theta

            # --- accumulate w(t) ‖v_θ‖² dt,  w = 1/(2g²) ---
            energy += (0.5 / g_k ** 2) * (v_theta ** 2).sum(dim=-1) * dt

            # --- Euler step: reverse ODE ---
            f_ref = -alpha * (V @ (lam_Q.unsqueeze(-1) * (V.T @ (x_t - template))))
            drift = f_ref + g_k * u_star - g_k ** 2 * s_theta
            x_t = x_t - drift * dt  # negative dt because we go backward

        accum += energy

    return accum / n_mc


# ═══════════════════════════════════════════════════════════════════════════
# Reverse path energy v2: J* + M_θ (two positive terms, no cross term)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def path_energy_v2_anomaly(
    score_net: torch.nn.Module,
    x_test: torch.Tensor,
    lam_Q: torch.Tensor,
    V: torch.Tensor,
    template: torch.Tensor,
    alpha: float,
    T: float,
    sample_prior_fn,
    lam_penalty: float = 10.0,
    K: int = 20,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Path energy v2: J* + M_θ (control energy + score magnitude).

    Drops the cross term from the full path energy decomposition, giving
    two strictly positive components:

        E_v2 = ∫ ½‖u*‖² dt  +  ∫ (g²/2)‖s_θ‖² dt

    This guarantees E_v2 ≥ J* always — the score network can only ADD
    signal, never destroy it. ODE trajectory is still used so that s_θ
    is evaluated along a coherent reverse path.
    """
    import math

    score_net.eval()
    n, d = x_test.shape
    device = x_test.device
    lam_safe = lam_Q.clamp(min=jitter)

    eps_t = 1e-4 * T
    ts = torch.linspace(T - eps_t, eps_t, K, device=device)
    dt = (T - 2 * eps_t) / K

    n_mc = 3
    accum = torch.zeros(n, device=device, dtype=x_test.dtype)

    for _ in range(n_mc):
        x_0 = sample_prior_fn()
        x_t = x_test.clone()

        j_star = torch.zeros(n, device=device, dtype=x_test.dtype)
        m_theta = torch.zeros(n, device=device, dtype=x_test.dtype)

        for k in range(K):
            t_k = ts[k].item()
            g_k = math.sqrt(2.0 * alpha)

            # --- u*_t in eigenspace ---
            A_tT = torch.exp(-alpha * (T - t_k) * lam_Q)
            Sig_tT = (1.0 - torch.exp(-2.0 * alpha * (T - t_k) * lam_safe)) / lam_safe
            D_tl_inv = 1.0 / (1.0 / lam_penalty + Sig_tT).clamp(min=jitter)

            xt_hat = V.T @ x_t
            xT_hat = V.T @ x_test
            m_hat  = V.T @ template
            residual_hat = xT_hat - A_tT.unsqueeze(-1) * xt_hat - (1.0 - A_tT).unsqueeze(-1) * m_hat

            u_star_hat = g_k * A_tT.unsqueeze(-1) * D_tl_inv.unsqueeze(-1) * residual_hat
            u_star = V @ u_star_hat

            # --- s_θ ---
            s_theta = score_net(t_k, x_t)

            # --- J* term: ½ ‖u*‖² ---
            j_star += 0.5 * (u_star ** 2).sum(dim=-1) * dt

            # --- M_θ term: (g²/2) ‖s_θ‖² ---
            m_theta += 0.5 * g_k ** 2 * (s_theta ** 2).sum(dim=-1) * dt

            # --- Euler step (same ODE trajectory) ---
            f_ref = -alpha * (V @ (lam_Q.unsqueeze(-1) * (V.T @ (x_t - template))))
            drift = f_ref + g_k * u_star - g_k ** 2 * s_theta
            x_t = x_t - drift * dt

        accum += j_star + m_theta

    return accum / n_mc


# ═══════════════════════════════════════════════════════════════════════════
# Directional scoring (H1)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_directional_anomaly(
    score_net: torch.nn.Module,
    x_test: torch.Tensor,
    lam_Q: torch.Tensor,
    V: torch.Tensor,
    template: torch.Tensor,
    alpha: float,
    T: float,
    sample_prior_fn,
    lam_penalty: float = 10.0,
    K: int = 10,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Per-node anomaly scores via directional score alignment.

    For each node, computes how consistently the score vector s_θ(t, x_t)
    points in the direction from the data centroid to x_test.  Fraud nodes
    with systematically shifted features will have scores that consistently
    align with the shift direction.

    Returns ``-⟨s_θ, d_i⟩`` averaged over K MC samples, where
    ``d_i = (x_i - x̄) / ‖x_i - x̄‖``.  Negative sign because the score
    points *toward* the mode — nodes far from the centroid in a consistent
    direction get large negative projections.
    """
    score_net.eval()
    n, d = x_test.shape
    accum = torch.zeros(n, device=x_test.device, dtype=x_test.dtype)

    centroid = x_test.mean(dim=0, keepdim=True)
    direction = x_test - centroid
    direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    direction = direction / direction_norm

    eps_t = 1e-4 * T
    for _ in range(K):
        x_0 = sample_prior_fn()
        t = eps_t + (T - 2.0 * eps_t) * torch.rand(1, device=x_test.device).item()

        stats = soc_bridge_statistics(
            lam_Q, V, alpha, t, T,
            lam_penalty=lam_penalty, jitter=jitter,
        )
        x_t, _ = soc_bridge_sample(stats, x_0, x_test, template)
        pred = score_net(t, x_t)
        accum += (pred * direction).sum(dim=-1)

    return -accum / K


# ═══════════════════════════════════════════════════════════════════════════
# Projected scoring (H6)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_projected_anomaly(
    score_net: torch.nn.Module,
    x_test: torch.Tensor,
    lam_Q: torch.Tensor,
    V: torch.Tensor,
    template: torch.Tensor,
    alpha: float,
    T: float,
    sample_prior_fn,
    anomaly_direction: torch.Tensor,
    lam_penalty: float = 10.0,
    K: int = 10,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Per-node anomaly scores via projection onto a fixed direction.

    Projects s_θ onto ``anomaly_direction`` (a global d-dim vector pointing
    from normal centroid toward anomaly territory).  Combines directional
    sensitivity with score magnitude along the discriminative axis.

    Returns ``-⟨s_θ, v⟩`` averaged over K MC samples.
    """
    score_net.eval()
    n = x_test.shape[0]
    accum = torch.zeros(n, device=x_test.device, dtype=x_test.dtype)

    v = anomaly_direction / anomaly_direction.norm().clamp(min=1e-8)

    eps_t = 1e-4 * T
    for _ in range(K):
        x_0 = sample_prior_fn()
        t = eps_t + (T - 2.0 * eps_t) * torch.rand(1, device=x_test.device).item()

        stats = soc_bridge_statistics(
            lam_Q, V, alpha, t, T,
            lam_penalty=lam_penalty, jitter=jitter,
        )
        x_t, _ = soc_bridge_sample(stats, x_0, x_test, template)
        pred = score_net(t, x_t)
        accum += (pred * v.unsqueeze(0)).sum(dim=-1)

    return -accum / K


# ═══════════════════════════════════════════════════════════════════════════
# Control energy scoring (analytic, no neural network)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def control_energy_anomaly(
    x_test: torch.Tensor,
    template: torch.Tensor,
    V: torch.Tensor,
    lam_Q: torch.Tensor,
    alpha: float,
    T: float,
    lam_penalty: float = 10.0,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Per-node SOC optimal control cost.

    The SOC total cost for steering the GOU reference from prior to data is

    .. math::
        J^* = \tfrac12\,\delta^\top D_{0,\lambda}^{-1}\,\delta

    where :math:`\delta = x_T - (I - A_{0:T})\mathbf m` (for large T where
    :math:`A_{0:T} \approx 0`) and :math:`D_{0,\lambda} = \lambda^{-1}I + \Sigma_{0:T}`.

    In the eigenspace of :math:`\bar Q_\rho`, per eigenvalue :math:`q`:

    .. math::
        D_{0,\lambda,q}^{-1} \approx q \quad (\lambda \to \infty)

    so high-frequency components of the template residual are weighted
    more heavily — exactly the graph-roughness signal.

    Returns per-node scores by summing over feature dimensions.
    """
    lam_safe = lam_Q.clamp(min=jitter)

    A_0T = torch.exp(-alpha * T * lam_Q)
    Sig_0T = (1.0 - torch.exp(-2.0 * alpha * T * lam_safe)) / lam_safe
    D_0l_inv = 1.0 / (1.0 / lam_penalty + Sig_0T).clamp(min=jitter)

    # Template residual in eigenspace: δ = x_T - A_{0:T}·0 - (I - A_{0:T})·m
    # With x_0 marginalized out (zero-mean prior): E[δ] = x_T - (I-A_{0:T})·m
    delta = x_test - V @ ((1.0 - A_0T).unsqueeze(-1) * (V.T @ template))

    # Weighted residual in eigenspace: D_{0,λ}^{-1/2} V^T δ
    delta_hat = V.T @ delta                           # (n, d) in eigenspace
    weighted = D_0l_inv.sqrt().unsqueeze(-1) * delta_hat  # weight per eigenvalue

    # Back to node space, per-node energy
    node_energy = (V @ weighted) ** 2                 # (n, d)
    return node_energy.sum(dim=-1)                    # (n,)


@torch.no_grad()
def precision_energy_anomaly(
    x_test: torch.Tensor,
    template: torch.Tensor,
    V: torch.Tensor,
    lam_Q: torch.Tensor,
    alpha: float,
    T: float,
    lam_penalty: float = 10.0,
    jitter: float = 1e-6,
    prior_mean_mode: str = "zero",
) -> torch.Tensor:
    r"""Per-node anomaly score using graph precision Q_rho directly.

    Computes J_i = (1/2) delta_i^T Q_rho delta_i, the Mahalanobis distance
    weighted by the graph precision. This is the Bayesian surprise under the
    GOUB predictive p(x_T|phi) = N(m, Q_rho^{-1}).
    """
    delta = _compute_delta(x_test, template, V, lam_Q, alpha, T, prior_mean_mode)

    # Spectral-projection score: score only the k computed eigenmodes.
    # Q_rho^{proj,1/2} delta = sum_j sqrt(q_j) v_j (v_j^T delta)
    # When k=n this equals Q_rho^{1/2} delta exactly.
    # When k<n, uncomputed modes are projected out (not scored).
    delta_hat = V.T @ delta                           # (k, d)
    weighted = lam_Q.sqrt().unsqueeze(-1) * delta_hat  # weight by q_j^{1/2}

    # Back to node space, per-node energy
    node_energy = (V @ weighted) ** 2                 # (n, d)
    return node_energy.sum(dim=-1)                    # (n,)


@torch.no_grad()
def precision_ratio_anomaly(
    x_test: torch.Tensor,
    template: torch.Tensor,
    V: torch.Tensor,
    lam_Q: torch.Tensor,
    alpha: float,
    T: float,
    lam_penalty: float = 10.0,
    jitter: float = 1e-6,
    prior_mean_mode: str = "zero",
) -> torch.Tensor:
    r"""Spectral control ratio using graph precision Q_rho.

    R_i = delta_i^T Q_rho delta_i / delta_i^T delta_i
    """
    ce = precision_energy_anomaly(x_test, template, V, lam_Q, alpha, T,
                                  lam_penalty, jitter, prior_mean_mode)
    delta = _compute_delta(x_test, template, V, lam_Q, alpha, T, prior_mean_mode)
    total_energy = (delta ** 2).sum(dim=-1).clamp(min=1e-12)
    return ce / total_energy


@torch.no_grad()
def ce_ratio_anomaly(
    x_test: torch.Tensor,
    template: torch.Tensor,
    V: torch.Tensor,
    lam_Q: torch.Tensor,
    alpha: float,
    T: float,
    lam_penalty: float = 10.0,
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Normalized control energy (Rayleigh quotient).

    CE_ratio_i = (δ_i^T D⁻¹ δ_i) / (δ_i^T δ_i)

    Measures the *effective spectral frequency* of each node's template
    residual.  Anomalous nodes whose features are concentrated in
    high-frequency graph components get a high ratio, independent of
    their overall feature magnitude.

    With zero template (δ = x), this is equivalent to spectral_ratio
    scoring but using the SOC precision weighting D⁻¹ instead of a
    heat-kernel cutoff.
    """
    lam_safe = lam_Q.clamp(min=jitter)

    A_0T = torch.exp(-alpha * T * lam_Q)
    Sig_0T = (1.0 - torch.exp(-2.0 * alpha * T * lam_safe)) / lam_safe
    D_0l_inv = 1.0 / (1.0 / lam_penalty + Sig_0T).clamp(min=jitter)

    delta = x_test - V @ ((1.0 - A_0T).unsqueeze(-1) * (V.T @ template))

    # Reuse control_energy_anomaly for numerator
    ce = control_energy_anomaly(x_test, template, V, lam_Q, alpha, T, lam_penalty, jitter)

    # Denominator: per-node ‖δ‖²
    total_energy = (delta ** 2).sum(dim=-1).clamp(min=1e-12)

    return ce / total_energy


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scale control energy scoring
# ═══════════════════════════════════════════════════════════════════════════

# Predefined (rho, lambda) grids for sweep convenience
CE_RHO_GRIDS = {
    "coarse": (0.0, 0.5, 1.0),
    "fine":   (0.0, 0.25, 0.5, 0.75, 1.0),
}
CE_LAM_GRIDS = {
    "coarse": (1.0, 10.0, 100.0),
    "fine":   (0.1, 1.0, 10.0, 100.0, 1000.0),
}


@torch.no_grad()
def multiscale_control_energy_anomaly(
    x_test: torch.Tensor,
    template: torch.Tensor,
    V: torch.Tensor,
    lam_L: torch.Tensor,
    alpha: float,
    T: float,
    kappa: float = 1.0,
    nu: float = 1.0,
    precision_family: str = "single_scale",
    eta: float = 0.5,
    kappa2: Optional[float] = None,
    rho_grid: tuple = (0.0, 0.5, 1.0),
    lam_grid: tuple = (1.0, 10.0, 100.0),
    aggregation: str = "max",
    jitter: float = 1e-6,
) -> torch.Tensor:
    r"""Multi-scale control energy: evaluate at a grid of (rho, lambda).

    For each (rho, lam) in the Cartesian product of ``rho_grid`` and
    ``lam_grid``, computes per-node control energy, min-max normalizes,
    then aggregates across all settings.

    Different (rho, lam) regimes detect different anomaly types:
    - rho=0, lam low  → soft feature outliers (isotropic)
    - rho=0, lam high → strict feature matching
    - rho=1, lam low  → soft graph-structural anomalies
    - rho=1, lam high → strict graph+feature matching

    Parameters
    ----------
    x_test : (n, d) normalized node features.
    template : (n, d) template signal.
    V : (n, k) or (k, k) eigenvectors of the Laplacian.
    lam_L : (k,) Laplacian eigenvalues.
    alpha : drift strength.
    T : terminal time.
    kappa, nu : Matern prior parameters.
    rho_grid : tuple of rho values to sweep.
    lam_grid : tuple of lambda values to sweep.
    aggregation : ``"max"``, ``"mean"``, ``"norm"``, or ``"rank"``.
    jitter : numerical stability.

    Returns
    -------
    scores : (n,)
    """
    n = x_test.shape[0]
    all_scores = []

    for rho in rho_grid:
        lam_Q = compute_Q_rho_eigenvalues(
            lam_L,
            rho=rho,
            kappa=kappa,
            nu=nu,
            precision_family=precision_family,
            eta=eta,
            kappa2=kappa2,
        )
        for lam_pen in lam_grid:
            s = control_energy_anomaly(
                x_test, template, V, lam_Q, alpha, T,
                lam_penalty=lam_pen, jitter=jitter,
            )
            s_min, s_max = s.min(), s.max()
            s_normed = (s - s_min) / (s_max - s_min).clamp(min=1e-12)
            all_scores.append(s_normed)

    stack = torch.stack(all_scores, dim=0)  # (K, n)

    if aggregation == "max":
        return stack.max(dim=0).values
    elif aggregation == "mean":
        return stack.mean(dim=0)
    elif aggregation == "norm":
        return stack.norm(dim=0)
    elif aggregation == "rank":
        ranks = torch.zeros_like(stack)
        for i in range(stack.shape[0]):
            order = stack[i].argsort()
            ranks[i, order] = torch.arange(n, device=stack.device, dtype=stack.dtype)
        return ranks.mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Graph-statistics-augmented scoring
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def graph_stats_augmented_score(
    base_scores: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    stats_weight: float = 0.5,
) -> torch.Tensor:
    r"""Augment base anomaly scores with graph-topology statistics.

    Blends the base score with two unsupervised graph signals:

    1. **Negative local affinity**: nodes dissimilar to their neighbors
       (low cosine similarity) are more suspicious.
    2. **Degree anomaly**: z-score of log(degree+1).  Extreme degrees
       (very high or very low) correlate with anomalous behavior.

    Final score: ``(1-w) * norm(base) + w * norm(graph_signal)``.

    Parameters
    ----------
    base_scores : (n,) any per-node anomaly scores.
    x : (n, d) node features (normalized).
    edge_index : (2, E).
    num_nodes : n.
    stats_weight : blend weight for graph statistics (0 = base only).

    Returns
    -------
    scores : (n,)
    """
    if stats_weight <= 0.0:
        return base_scores

    def _normalize(s):
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min).clamp(min=1e-12)

    def _zscore(s):
        return (s - s.mean()) / s.std().clamp(min=1e-8)

    # Local affinity: lower → more anomalous
    aff = compute_local_affinity(x, edge_index, num_nodes)
    neg_aff = -aff

    # Degree anomaly: z-score of log(degree+1), absolute value
    src, dst = edge_index[0], edge_index[1]
    deg = degree(src, num_nodes=num_nodes).float()
    deg = deg + degree(dst, num_nodes=num_nodes).float()
    deg_z = _zscore(torch.log1p(deg)).abs()

    graph_signal = _normalize(neg_aff) + _normalize(deg_z)

    return (1.0 - stats_weight) * _normalize(base_scores) \
         + stats_weight * _normalize(graph_signal)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════════

def compute_anomaly_scores(
    trainer,
    x_test: torch.Tensor,
    K: int = 10,
    method: str = "magnitude",
    hybrid_lambda: float = 1.0,
    ce_rho_grid: str = "coarse",
    ce_lam_grid: str = "coarse",
    ce_aggregation: str = "max",
    stats_weight: float = 0.0,
    anomaly_dir_mode: str = "centroid",
) -> torch.Tensor:
    """Compute anomaly scores using a trained :class:`SOCTrainer`.

    Parameters
    ----------
    trainer : a :class:`SOCTrainer` that has been trained.
    x_test : (n, d) test node features (raw, un-normalised).
    K : number of Monte Carlo samples.
    method : ``"magnitude"``, ``"spectral_ratio"``, ``"magnitude+rough"``,
        ``"control_energy"``, ``"magnitude+control"``, ``"directional"``,
        ``"projected"``, ``"magnitude+directional"``, ``"magnitude+projected"``,
        ``"magnitude+raw_directional"``, ``"raw_directional"``,
        ``"multiscale_ce"``, ``"multiscale_ce+stats"``.
    hybrid_lambda : weight for blending in hybrid methods.
    ce_rho_grid : ``"coarse"`` or ``"fine"`` (for multiscale_ce).
    ce_lam_grid : ``"coarse"`` or ``"fine"`` (for multiscale_ce).
    ce_aggregation : ``"max"``, ``"mean"``, ``"norm"``, ``"rank"``.
    stats_weight : blend weight for graph statistics (multiscale_ce+stats).
    anomaly_dir_mode : ``"centroid"`` (legacy, ≈0 after z-score),
        ``"feat_std"`` (variance-weighted raw sum — recovers trivial baseline),
        ``"ones"`` (equal-weight z-score sum).

    Returns
    -------
    scores : (n,)
    """
    x_normed = trainer.normalize(x_test.to(trainer.device).float())
    n = x_normed.shape[0]
    edge_index = trainer.edge_index

    common = dict(
        score_net=trainer.score_net,
        x_test=x_normed,
        lam_Q=trainer.lam_Q_rho,
        V=trainer.V,
        template=trainer.template,
        alpha=trainer.cfg.alpha,
        T=trainer.cfg.T,
        sample_prior_fn=trainer._sample_prior,
        lam_penalty=trainer.cfg.lam_penalty,
        jitter=trainer.cfg.jitter,
    )

    def _normalize(s):
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min).clamp(min=1e-12)

    prior_mean_mode = getattr(trainer.cfg, "prior_mean_mode", "zero")

    if method == "precision_energy":
        scores = precision_energy_anomaly(
            x_test=x_normed,
            template=trainer.template,
            V=trainer.V,
            lam_Q=trainer.lam_Q_rho,
            alpha=trainer.cfg.alpha,
            T=trainer.cfg.T,
            prior_mean_mode=prior_mean_mode,
        )

    elif method == "precision_ratio":
        scores = precision_ratio_anomaly(
            x_test=x_normed,
            template=trainer.template,
            V=trainer.V,
            lam_Q=trainer.lam_Q_rho,
            alpha=trainer.cfg.alpha,
            T=trainer.cfg.T,
            prior_mean_mode=prior_mean_mode,
        )

    elif method == "path_energy":
        scores = path_energy_anomaly(**common, K=K)

    elif method == "path_energy_v2":
        scores = path_energy_v2_anomaly(**common, K=K)

    elif method == "ce_ratio":
        scores = ce_ratio_anomaly(
            x_test=x_normed,
            template=trainer.template,
            V=trainer.V,
            lam_Q=trainer.lam_Q_rho,
            alpha=trainer.cfg.alpha,
            T=trainer.cfg.T,
            lam_penalty=trainer.cfg.lam_penalty,
            jitter=trainer.cfg.jitter,
        )

    elif method == "control_energy":
        scores = control_energy_anomaly(
            x_test=x_normed,
            template=trainer.template,
            V=trainer.V,
            lam_Q=trainer.lam_Q_rho,
            alpha=trainer.cfg.alpha,
            T=trainer.cfg.T,
            lam_penalty=trainer.cfg.lam_penalty,
            jitter=trainer.cfg.jitter,
        )

    elif method == "magnitude+control":
        mag = score_magnitude_anomaly(**common, K=K)
        ctrl = control_energy_anomaly(
            x_test=x_normed,
            template=trainer.template,
            V=trainer.V,
            lam_Q=trainer.lam_Q_rho,
            alpha=trainer.cfg.alpha,
            T=trainer.cfg.T,
            lam_penalty=trainer.cfg.lam_penalty,
            jitter=trainer.cfg.jitter,
        )
        beta = hybrid_lambda
        scores = (1.0 - beta) * _normalize(mag) + beta * _normalize(ctrl)

    elif method == "spectral_ratio":
        ge = trainer.graph_eigen
        base = score_magnitude_anomaly(**common, K=K)
        with torch.no_grad():
            scores = spectral_ratio_score(
                x_normed, ge, tau=trainer.cfg.filter_tau,
            )

    elif method == "magnitude+rough":
        base = score_magnitude_anomaly(**common, K=K)
        scores = hybrid_score(base, x_normed, edge_index, n, lam=hybrid_lambda)

    elif method == "directional":
        scores = score_directional_anomaly(**common, K=K)

    elif method == "projected":
        anomaly_dir = _compute_anomaly_direction(trainer, mode=anomaly_dir_mode)
        scores = score_projected_anomaly(
            **common, anomaly_direction=anomaly_dir, K=K,
        )

    elif method == "magnitude+directional":
        mag = score_magnitude_anomaly(**common, K=K)
        direc = score_directional_anomaly(**common, K=K)
        beta = hybrid_lambda
        scores = (1.0 - beta) * _normalize(mag) + beta * _normalize(direc)

    elif method == "magnitude+projected":
        anomaly_dir = _compute_anomaly_direction(trainer, mode=anomaly_dir_mode)
        mag = score_magnitude_anomaly(**common, K=K)
        proj = score_projected_anomaly(
            **common, anomaly_direction=anomaly_dir, K=K,
        )
        beta = hybrid_lambda
        scores = (1.0 - beta) * _normalize(mag) + beta * _normalize(proj)

    elif method == "magnitude+raw_directional":
        mag = score_magnitude_anomaly(**common, K=K)
        anomaly_dir = _compute_anomaly_direction(trainer, mode=anomaly_dir_mode)
        v = anomaly_dir / anomaly_dir.norm().clamp(min=1e-8)
        raw_dir = -(x_normed @ v)
        beta = hybrid_lambda
        scores = (1.0 - beta) * _normalize(mag) + beta * _normalize(raw_dir)

    elif method == "raw_directional":
        anomaly_dir = _compute_anomaly_direction(trainer, mode=anomaly_dir_mode)
        v = anomaly_dir / anomaly_dir.norm().clamp(min=1e-8)
        scores = -(x_normed @ v)

    elif method in ("multiscale_ce", "multiscale_ce+stats"):
        rho_vals = CE_RHO_GRIDS.get(ce_rho_grid, CE_RHO_GRIDS["coarse"])
        lam_vals = CE_LAM_GRIDS.get(ce_lam_grid, CE_LAM_GRIDS["coarse"])
        lam_L = getattr(trainer, "lam_L_model", trainer.graph_eigen.lam_L)
        scores = multiscale_control_energy_anomaly(
            x_test=x_normed,
            template=trainer.template,
            V=trainer.V,
            lam_L=lam_L,
            alpha=trainer.cfg.alpha,
            T=trainer.cfg.T,
            kappa=trainer.cfg.kappa,
            nu=trainer.cfg.nu,
            precision_family=getattr(trainer.cfg, "precision_family", "single_scale"),
            eta=getattr(trainer.cfg, "eta", 0.5),
            kappa2=getattr(trainer.cfg, "kappa2", None),
            rho_grid=rho_vals,
            lam_grid=lam_vals,
            aggregation=ce_aggregation,
            jitter=trainer.cfg.jitter,
        )
        if method == "multiscale_ce+stats":
            scores = graph_stats_augmented_score(
                scores, x_normed, edge_index, n,
                stats_weight=stats_weight,
            )

    else:  # "magnitude"
        scores = score_magnitude_anomaly(**common, K=K)

    return torch.nan_to_num(scores, nan=0.0)


def _compute_anomaly_direction(trainer, mode: str = "centroid") -> torch.Tensor:
    """Compute anomaly direction in normalised feature space (unsupervised).

    Three modes:

    ``"centroid"`` (legacy)
        ``-mean(x_normed)``.  After z-score normalization this is ≈ 0,
        making projected/raw_directional scoring ineffective.

    ``"feat_std"``
        ``feat_std`` vector (per-feature standard deviations from the raw data).
        In normalized space, ``-(x_normed @ feat_std) ∝ -sum(x_raw)``, so this
        *exactly* recovers the variance-weighted raw feature sum.  Unsupervised
        and the theoretically correct direction when anomalies are a location
        shift in raw feature space (e.g. DGraph fraud).

    ``"ones"``
        ``(1, 1, …, 1)`` — equal-weight z-score sum.  Equivalent to
        ``-sum((x - mean) / std)``.
    """
    if mode == "feat_std":
        return trainer.feat_std.squeeze()
    elif mode == "ones":
        d = trainer.x_T.shape[1]
        return torch.ones(d, device=trainer.device, dtype=trainer.x_T.dtype)
    else:
        return -trainer.x_T.mean(dim=0)
