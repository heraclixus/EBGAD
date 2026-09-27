"""Smoke tests for backward-compatible EB design options."""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import torch

from soc.prior_optimizer import (
    Q_rho_eigenvalues as np_q_rho,
    apply_modeled_subspace as np_apply_modeled_subspace,
    coordinate_descent_stationary,
)
from soc.soc_bridge import (
    compute_Q_rho_eigenvalues as torch_q_rho,
    apply_modeled_subspace as torch_apply_modeled_subspace,
)


def test_precision_family_backward_compatibility() -> None:
    lam = np.array([0.0, 0.2, 1.0], dtype=float)
    q_legacy = np_q_rho(lam, 0.7, 1.5)
    q_two_scale = np_q_rho(
        lam,
        0.7,
        1.5,
        precision_family="two_scale",
        eta=1.0,
        kappa2=3.0,
    )
    assert np.allclose(q_legacy, q_two_scale)

    lam_t = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float32)
    q_legacy_t = torch_q_rho(lam_t, 0.7, 1.5)
    q_two_scale_t = torch_q_rho(
        lam_t,
        0.7,
        1.5,
        precision_family="two_scale",
        eta=1.0,
        kappa2=3.0,
    )
    assert torch.allclose(q_legacy_t, q_two_scale_t)


def test_modeled_subspace_projection() -> None:
    S = np.array([1.0, 2.0, 3.0], dtype=float)
    lam = np.array([0.0, 0.2, 1.0], dtype=float)
    S_eff, lam_eff, mask = np_apply_modeled_subspace(
        S, lam, mode="drop_nullspace", tol=1e-8,
    )
    assert mask.tolist() == [False, True, True]
    assert S_eff.shape == (2,)
    assert lam_eff.shape == (2,)

    lam_t = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float32)
    V_t = torch.eye(3)
    lam_eff_t, V_eff_t, mask_t = torch_apply_modeled_subspace(
        lam_t, V_t, mode="drop_nullspace", tol=1e-8,
    )
    assert mask_t.tolist() == [False, True, True]
    assert tuple(lam_eff_t.shape) == (2,)
    assert tuple(V_eff_t.shape) == (3, 2)


def test_two_scale_coordinate_descent_smoke() -> None:
    S = np.array([1.0, 2.0, 3.0], dtype=float)
    lam = np.array([0.0, 0.2, 1.0], dtype=float)
    opt = coordinate_descent_stationary(
        S,
        lam,
        n=5,
        d=2,
        precision_family="two_scale",
        eta_init=0.6,
        kappa2_init=4.0,
        optimize_eta=True,
        optimize_kappa2=True,
        modeled_subspace="drop_nullspace",
        n_iters=2,
    )
    assert opt["precision_family"] == "two_scale"
    assert opt["modeled_subspace"] == "drop_nullspace"
    assert 0.0 <= opt["rho"] <= 1.0
    assert 0.0 <= opt["eta"] <= 1.0
    assert opt["kappa"] > 0.0
    assert opt["kappa2"] > 0.0
    assert np.isfinite(opt["marginal_likelihood"])


if __name__ == "__main__":
    test_precision_family_backward_compatibility()
    test_modeled_subspace_projection()
    test_two_scale_coordinate_descent_smoke()
    print("All EB design-option smoke tests passed.")
