"""Template bandwidth: what each label-free criterion selects, and what the selection is worth.

One eigendecomposition per dataset. For every candidate bandwidth gamma the template, the residual,
the EB fit of (rho, kappa) and the two equilibrium scores are computed, together with

  ml        residual-space profile likelihood (Algorithm 1 as submitted, no Jacobian)
  ml_jac    ml + D * sum_j log|t_j|, t_j = 1 - 1/(gamma^2 + lam_j): the likelihood of the features
  annih     share of the feature energy that the residual map removes, 1 - sum_j S_j / sum_j E_j
  gamma0    the graph-statistic center of the grid, (0.5 + h) / sqrt(max(d_e, 10) / 10)

AUROC values are diagnostics of the criteria; no criterion reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
from data_utils import load_data  # noqa: E402
from soc.prior_optimizer import compute_spectral_energy  # noqa: E402
from soc.soc_bridge import compute_template  # noqa: E402
from soc.soc_trainer import SOCTrainer  # noqa: E402
from test_dynamic_score_bank import (  # noqa: E402
    build_trainer_config, compute_graph_stats, density_adjusted_template_gammas, fit_best_entry,
    horizon_score_bank, q_for_fit, safe_auc,
)
from test_multiscale_gou_poe import TEMPLATE_GAMMAS, ks_null_deviation  # noqa: E402

KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
NAMES = {"yelpchi": "YelpChi", "facebook": "Facebook"}


def truncation(n):
    return None if n < 20_000 else (128 if n >= 1_000_000 else 200 if n >= 500_000 else 300 if n >= 100_000 else 500)


def gamma_center(h, de):
    return float(np.clip((0.5 + h) / np.sqrt(max(de, 10.0) / 10.0), 0.05, 10.0))


def block_spectrum(data):
    """Exact full spectrum of the symmetric normalized Laplacian, one dense block per connected component."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = data.x.shape[0]
    ei = data.edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    L = sp.identity(n, format="csr") - sp.diags(dinv) @ A @ sp.diags(dinv)
    ncomp, lab = connected_components(A, directed=False)
    order = np.argsort(lab, kind="stable")
    sizes = np.bincount(lab, minlength=ncomp)
    lam = np.zeros(n)
    rows, cols, vals = [], [], []
    start = 0
    col = 0
    for c in range(ncomp):
        idx = order[start:start + sizes[c]]
        start += sizes[c]
        B = L[idx][:, idx].toarray()
        w, U = np.linalg.eigh(0.5 * (B + B.T))
        w = np.where(w < 1e-10, 0.0, w)
        m = len(idx)
        lam[col:col + m] = w
        rr, cc = np.meshgrid(idx, np.arange(col, col + m), indexing="ij")
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(U.ravel())
        col += m
    V = np.zeros((n, n), dtype=np.float32)
    V[np.concatenate(rows), np.concatenate(cols)] = np.concatenate(vals).astype(np.float32)
    o = np.argsort(lam, kind="stable")
    return lam[o], V[:, o], int(sizes.max()), int(ncomp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-spectrum", action="store_true",
                    help="exact full spectrum by connected component (graphs made of small components)")
    args = ap.parse_args()
    ds = args.dataset
    data = load_data(NAMES.get(ds, ds))
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    h, de = compute_graph_stats(data)
    g0 = gamma_center(h, de)
    grid3 = [float(g) for g in density_adjusted_template_gammas(h, de)]
    n = data.x.shape[0]
    k = None if args.block_spectrum else truncation(n)
    pca = 64 if data.x.shape[1] > 100 else None

    trainer = SOCTrainer(build_trainer_config(data, gamma=1.0, pca_dim=pca, truncated_k=k, modeled_subspace="legacy",
                                              graph_type="original", template_type="low"))
    if args.block_spectrum:
        # preprocess features exactly as the pipeline does, but take the spectrum by components
        lam_np, V_np, largest, ncomp = block_spectrum(data)
        small = SOCTrainer(build_trainer_config(data, gamma=1.0, pca_dim=pca, truncated_k=8, modeled_subspace="legacy",
                                                graph_type="original", template_type="low"))
        small.train(data, device="cpu")
        x = small.x_T.detach().cpu()
        V, lam = torch.from_numpy(V_np), torch.from_numpy(lam_np.astype(np.float32))
        spectrum = {"kind": "block", "components": ncomp, "largest_component": largest}
    else:
        trainer.train(data, device="cpu")
        x = trainer.x_T.detach().cpu()
        V, lam = trainer.V_template.detach().cpu(), trainer.graph_eigen.lam_L.detach().cpu()
        spectrum = {"kind": "full" if k is None else "truncated", "k": int(V.shape[1])}
    Vn, lamn = V.numpy(), lam.numpy().astype(np.float64)
    d = int(x.shape[1])
    E = compute_spectral_energy(x.numpy(), Vn)  # feature energy per mode
    total_energy = float(np.sum(x.numpy().astype(np.float64) ** 2))

    gammas = sorted(set([float(g) for g in TEMPLATE_GAMMAS] + [round(g0, 4)]))
    rows = []
    for g in gammas:
        m = compute_template(x, V, lam, template_type="low", gamma=g, nu=1.0)
        delta = (x - m).numpy()
        S = compute_spectral_energy(delta, Vn)
        entry = {"key": "g=%g" % g, "graph_type": "original", "template_gamma": g, "V": Vn, "lam_L": lamn,
                 "delta": delta, "S": S, "n": n, "d": d}
        fit = fit_best_entry([entry], [np.inf], KAPPAS, False)
        bank = horizon_score_bank(entry, q_for_fit(entry, fit), [np.inf])
        J = np.nan_to_num(np.asarray(bank["J@inf"], np.float64))
        R = np.nan_to_num(np.asarray(bank["R@inf"], np.float64))
        t = 1.0 - 1.0 / (g ** 2 + lamn)
        tabs = np.maximum(np.abs(t), 1e-12)
        rows.append({
            "gamma": g, "in_grid3": any(abs(g - a) < 1e-9 for a in grid3), "is_center": abs(g - round(g0, 4)) < 1e-9,
            "ml": float(fit["ml"]), "ml_jac": float(fit["ml"] + d * np.sum(np.log(tabs))),
            "rho": float(fit["rho"]), "kappa": float(fit["kappa"]),
            "annihilated": 1.0 - float(np.sum(delta.astype(np.float64) ** 2)) / total_energy,
            "residual_energy_modes": float(np.sum(S)) / max(float(np.sum(E)), 1e-300),
            "min_abs_t": float(np.min(tabs)), "n_negative_t": int(np.sum(t < 0)),
            "auc_J": 100 * safe_auc(y[mask], J[mask]), "auc_R": 100 * safe_auc(y[mask], R[mask]),
            "ks_J": float(ks_null_deviation(J)), "ks_R": float(ks_null_deviation(R)),
        })
        print(ds, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rows[-1].items()}, flush=True)
    out = {"dataset": ds, "h": float(h), "d_e": float(de), "gamma0": g0, "grid3": grid3, "spectrum": spectrum,
           "n": n, "d": d, "pca": pca, "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
