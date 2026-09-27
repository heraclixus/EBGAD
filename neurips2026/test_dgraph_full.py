"""DGraph comprehensive test with cached eigenpairs.

Sweep: normalization × gamma × k × scoring to find what works.
The reported 65.7% used k=256; our zscore pipeline got 37.5%.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import kstest

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def ks_null(scores):
    s = scores[scores > 0]
    if len(s) < 10: return 0.0
    mu, var = s.mean(), s.var()
    if var < 1e-12 or mu < 1e-12: return 0.0
    k = max(0.01, 2*mu**2/var); a = var/(2*mu)
    try: return kstest(s/a, 'chi2', args=(k,))[0]
    except: return 0.0


def score_nodes(delta, V, q):
    q_safe = np.maximum(q, 1e-12)
    delta_hat = V.T @ delta
    w = np.sqrt(q_safe)[:, None] * delta_hat
    je = np.sum((V @ w)**2, axis=1)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot
    return je, jr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cache_dir = "cache/dgraph_eigen"
    from data_utils import load_data
    data = load_data("dgraph")
    x_raw = data.x.float().numpy()
    y = data.y.numpy()
    n, d = x_raw.shape
    labeled_mask = (y >= 0) & (y <= 1)
    print("DGraph: n=%d d=%d labeled=%d anom=%.1f%%" %
          (n, d, labeled_mask.sum(), y[labeled_mask].mean()*100), flush=True)

    best_overall = 0
    best_config = ""

    for norm_name, norm_fn in [("zscore", StandardScaler()),
                                ("minmax", MinMaxScaler()),
                                ("raw", None)]:
        if norm_fn:
            x = norm_fn.fit_transform(x_raw).astype(np.float32)
        else:
            x = x_raw.astype(np.float32)

        for k in [128, 256, 500]:
            fname = "labeled_original_k%d_lowrank.pt" % k
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                continue
            cache = torch.load(fpath, weights_only=False)
            V = cache["V"].numpy()
            lam_L = cache["lam_L"].numpy()

            idx = np.where(labeled_mask)[0]
            x_sub = x[idx]
            y_sub = y[idx]
            mask_sub = (y_sub >= 0) & (y_sub <= 1)

            for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                smoother = 1.0 / (gamma**2 + lam_L)
                tmpl = V @ (smoother[:, None] * (V.T @ x_sub))
                delta = x_sub - tmpl
                S = compute_spectral_energy(delta, V)

                best_ml, rho, kappa = -np.inf, 0.5, 0.0
                for kk in KAPPAS:
                    r = solve_rho_newton(S, lam_L, kk, d)
                    q = Q_rho_eigenvalues(lam_L, r, kk)
                    ml = marginal_log_likelihood_stationary(S, q, len(x_sub), d)
                    if ml > best_ml:
                        best_ml, rho, kappa = ml, r, kk

                q = Q_rho_eigenvalues(lam_L, rho, kappa)
                je, jr = score_nodes(delta, V, q)

                if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                    continue

                auc_je = roc_auc_score(y_sub[mask_sub], je[mask_sub])
                auc_jr = roc_auc_score(y_sub[mask_sub], jr[mask_sub])
                ks_je = ks_null(je[mask_sub])
                ks_jr = ks_null(jr[mask_sub])
                ks_pick = "J*" if ks_je > ks_jr else "R"
                auc_ks = auc_je if ks_pick == "J*" else auc_jr
                best_auc = max(auc_je, auc_jr)

                if best_auc > best_overall:
                    best_overall = best_auc
                    best_config = "%s k=%d g=%s %s" % (norm_name, k, gamma,
                        "J*" if auc_je > auc_jr else "R")

                print("  %s k=%d g=%s: J*=%.1f%% R=%.1f%% KS=%s=%.1f%% rho=%.3f" %
                      (norm_name, k, gamma, auc_je*100, auc_jr*100,
                       ks_pick, auc_ks*100, rho), flush=True)

    print("\n=== BEST ===")
    print("  %.1f%% (%s)" % (best_overall*100, best_config))
    print("  Reported: 65.7%")
    print("  Baseline DIF: 64.2%")


if __name__ == "__main__":
    main()
