"""DGraph unsupervised pipeline using cached eigenpairs.

Same logic as test_unsupervised_final.py but uses precomputed eigenpairs
since DGraph (3.7M nodes) is too large for online eigendecomposition.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]


def load_dgraph_data():
    """Load DGraph and return features, labels, labeled mask."""
    from data_utils import load_data
    data = load_data("dgraph")
    return data


def compute_template(x, V, lam_L, gamma):
    """Low-pass template: m = S_{gamma} x."""
    smoother_eig = 1.0 / (gamma**2 + lam_L)
    return V @ (smoother_eig[:, None] * (V.T @ x))


def score_nodes(delta, V, q):
    """Compute J* and R scores."""
    q_safe = np.maximum(q, 1e-12)
    delta_hat = V.T @ delta  # (k, d)

    # J* (precision energy)
    w = np.sqrt(q_safe)[:, None] * delta_hat
    je = np.sum((V @ w)**2, axis=1)

    # R (spectral ratio)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot

    return je, jr


def run(cache_dir="cache/dgraph_eigen", device="cpu"):
    print("Loading DGraph...", flush=True)
    data = load_dgraph_data()
    x_raw = data.x.float().numpy()
    y = data.y.numpy()
    n, d = x_raw.shape
    print("  n=%d d=%d" % (n, d), flush=True)

    # Labeled mask (y=0 normal, y=1 anomaly; exclude unlabeled)
    labeled_mask = (y >= 0) & (y <= 1)
    y_labeled = y[labeled_mask]
    print("  labeled=%d anomalies=%d (%.1f%%)" %
          (labeled_mask.sum(), (y_labeled == 1).sum(),
           (y_labeled == 1).mean() * 100), flush=True)

    # Normalize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)

    # Try different cached eigenpairs
    results = []
    for subset in ["labeled"]:
        for k in [128, 256]:
            fname = "%s_original_k%d_lowrank.pt" % (subset, k)
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                print("  Cache not found: %s" % fpath, flush=True)
                continue

            print("\n  Loading %s..." % fname, flush=True)
            cache = torch.load(fpath, weights_only=False)
            V = cache["V"].numpy()  # (n_subset, k)
            lam_L = cache["lam_L"].numpy()  # (k,)

            # Get subset indices if labeled
            if subset == "labeled":
                idx = np.where(labeled_mask)[0]
                x_sub = x[idx]
                y_sub = y[idx]
                mask_sub = (y_sub >= 0) & (y_sub <= 1)
            else:
                x_sub = x
                y_sub = y
                mask_sub = labeled_mask

            n_sub = x_sub.shape[0]

            for gamma in GAMMAS:
                tmpl = compute_template(x_sub, V, lam_L, gamma)
                delta = x_sub - tmpl
                S = compute_spectral_energy(delta, V)

                # ML-optimize rho
                best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
                for kappa in KAPPAS:
                    rho = solve_rho_newton(S, lam_L, kappa, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kappa)
                    ml = marginal_log_likelihood_stationary(S, q, n_sub, d)
                    if ml > best_ml:
                        best_ml, best_rho, best_kappa = ml, rho, kappa

                q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
                je, jr = score_nodes(delta, V, q)

                if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                    continue

                # hf_frac rule
                n_modes = len(S)
                k10 = max(1, n_modes // 10)
                hf_frac = float(S[-k10:].sum() / (S.sum() + 1e-12))
                use_R = hf_frac < 0.05
                scores = jr if use_R else je
                score_name = "R" if use_R else "J*"

                auc_sel = roc_auc_score(y_sub[mask_sub], scores[mask_sub])
                auc_je = roc_auc_score(y_sub[mask_sub], je[mask_sub])
                auc_jr = roc_auc_score(y_sub[mask_sub], jr[mask_sub])

                print("    k=%d g=%s: ML=%.0f rho=%.3f hf=%.4f %s=%.1f%% (J*=%.1f%% R=%.1f%%)" %
                      (k, gamma, best_ml, best_rho, hf_frac, score_name,
                       auc_sel*100, auc_je*100, auc_jr*100), flush=True)

                results.append({
                    "k": k, "gamma": gamma, "subset": subset,
                    "auc_selected": auc_sel, "auc_je": auc_je, "auc_jr": auc_jr,
                    "score_name": score_name, "hf_frac": hf_frac,
                    "rho": best_rho, "kappa": best_kappa, "ml": best_ml,
                })

    if results:
        # ML-best config
        ml_best = max(results, key=lambda r: r["ml"])
        # Oracle
        oracle = max(results, key=lambda r: r["auc_selected"])

        print("\n=== RESULTS ===")
        print("ML-selected: %.1f%% (k=%d g=%s %s)" %
              (ml_best["auc_selected"]*100, ml_best["k"], ml_best["gamma"],
               ml_best["score_name"]))
        print("Oracle:      %.1f%% (k=%d g=%s)" %
              (oracle["auc_selected"]*100, oracle["k"], oracle["gamma"]))
        print("Reported:    65.7%% (k=256, J*)")
        print("Best base:   64.2%% (DIF)")

        os.makedirs("results", exist_ok=True)
        with open("results/unsup_final_dgraph.json", "w") as f:
            json.dump({"ml_best": ml_best, "oracle": oracle, "all": results},
                      f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(device=args.device)
