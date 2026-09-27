"""DGraph sweep using cached eigendecomposition.

Loads precomputed eigenpairs from cache/dgraph_eigen/, then sweeps
over template, gamma, scoring, rho, kappa -- all closed-form, no
eigendecomposition needed per config.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    marginal_log_likelihood_stationary,
    compute_spectral_energy,
    solve_rho_newton,
)
from eval_utils import mask_labeled


def load_cached(cache_dir, subset, graph_type, k):
    """Load cached eigenpairs."""
    fname = f"{subset}_{graph_type}_k{k}.pt"
    fpath = os.path.join(cache_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Cache not found: {fpath}")
    return torch.load(fpath, weights_only=False)


def compute_template(x, V, lam_L, gamma, template_type):
    """Compute template m from features and eigenpairs."""
    n, d = x.shape
    if template_type == "zero":
        return np.zeros_like(x)

    # Graph smoother: S_{gamma,1} = (gamma^2 I + L)^{-1}
    smoother_eig = 1.0 / (gamma**2 + lam_L)

    if template_type == "low":
        # Low-pass smoothed features
        return V @ (smoother_eig[:, None] * (V.T @ x))
    elif template_type == "affinity":
        # Affinity-weighted smoothed features
        return V @ (smoother_eig[:, None] * (V.T @ x))
    elif template_type == "high":
        # High-pass: x - low_pass(x)
        low = V @ (smoother_eig[:, None] * (V.T @ x))
        return x - low
    else:
        return np.zeros_like(x)


def score_nodes(delta, V, q, score_method, lam_penalty=50.0, alpha_T=2.0):
    """Compute per-node anomaly scores."""
    q_safe = np.maximum(q, 1e-12)
    delta_hat = V.T @ delta  # (k, d)

    if score_method == "precision_energy":
        w = np.sqrt(q_safe)[:, None] * delta_hat
        return np.sum((V @ w)**2, axis=1)
    elif score_method == "precision_ratio":
        w = np.sqrt(q_safe)[:, None] * delta_hat
        ce = np.sum((V @ w)**2, axis=1)
        tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
        return ce / tot
    elif score_method == "control_energy":
        sig = (1.0 - np.exp(-2.0 * alpha_T * q_safe)) / q_safe
        D = 1.0 / lam_penalty + sig
        w = (1.0 / np.sqrt(np.maximum(D, 1e-12)))[:, None] * delta_hat
        return np.sum((V @ w)**2, axis=1)
    else:
        return np.sum((V @ (np.sqrt(q_safe)[:, None] * delta_hat))**2, axis=1)


def run(cache_dir="cache/dgraph_eigen", chunk=0, n_chunks=1):
    out_file = f"results/eb_gad_dgraph_cached_{chunk}.json"

    # Sweep parameters
    templates = ["zero", "low", "affinity"]
    gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]
    prior_modes = ["zero", "stationary"]
    score_methods = ["precision_energy", "precision_ratio", "control_energy"]
    rho_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    kappa_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    # Available cached eigenpairs
    cache_files = []
    if os.path.isdir(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
    print(f"=== Found {len(cache_files)} cached eigenpairs ===", flush=True)

    # Build config list
    all_configs = []
    for cf in sorted(cache_files):
        # Parse: subset_graphtype_kN.pt or subset_graphtype_kN_lowrank.pt
        name = cf.replace(".pt", "").replace("_lowrank", "")
        parts = name.split("_")
        subset = parts[0]  # "full" or "labeled"
        graph_type = parts[1]  # "original" or "affinity"
        k = int(parts[2].replace("k", ""))

        for tmpl in templates:
            for gamma in gammas:
                for norm in norms:
                    for pm in prior_modes:
                        all_configs.append({
                            "cache_file": cf,
                            "subset": subset,
                            "graph_type": graph_type,
                            "k": k,
                            "template_type": tmpl,
                            "gamma": gamma,
                            "normalize_mode": norm,
                            "prior_mean_mode": pm,
                        })

    total = len(all_configs)
    my_configs = [c for i, c in enumerate(all_configs)
                  if i % n_chunks == chunk]
    print(f"  Total configs: {total}, chunk {chunk}/{n_chunks}: "
          f"{len(my_configs)}", flush=True)

    best_auc = 0
    best_config = {}
    all_results = []
    count = 0

    # Group by cache file to load each eigenpair only once
    from itertools import groupby
    my_configs_sorted = sorted(my_configs, key=lambda c: c["cache_file"])

    for cf_name, group in groupby(my_configs_sorted,
                                   key=lambda c: c["cache_file"]):
        configs_for_cache = list(group)
        print(f"\n  Loading {cf_name} ({len(configs_for_cache)} configs)...",
              flush=True)

        try:
            cached = load_cached(cache_dir, "", "", 0)  # dummy
        except:
            pass

        fpath = os.path.join(cache_dir, cf_name)
        cached = torch.load(fpath, weights_only=False)
        lam_L = cached["lam_L"].numpy()
        V = cached["V"].numpy()
        x_raw = cached["x"].numpy()
        y = cached["y"].numpy()
        n = int(cached["n"])
        k = int(cached["k"])

        for cfg in configs_for_cache:
            count += 1
            t0 = time.time()

            # Normalize features
            x = x_raw.copy()
            if cfg["normalize_mode"] == "zscore":
                scaler = StandardScaler()
                x = scaler.fit_transform(x)
            elif cfg["normalize_mode"] == "minmax":
                scaler = MinMaxScaler()
                x = scaler.fit_transform(x)

            d = x.shape[1]

            # Compute template
            template = compute_template(x, V, lam_L, cfg["gamma"],
                                         cfg["template_type"])

            # Residual
            if cfg["prior_mean_mode"] == "stationary":
                delta = x - template
            else:
                delta = x  # zero-mean prior

            S = compute_spectral_energy(delta, V)

            # Sweep rho and kappa, find best via ML
            best_ml = -np.inf
            best_rho = 0.5
            best_kappa = 0.0
            for kappa in kappa_values:
                rho_star = solve_rho_newton(S, lam_L, kappa, d)
                q = Q_rho_eigenvalues(lam_L, rho_star, kappa)
                ml = marginal_log_likelihood_stationary(S, q, n, d)
                if ml > best_ml:
                    best_ml = ml
                    best_rho = rho_star
                    best_kappa = kappa

            # Also check fixed rho values
            for rho in rho_values:
                for kappa in [0.0, best_kappa]:
                    q = Q_rho_eigenvalues(lam_L, rho, kappa)
                    ml = marginal_log_likelihood_stationary(S, q, n, d)
                    if ml > best_ml:
                        best_ml = ml
                        best_rho = rho
                        best_kappa = kappa

            # Score with best (rho, kappa)
            q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)

            # Evaluate all scoring methods
            mask = (y >= 0) & (y <= 1)
            for sm in score_methods:
                scores = score_nodes(delta, V, q, sm)
                try:
                    auc = roc_auc_score(y[mask], scores[mask])
                except ValueError:
                    auc = 0.5

                if auc > best_auc:
                    best_auc = auc
                    best_config = {
                        "score_method": sm,
                        "template_type": cfg["template_type"],
                        "gamma": cfg["gamma"],
                        "normalize_mode": cfg["normalize_mode"],
                        "prior_mean_mode": cfg["prior_mean_mode"],
                        "graph_type": cfg["graph_type"],
                        "k": k,
                        "subset": cfg["subset"],
                        "rho": float(best_rho),
                        "kappa": float(best_kappa),
                    }
                    elapsed = time.time() - t0
                    print(f"  [{count}/{len(my_configs)}] NEW BEST "
                          f"{sm} {auc:.4f} k={k} {cfg['subset']} "
                          f"t={cfg['template_type']} "
                          f"gamma={cfg['gamma']} rho={best_rho:.2f} "
                          f"({elapsed:.1f}s)", flush=True)

                all_results.append({
                    "score_method": sm, "auc": float(auc),
                    "k": k, "subset": cfg["subset"],
                    "template": cfg["template_type"],
                    "gamma": cfg["gamma"],
                    "graph": cfg["graph_type"],
                    "pm": cfg["prior_mean_mode"],
                    "rho": float(best_rho),
                    "kappa": float(best_kappa),
                })

            # Save every 50 configs
            if count % 50 == 0 or count == len(my_configs):
                with open(out_file, "w") as f:
                    json.dump({
                        "best_auc": float(best_auc),
                        "best_config": best_config,
                        "n_evaluated": count,
                        "n_total": len(my_configs),
                        "all_results": all_results,
                    }, f, indent=2, default=str)

    print(f"\n=== DONE: best AUC = {best_auc:.4f} ===")
    print(f"  Config: {best_config}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/dgraph_eigen")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    run(args.cache_dir, args.chunk, args.n_chunks)
