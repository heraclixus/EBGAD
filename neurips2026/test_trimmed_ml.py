"""Test trimmed ML for gamma selection.

Standard ML: compute S_j from ALL nodes.
Trimmed ML: sort nodes by ||delta_i||^2, keep bottom p%, recompute S_j, then ML.

This should prefer gammas that explain NORMALS well, not the entire dataset.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0]
TEMPLATES = ["low", "affinity"]
TRIM_FRACS = [1.0, 0.95, 0.90, 0.85, 0.80]

BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 78.1, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 83.5,
}


def compute_trimmed_spectral_energy(delta, V, trim_frac):
    """Compute spectral energy S_j using only the bottom trim_frac of nodes by residual norm."""
    n = delta.shape[0]
    if trim_frac >= 1.0:
        return compute_spectral_energy(delta, V)

    norms = np.sum(delta**2, axis=1)
    k = max(1, int(n * trim_frac))
    keep_idx = np.argsort(norms)[:k]

    delta_trimmed = delta[keep_idx]
    V_trimmed = V[keep_idx]

    # S_j = ||delta_trimmed^T v_j||^2 (but v_j is now subsampled)
    # This isn't exactly right because V was computed on all nodes.
    # Better: just recompute the projection using the trimmed nodes.
    proj = V_trimmed.T @ delta_trimmed  # (k_modes, d)
    S = np.sum(proj**2, axis=1)
    return S


def run_config(data, tmpl, gamma, pca_dim, device="cpu"):
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type=tmpl, graph_type="original", gamma=gamma,
    )
    if pca_dim is not None and data.x.shape[1] > pca_dim:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=pca_dim, encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))
    elif pca_dim is not None:
        return None
    try:
        trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl_t = trainer.template.cpu()
        x = trainer.x_T.cpu()
        n, d = x.shape
        delta = x.numpy() - tmpl_t.numpy()
        return {"V": V, "lam_L": lam_L, "tmpl": tmpl_t, "x": x, "delta": delta, "n": n, "d": d}
    except:
        return None


def optimize_rho(S, lam_L, d, n_eff):
    """ML-optimize rho given spectral energy S."""
    best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
    for kappa in KAPPAS:
        rho = solve_rho_newton(S, lam_L, kappa, d)
        q = Q_rho_eigenvalues(lam_L, rho, kappa)
        ml = marginal_log_likelihood_stationary(S, q, n_eff, d)
        if ml > best_ml:
            best_ml, best_rho, best_kappa = ml, rho, kappa
    return best_ml, best_rho, best_kappa


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d_feat = data.x.shape[1]
    pca_dim = 64 if d_feat > 100 else None

    # For each trim fraction, track the ML-best config and its AUROC
    results_by_trim = {}

    for trim in TRIM_FRACS:
        best_ml = -np.inf
        best_info = None

        for tmpl in TEMPLATES:
            for gamma in GAMMAS:
                result = run_config(data, tmpl, gamma, pca_dim, device)
                if result is None:
                    continue

                V, lam_L, delta = result["V"], result["lam_L"], result["delta"]
                n, d = result["n"], result["d"]

                # Compute trimmed spectral energy
                n_eff = max(1, int(n * trim))
                S = compute_trimmed_spectral_energy(delta, V, trim)
                ml, rho, kappa = optimize_rho(S, lam_L, d, n_eff)

                if ml > best_ml:
                    best_ml = ml
                    # Compute scores using FULL data (not trimmed) with optimized rho
                    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
                    V_t = torch.from_numpy(V).float()
                    with torch.no_grad():
                        je = precision_energy_anomaly(
                            result["x"], result["tmpl"], V_t, lam_Q, 2.0, 1.0,
                            prior_mean_mode="stationary").cpu().numpy()
                        jr = precision_ratio_anomaly(
                            result["x"], result["tmpl"], V_t, lam_Q, 2.0, 1.0,
                            prior_mean_mode="stationary").cpu().numpy()

                    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                        continue

                    auc_je = roc_auc_score(y[mask], je[mask])
                    auc_jr = roc_auc_score(y[mask], jr[mask])
                    best_info = {
                        "ml": ml, "rho": rho, "kappa": kappa,
                        "tmpl": tmpl, "gamma": gamma,
                        "auc_je": auc_je, "auc_jr": auc_jr,
                        "auc_max": max(auc_je, auc_jr),
                        "best_score": "J*" if auc_je >= auc_jr else "R",
                    }

        if best_info:
            results_by_trim[trim] = best_info
            print("    trim=%.2f: %s g=%s -> max(J*,R)=%.1f%% (J*=%.1f%% R=%.1f%%) rho=%.3f" %
                  (trim, best_info["tmpl"], best_info["gamma"],
                   best_info["auc_max"]*100, best_info["auc_je"]*100,
                   best_info["auc_jr"]*100, best_info["rho"]), flush=True)

    # Also compute oracle across all configs
    oracle = 0
    for tmpl in TEMPLATES:
        for gamma in GAMMAS:
            result = run_config(data, tmpl, gamma, pca_dim, device)
            if result is None:
                continue
            V, lam_L, delta = result["V"], result["lam_L"], result["delta"]
            n, d = result["n"], result["d"]
            S = compute_spectral_energy(delta, V)
            _, rho, kappa = optimize_rho(S, lam_L, d, n)
            lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
            V_t = torch.from_numpy(V).float()
            with torch.no_grad():
                je = precision_energy_anomaly(
                    result["x"], result["tmpl"], V_t, lam_Q, 2.0, 1.0,
                    prior_mean_mode="stationary").cpu().numpy()
                jr = precision_ratio_anomaly(
                    result["x"], result["tmpl"], V_t, lam_Q, 2.0, 1.0,
                    prior_mean_mode="stationary").cpu().numpy()
            if not np.any(np.isnan(je)):
                auc_je = roc_auc_score(y[mask], je[mask])
                auc_jr = roc_auc_score(y[mask], jr[mask])
                oracle = max(oracle, auc_je, auc_jr)

    return {"trims": results_by_trim, "oracle": oracle}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    ds = args.dataset
    print("=== %s ===" % ds, flush=True)
    r = run_dataset(ds, args.device)
    if r:
        base = BASELINES.get(ds, 0)
        print("\n  Oracle: %.1f%%  Baseline: %.1f%%" % (r["oracle"]*100, base))
        for trim in TRIM_FRACS:
            if trim in r["trims"]:
                info = r["trims"][trim]
                print("  trim=%.2f: %.1f%% (%s g=%s %s) vsBase=%+.1f" %
                      (trim, info["auc_max"]*100, info["tmpl"], info["gamma"],
                       info["best_score"], info["auc_max"]*100 - base))

        os.makedirs("results", exist_ok=True)
        with open("results/trimmed_ml_%s.json" % ds, "w") as f:
            json.dump(r, f, indent=2, default=str)


if __name__ == "__main__":
    main()
