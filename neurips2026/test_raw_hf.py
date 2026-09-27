"""Compute gamma-invariant high-frequency energy fraction from RAW features.

hf_raw = sum(S_raw[top 10%]) / sum(S_raw)
where S_raw_j = ||X^T v_j||^2 (raw features, not residuals)
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
GT_SCORE = {"weibo":"J*","reddit":"J*","amazon":"J*","yelpchi":"J*","blogcatalog":"J*",
            "facebook":"R","acm":"J*","elliptic":"R","elliptic_plus_plus":"R","t_finance":"R"}


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    # Run with gamma=0.5 to get eigenvectors and normalized features
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="low", graph_type="original", gamma=0.5)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))

    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    x = trainer.x_T.cpu().numpy()
    tmpl = trainer.template.cpu().numpy()
    n, d_eff = x.shape

    # Raw feature spectral energy
    S_raw = compute_spectral_energy(x, V)
    n_modes = len(S_raw)
    k10 = max(1, n_modes // 10)
    hf_raw = float(S_raw[-k10:].sum() / (S_raw.sum() + 1e-12))

    # Centered feature spectral energy
    x_c = x - x.mean(axis=0, keepdims=True)
    S_centered = compute_spectral_energy(x_c, V)
    hf_centered = float(S_centered[-k10:].sum() / (S_centered.sum() + 1e-12))

    # Residual spectral energy at gamma=0.5
    delta_05 = x - tmpl
    S_05 = compute_spectral_energy(delta_05, V)
    hf_res05 = float(S_05[-k10:].sum() / (S_05.sum() + 1e-12))

    # Residual spectral energy at gamma=1.0
    smoother_1 = 1.0 / (1.0**2 + lam_L)
    tmpl_1 = V @ (smoother_1[:, None] * (V.T @ x))
    delta_10 = x - tmpl_1
    S_10 = compute_spectral_energy(delta_10, V)
    hf_res10 = float(S_10[-k10:].sum() / (S_10.sum() + 1e-12))

    # Also: compute J* and R at gamma=0.5 and 1.0 for AUROC
    results = {}
    for gamma in [0.5, 1.0]:
        smoother = 1.0 / (gamma**2 + lam_L)
        t = V @ (smoother[:, None] * (V.T @ x))
        delta = x - t
        S = compute_spectral_energy(delta, V)
        _, rho, kappa = -np.inf, 0.5, 0.0
        best_ml = -np.inf
        for k in KAPPAS:
            r = solve_rho_newton(S, lam_L, k, d_eff)
            q = Q_rho_eigenvalues(lam_L, r, k)
            ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
            if ml > best_ml:
                best_ml, rho, kappa = ml, r, k

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
        V_t = torch.from_numpy(V).float()
        x_t = torch.from_numpy(x).float()
        t_t = torch.from_numpy(t).float()
        with torch.no_grad():
            je = precision_energy_anomaly(x_t, t_t, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x_t, t_t, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()
        if not np.any(np.isnan(je)):
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            results[gamma] = {"je": auc_je, "jr": auc_jr, "rho": rho}

    return {
        "hf_raw": hf_raw, "hf_centered": hf_centered,
        "hf_res05": hf_res05, "hf_res10": hf_res10,
        "scores": results, "gt": GT_SCORE.get(ds, "?"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    ds = args.dataset
    print("=== %s ===" % ds, flush=True)
    try:
        r = run_dataset(ds, args.device)
        print("  hf_raw=%.4f  hf_centered=%.4f  hf_res05=%.4f  hf_res10=%.4f  gt=%s" %
              (r["hf_raw"], r["hf_centered"], r["hf_res05"], r["hf_res10"], r["gt"]))
        for gamma, s in sorted(r["scores"].items()):
            print("  g=%s: J*=%.1f%%  R=%.1f%%  rho=%.3f" %
                  (gamma, s["je"]*100, s["jr"]*100, s["rho"]))

        os.makedirs("results", exist_ok=True)
        with open("results/raw_hf_%s.json" % ds, "w") as f:
            json.dump(r, f, indent=2, default=str)
    except Exception as e:
        import traceback
        print("  ERROR: %s" % str(e)[:80])
        traceback.print_exc()


if __name__ == "__main__":
    main()
