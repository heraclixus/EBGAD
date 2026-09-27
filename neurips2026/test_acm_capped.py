"""ACM: test capped J* (= J* with eigenvalue floor) as alternative to C score.

Also reproduce reported best config for verification.
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
from soc.soc_anomaly import (precision_energy_anomaly, precision_ratio_anomaly,
                              control_energy_anomaly)
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def score_capped_je(x, tmpl, V, lam_Q, cap_percentile=90):
    """J* with eigenvalue floor at the given percentile."""
    delta = x - tmpl
    q = lam_Q.numpy()
    q_floor = np.percentile(q, 100 - cap_percentile)  # floor at bottom percentile
    q_capped = np.maximum(q, q_floor)

    delta_np = delta.numpy()
    V_np = V.numpy()
    delta_hat = V_np.T @ delta_np
    w = np.sqrt(q_capped)[:, None] * delta_hat
    node_energy = np.sum((V_np @ w)**2, axis=1)
    return node_energy


def run_config(data, gamma, pca_dim, device, kappa_min=None):
    """Run config with optional kappa floor."""
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="low", graph_type="original", gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))

    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl.numpy()
    S = compute_spectral_energy(delta, V.numpy())

    # ML optimize rho, with optional kappa minimum
    kappas = KAPPAS
    if kappa_min is not None:
        kappas = [k for k in KAPPAS if k >= kappa_min]
        if not kappas:
            kappas = [kappa_min]

    best_ml, rho, kappa = -np.inf, 0.5, 0.0
    for k in kappas:
        r = solve_rho_newton(S, lam_L, k, d_eff)
        q = Q_rho_eigenvalues(lam_L, r, k)
        ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
        if ml > best_ml:
            best_ml, rho, kappa = ml, r, k

    q = Q_rho_eigenvalues(lam_L, rho, kappa)
    lam_Q = torch.from_numpy(q).float()
    V_t = V.float()

    # Standard scores
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()
        ce = control_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                    lam_penalty=50.0).cpu().numpy()

    # Capped J* at various percentiles
    capped_scores = {}
    for pct in [50, 75, 90, 95]:
        capped = score_capped_je(x, tmpl, V_t, lam_Q, cap_percentile=pct)
        capped_scores["J*_cap%d" % pct] = capped

    # Spectral condition number
    q_ratio = q.max() / (q.min() + 1e-12)

    return {
        "J*": je, "R": jr, "C": ce,
        **capped_scores,
        "rho": rho, "kappa": kappa, "ml": best_ml,
        "q_ratio": q_ratio, "q_min": q.min(), "q_max": q.max(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_data("acm")
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    print("ACM: n=%d d=%d" % (data.x.shape[0], data.x.shape[1]), flush=True)

    # Test configs
    configs = [
        # Our default pipeline
        {"gamma": 1.0, "pca": 64, "kappa_min": None, "label": "default (g=1.0 p64)"},
        # Reported-like config
        {"gamma": 0.5, "pca": 128, "kappa_min": None, "label": "reported-like (g=0.5 p128)"},
        {"gamma": 0.5, "pca": 64, "kappa_min": None, "label": "g=0.5 p64"},
        # With kappa floor (prevents extreme spectral range)
        {"gamma": 0.5, "pca": 128, "kappa_min": 0.5, "label": "g=0.5 p128 kmin=0.5"},
        {"gamma": 0.5, "pca": 128, "kappa_min": 1.0, "label": "g=0.5 p128 kmin=1.0"},
        {"gamma": 1.0, "pca": 64, "kappa_min": 0.5, "label": "g=1.0 p64 kmin=0.5"},
        {"gamma": 1.0, "pca": 128, "kappa_min": None, "label": "g=1.0 p128"},
        {"gamma": 1.0, "pca": 128, "kappa_min": 0.5, "label": "g=1.0 p128 kmin=0.5"},
        # Different PCA
        {"gamma": 0.5, "pca": 256, "kappa_min": None, "label": "g=0.5 p256"},
    ]

    for cfg in configs:
        print("\n--- %s ---" % cfg["label"], flush=True)
        try:
            r = run_config(data, cfg["gamma"], cfg["pca"], args.device, cfg.get("kappa_min"))
            aucs = {}
            for sname in ["J*", "R", "C", "J*_cap50", "J*_cap75", "J*_cap90", "J*_cap95"]:
                if sname in r and not np.any(np.isnan(r[sname])):
                    aucs[sname] = roc_auc_score(y[mask], r[sname][mask])

            print("  rho=%.3f kappa=%.2f q_ratio=%.0f (q_min=%.4f q_max=%.2f)" %
                  (r["rho"], r["kappa"], r["q_ratio"], r["q_min"], r["q_max"]))
            for sname, auc in sorted(aucs.items(), key=lambda x: -x[1]):
                print("  %s: %.1f%%" % (sname, auc*100))
        except Exception as e:
            print("  ERROR: %s" % str(e)[:80])


if __name__ == "__main__":
    main()
