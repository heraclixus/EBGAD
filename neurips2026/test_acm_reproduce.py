"""Reproduce ACM reported best: C score, PCA=128, gamma=0.5, rho=0.9, kappa=0.1.

Run the exact reported config and verify 93.5%.
Also test nearby configs to understand sensitivity.
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
                              control_energy_anomaly, ce_ratio_anomaly)
from soc.prior_optimizer import Q_rho_eigenvalues


def run_fixed_config(data, gamma, pca_dim, rho, kappa, device="cpu"):
    d = data.x.shape[1]
    cfg = dict(kappa=kappa, nu=1.0, rho=rho, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="affinity", graph_type="original", gamma=gamma)
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

    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()

    scores = {}
    with torch.no_grad():
        scores["J*"] = precision_energy_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                                prior_mean_mode="stationary").cpu().numpy()
        scores["R"] = precision_ratio_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                              prior_mean_mode="stationary").cpu().numpy()
        scores["C"] = control_energy_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                             lam_penalty=50.0).cpu().numpy()
        scores["C_R"] = ce_ratio_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                         lam_penalty=50.0).cpu().numpy()
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_data("acm")
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    print("ACM: n=%d d=%d" % (data.x.shape[0], data.x.shape[1]), flush=True)

    # Reported best config
    configs = [
        {"label": "REPORTED (aff g=0.5 p128 rho=0.9 k=0.1)",
         "gamma": 0.5, "pca": 128, "rho": 0.9, "kappa": 0.1},
        # Variations
        {"label": "aff g=0.2 p128 rho=0.9 k=0.1",
         "gamma": 0.2, "pca": 128, "rho": 0.9, "kappa": 0.1},
        {"label": "aff g=0.5 p64 rho=0.9 k=0.1",
         "gamma": 0.5, "pca": 64, "rho": 0.9, "kappa": 0.1},
        {"label": "aff g=0.5 p128 rho=0.5 k=0.1",
         "gamma": 0.5, "pca": 128, "rho": 0.5, "kappa": 0.1},
        {"label": "aff g=0.5 p128 rho=0.9 k=0.5",
         "gamma": 0.5, "pca": 128, "rho": 0.9, "kappa": 0.5},
        {"label": "aff g=0.5 p128 rho=0.9 k=1.0",
         "gamma": 0.5, "pca": 128, "rho": 0.9, "kappa": 1.0},
        # Low template variants
        {"label": "LOW g=0.5 p128 rho=0.9 k=0.1",
         "gamma": 0.5, "pca": 128, "rho": 0.9, "kappa": 0.1},
        {"label": "LOW g=1.0 p128 rho=0.9 k=0.1",
         "gamma": 1.0, "pca": 128, "rho": 0.9, "kappa": 0.1},
        {"label": "LOW g=1.0 p64 rho=0.9 k=0.1",
         "gamma": 1.0, "pca": 64, "rho": 0.9, "kappa": 0.1},
    ]

    # Override template for LOW variants
    for cfg in configs:
        if cfg["label"].startswith("LOW"):
            pass  # handled below

    for cfg in configs:
        print("\n--- %s ---" % cfg["label"], flush=True)
        try:
            # Temporarily modify template type for LOW configs
            if cfg["label"].startswith("LOW"):
                # Run with low template
                d = data.x.shape[1]
                trainer_cfg = dict(kappa=cfg["kappa"], nu=1.0, rho=cfg["rho"],
                                   lam_penalty=50.0, alpha=2.0, T=1.0,
                                   normalize_mode="zscore", prior_mean_mode="stationary",
                                   laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
                                   epochs=0, normalize_features=True,
                                   template_type="low", graph_type="original",
                                   gamma=cfg["gamma"])
                if cfg["pca"] and d > cfg["pca"]:
                    trainer_cfg.update(dict(use_encoder=True, encoder_type="pca",
                                           encoder_hid_dim=cfg["pca"], encoder_num_layers=1,
                                           encoder_dropout=0.0, encoder_lr=0.0,
                                           encoder_epochs=0, encoder_alpha=1.0,
                                           encoder_weight_decay=0.0))
                trainer = SOCTrainer(SOCTrainerConfig(**trainer_cfg))
                trainer.train(data, device=args.device)
                V = trainer.V.cpu()
                lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
                tmpl = trainer.template.cpu()
                x = trainer.x_T.cpu()
                lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, cfg["rho"], cfg["kappa"])).float()
                with torch.no_grad():
                    je = precision_energy_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                                  prior_mean_mode="stationary").cpu().numpy()
                    jr = precision_ratio_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                                 prior_mean_mode="stationary").cpu().numpy()
                    ce = control_energy_anomaly(x, tmpl, V, lam_Q, 2.0, 1.0,
                                                lam_penalty=50.0).cpu().numpy()
                scores = {"J*": je, "R": jr, "C": ce}
            else:
                scores = run_fixed_config(data, cfg["gamma"], cfg["pca"],
                                         cfg["rho"], cfg["kappa"], args.device)

            for sname, sarr in sorted(scores.items()):
                if not np.any(np.isnan(sarr)):
                    auc = roc_auc_score(y[mask], sarr[mask])
                    print("  %s: %.1f%%" % (sname, auc*100))
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()


if __name__ == "__main__":
    main()
