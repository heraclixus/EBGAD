"""Test: Fixed default config, ML only for rho/kappa.

Uses a single default config (template=low, graph=original, gamma=0.5, no PCA)
and optimizes only rho/kappa via ML. Reports J* and R AUROC.

This tests whether ML works for continuous parameter optimization
when discrete choices are fixed.
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

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Several fixed configs to test
FIXED_CONFIGS = [
    {"name": "default", "template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"name": "affinity_tmpl", "template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"name": "aff_graph", "template_type": "low", "graph_type": "affinity", "gamma": 0.5, "pca": None},
    {"name": "small_gamma", "template_type": "low", "graph_type": "original", "gamma": 0.1, "pca": None},
    {"name": "pca64", "template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": 64},
]


def evaluate_fixed(data, ds, cfg_spec, device="cpu"):
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, template_type=cfg_spec["template_type"],
        graph_type=cfg_spec["graph_type"], gamma=cfg_spec["gamma"],
        normalize_mode="zscore", prior_mean_mode="stationary",
        laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
        epochs=0, normalize_features=True,
    )
    if cfg_spec["pca"] is not None:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=cfg_spec["pca"], encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))

    try:
        trainer_cfg = SOCTrainerConfig(**cfg_kwargs)
        trainer = SOCTrainer(trainer_cfg)
        trainer.train(data, device=device)

        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl = trainer.template.cpu().numpy()
        x = data.x.float()
        if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
            x = trainer.normalizer(x)
        if hasattr(trainer, 'encoder') and trainer.encoder is not None:
            with torch.no_grad():
                x = trainer.encoder(x.to(device),
                                    data.edge_index.to(device)).cpu()
        x_np = x.numpy()
        n, d = x_np.shape
        delta = x_np - tmpl
        S = compute_spectral_energy(delta, V)

        # ML-optimize rho for each kappa
        best_ml = -np.inf
        best_rho, best_kappa = 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa

        # Also sweep rho manually to check
        best_auc_je_sweep = 0
        best_auc_jr_sweep = 0
        for rho in np.linspace(0, 1, 11):
            for kappa in KAPPAS:
                lam_Q = torch.from_numpy(
                    Q_rho_eigenvalues(lam_L, rho, kappa)).float()
                V_t = torch.from_numpy(V).float()
                tmpl_t = torch.from_numpy(tmpl).float()
                x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x_np).float()
                with torch.no_grad():
                    je = precision_energy_anomaly(
                        x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                        prior_mean_mode="stationary").cpu().numpy()
                    jr = precision_ratio_anomaly(
                        x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                        prior_mean_mode="stationary").cpu().numpy()
                y = data.y.numpy()
                mask = (y >= 0) & (y <= 1)
                auc_je = roc_auc_score(y[mask], je[mask])
                auc_jr = roc_auc_score(y[mask], jr[mask])
                best_auc_je_sweep = max(best_auc_je_sweep, auc_je)
                best_auc_jr_sweep = max(best_auc_jr_sweep, auc_jr)

        # Score at ML-optimal rho/kappa
        lam_Q = torch.from_numpy(
            Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t = torch.from_numpy(V).float()
        tmpl_t = torch.from_numpy(tmpl).float()
        x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x_np).float()
        with torch.no_grad():
            je = precision_energy_anomaly(
                x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(
                x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                prior_mean_mode="stationary").cpu().numpy()
        y = data.y.numpy()
        mask = (y >= 0) & (y <= 1)

        return {
            "ml_je": float(roc_auc_score(y[mask], je[mask])),
            "ml_jr": float(roc_auc_score(y[mask], jr[mask])),
            "oracle_je": float(best_auc_je_sweep),
            "oracle_jr": float(best_auc_jr_sweep),
            "rho": float(best_rho),
            "kappa": float(best_kappa),
            "ml": float(best_ml),
        }
    except Exception as e:
        return {"error": str(e)[:80]}


def main(device="cpu"):
    datasets = ["enron", "weibo", "reddit", "amazon", "yelpchi",
                "blogcatalog", "facebook", "acm",
                "elliptic", "elliptic_plus_plus", "t_finance"]

    all_results = {}
    for ds in datasets:
        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)
        data = load_data(load_name)
        print(f"\n=== {ds} ({data.num_nodes} nodes, {data.x.shape[1]} feat) ===",
              flush=True)

        ds_results = {}
        for cfg in FIXED_CONFIGS:
            # Skip PCA if features are already low-dim
            if cfg["pca"] is not None and data.x.shape[1] <= cfg["pca"]:
                continue

            result = evaluate_fixed(data, ds, cfg, device)
            if "error" not in result:
                ml_best = max(result["ml_je"], result["ml_jr"])
                oracle_best = max(result["oracle_je"], result["oracle_jr"])
                gap = ml_best - oracle_best
                print(f"  {cfg['name']:15s}  ML:J*={result['ml_je']*100:5.1f}% R={result['ml_jr']*100:5.1f}%  "
                      f"Oracle:J*={result['oracle_je']*100:5.1f}% R={result['oracle_jr']*100:5.1f}%  "
                      f"gap={gap*100:+5.1f}pp  rho={result['rho']:.2f} kappa={result['kappa']}")
                ds_results[cfg["name"]] = result

        all_results[ds] = ds_results
        with open("results/fixed_config_ml_test.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n=== SUMMARY: ML-optimal rho vs oracle rho (within fixed config) ===")
    print(f"{'Dataset':15s}  {'Config':15s}  {'ML best':>8s}  {'Oracle':>8s}  {'Gap':>6s}")
    for ds, ds_results in all_results.items():
        for cfg_name, r in ds_results.items():
            if "error" in r:
                continue
            ml_b = max(r["ml_je"], r["ml_jr"]) * 100
            or_b = max(r["oracle_je"], r["oracle_jr"]) * 100
            print(f"{ds:15s}  {cfg_name:15s}  {ml_b:7.1f}%  {or_b:7.1f}%  {ml_b-or_b:+5.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.device)
