"""Quick ablation: how much does each discrete choice matter?

For each dataset, using the sweep-best config:
1. Fix everything, swap J* <-> R: what's the AUROC gap?
2. Fix everything, swap original <-> affinity graph: what's the gap?

This tells us which choice matters more and where.
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

# Best configs from sweeps
BEST = {
    "enron":    {"template_type": "high", "graph_type": "affinity", "gamma": 0.1, "pca": None},
    "weibo":    {"template_type": "low", "graph_type": "original", "gamma": 0.2, "pca": 64},
    "reddit":   {"template_type": "affinity", "graph_type": "original", "gamma": 2.0, "pca": None},
    "amazon":   {"template_type": "low", "graph_type": "original", "gamma": 5.0, "pca": 64},
    "yelpchi":  {"template_type": "low", "graph_type": "original", "gamma": 1.0, "pca": 64},
    "blogcatalog": {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": 64},
    "facebook": {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": 16},
    "acm":      {"template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": 128},
    "elliptic": {"template_type": "affinity", "graph_type": "original", "gamma": 0.7, "pca": None},
    "elliptic_plus_plus": {"template_type": "affinity", "graph_type": "original", "gamma": 0.7, "pca": None},
    "t_finance": {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": None},
}


def eval_config(data, cfg_dict, device="cpu"):
    """Evaluate one config: ML-optimize rho, return J* and R AUROC."""
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        **{k: v for k, v in cfg_dict.items() if k != "pca"},
    )
    if cfg_dict.get("pca") and data.x.shape[1] > cfg_dict["pca"]:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=cfg_dict["pca"], encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))

    try:
        trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl = trainer.template.cpu().numpy()
        x = data.x.float()
        if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
            x = trainer.normalizer(x)
        if hasattr(trainer, 'encoder') and trainer.encoder is not None:
            with torch.no_grad():
                x = trainer.encoder(x.to(device), data.edge_index.to(device)).cpu()
        x_np = x.numpy()
        n, d = x_np.shape
        delta = x_np - tmpl
        S = compute_spectral_energy(delta, V)

        best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t, tmpl_t = torch.from_numpy(V).float(), torch.from_numpy(tmpl).float()
        x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x_np).float()

        with torch.no_grad():
            je = precision_energy_anomaly(x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()

        y = data.y.numpy()
        mask = (y >= 0) & (y <= 1)
        return {
            "je": float(roc_auc_score(y[mask], je[mask])),
            "jr": float(roc_auc_score(y[mask], jr[mask])),
            "rho": best_rho, "kappa": best_kappa,
        }
    except Exception as e:
        return {"error": str(e)[:60]}


def main(device="cpu"):
    datasets = list(BEST.keys())

    print(f"{'Dataset':15s}  {'Best J*':>7s} {'Best R':>7s} {'|J*-R|':>6s}  "
          f"{'Orig J*':>7s} {'Orig R':>7s} {'Aff J*':>7s} {'Aff R':>6s} {'|O-A|':>6s}")

    results = {}
    for ds in datasets:
        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)
        data = load_data(load_name)

        cfg = dict(BEST[ds])

        # 1. Best config: get J* and R
        best = eval_config(data, cfg, device)
        if "error" in best:
            print(f"{ds:15s}  ERROR: {best['error']}")
            continue

        # 2. Swap graph type
        cfg_swap = dict(cfg)
        cfg_swap["graph_type"] = "affinity" if cfg["graph_type"] == "original" else "original"
        swap = eval_config(data, cfg_swap, device)

        if "error" in swap:
            swap = {"je": 0, "jr": 0}

        score_gap = abs(best["je"] - best["jr"]) * 100
        best_orig = best if cfg["graph_type"] == "original" else swap
        best_aff = swap if cfg["graph_type"] == "original" else best
        graph_gap = abs(max(best_orig["je"], best_orig["jr"]) -
                       max(best_aff["je"], best_aff["jr"])) * 100

        print(f"{ds:15s}  "
              f"{best['je']*100:6.1f}% {best['jr']*100:6.1f}% {score_gap:5.1f}  "
              f"{best_orig['je']*100:6.1f}% {best_orig['jr']*100:6.1f}% "
              f"{best_aff['je']*100:6.1f}% {best_aff['jr']*100:6.1f}% {graph_gap:5.1f}")

        results[ds] = {
            "best_je": best["je"], "best_jr": best["jr"],
            "score_gap": score_gap,
            "orig": best_orig, "aff": best_aff,
            "graph_gap": graph_gap,
            "best_graph": cfg["graph_type"],
        }

    with open("results/ablation_gaps.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== SUMMARY ===")
    score_gaps = [r["score_gap"] for r in results.values()]
    graph_gaps = [r["graph_gap"] for r in results.values()]
    print(f"Score gap (J* vs R):     median={np.median(score_gaps):.1f}pp  "
          f"mean={np.mean(score_gaps):.1f}pp  max={np.max(score_gaps):.1f}pp")
    print(f"Graph gap (orig vs aff): median={np.median(graph_gaps):.1f}pp  "
          f"mean={np.mean(graph_gaps):.1f}pp  max={np.max(graph_gaps):.1f}pp")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.device)
