"""Prior optimizer v4: Data-driven kappa bounds from variance matching.

Key changes vs v3:
- Uses compute_kappa_bounds() for principled kappa upper bound per iteration
- Sweeps multiple bound percentiles [25, 50, 75] to explore loose-to-tight range
- Adds Amazon and BlogCatalog to the default dataset list
- Finer gamma grid
- Saves per-dataset JSON immediately (crash-safe)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, Q_rho_eigenvalues,
    compute_kappa_bounds,
)
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc
from soc.soc_anomaly import compute_anomaly_scores

import dataclasses, itertools


def run_dataset(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    discrete_choices = []
    for t in ["zero", "low", "high", "affinity"]:
        for g in ["original", "affinity"]:
            for gamma in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0]:
                for norm in ["zscore", "minmax"]:
                    discrete_choices.append({
                        "template_type": t, "graph_type": g,
                        "gamma": gamma, "normalize_mode": norm,
                    })

    best_ce = 0; best_cr = 0
    best_ce_cfg = {}; best_cr_cfg = {}

    for i, disc in enumerate(discrete_choices):
        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5,
            lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type=disc["template_type"],
            graph_type=disc["graph_type"],
            gamma=disc["gamma"],
            normalize_mode=disc["normalize_mode"],
            laplacian_variant="sym",
            d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=device)
        except:
            continue

        x_normed = trainer.normalize(data.x.to(device).float())
        template = trainer.template
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
        delta = (x_normed - template).cpu().numpy()
        S = compute_spectral_energy(delta, V)

        # Multi-start with data-driven bounds at different percentiles
        best_opt = None; best_opt_ml = -np.inf
        for rho_init in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for kappa_init in [0.1, 0.5, 1.0, 3.0]:
                for pctile in [25, 50, 75]:
                    try:
                        opt = coordinate_descent(
                            S, lam_L, n, d,
                            rho_init=rho_init, kappa_init=kappa_init,
                            n_iters=10,
                            use_data_driven_bounds=True,
                            bound_percentile=pctile,
                        )
                        if opt["marginal_likelihood"] > best_opt_ml:
                            best_opt_ml = opt["marginal_likelihood"]
                            best_opt = opt
                    except:
                        pass

                # Also try without bounds (v3 behavior) as fallback
                try:
                    opt = coordinate_descent(
                        S, lam_L, n, d,
                        rho_init=rho_init, kappa_init=kappa_init,
                        n_iters=10,
                        use_data_driven_bounds=False,
                    )
                    if opt["marginal_likelihood"] > best_opt_ml:
                        best_opt_ml = opt["marginal_likelihood"]
                        best_opt = opt
                except:
                    pass

        if best_opt is None:
            continue

        torch.manual_seed(0); np.random.seed(0)
        full_cfg = {
            "rho": best_opt["rho"], "kappa": best_opt["kappa"],
            "gamma": disc["gamma"], "alpha": best_opt["alpha_T"], "T": 1.0,
            "lam_penalty": best_opt["lam_penalty"],
            "template_type": disc["template_type"],
            "graph_type": disc["graph_type"],
            "normalize_mode": disc["normalize_mode"],
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 1, "parameterization": "score",
            "time_weighting": "importance",
            "laplacian_variant": "sym",
        }
        try:
            for method in ["precision_energy", "precision_ratio"]:
                eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                r = evaluate_single_trial(data, eval_cfg, device=device)
                if method == "precision_energy" and r.auc > best_ce:
                    best_ce = r.auc
                    best_ce_cfg = {**full_cfg, "score_method": "precision_energy",
                                   "kappa_bounds": list(best_opt.get("kappa_bounds", [0.01, 50.0]))}
                if method == "precision_ratio" and r.auc > best_cr:
                    best_cr = r.auc
                    best_cr_cfg = {**full_cfg, "score_method": "precision_ratio",
                                   "kappa_bounds": list(best_opt.get("kappa_bounds", [0.01, 50.0]))}
        except:
            pass

        if (i + 1) % 50 == 0:
            print("  [%d/%d] best_ce=%.1f%% best_cr=%.1f%%" %
                  (i + 1, len(discrete_choices), best_ce * 100, best_cr * 100), flush=True)

    best_overall = max(best_ce, best_cr)
    best_cfg = best_ce_cfg if best_ce >= best_cr else best_cr_cfg
    print("  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%)" % (best_overall * 100, best_ce * 100, best_cr * 100))
    return {"dataset": ds, "best_auc": best_overall,
            "best_ce": best_ce, "best_cr": best_cr,
            "best_config": best_cfg}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="enron,weibo,reddit,yelpchi,facebook,acm,amazon,blogcatalog,t_finance,elliptic,elliptic_plus_plus,dgraph")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_v4_configs_%s.json" % ds)
        if os.path.exists(out_file):
            print("=== %s === SKIP (already exists: %s)" % (ds, out_file))
            continue

        print("=== %s ===" % ds)
        result = run_dataset(ds, device=args.device)

        # Save immediately per dataset (crash-safe)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("  Saved to %s" % out_file)

    # Print summary of all available results
    print("\n=== SUMMARY ===")
    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_v4_configs_%s.json" % ds)
        if os.path.exists(out_file):
            with open(out_file) as f:
                r = json.load(f)
            print("%-18s  best=%.1f%%  CE=%.1f%%  CR=%.1f%%  kappa_bounds=%s" % (
                ds, r["best_auc"]*100, r["best_ce"]*100, r["best_cr"]*100,
                r["best_config"].get("kappa_bounds", "N/A")))
        else:
            print("%-18s  not ready" % ds)


if __name__ == "__main__":
    main()
