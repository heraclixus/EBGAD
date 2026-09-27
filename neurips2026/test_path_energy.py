"""Compare path_energy vs control_energy vs magnitude on best known hyperparams."""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse, json, sys, time
import numpy as np
import torch
from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data

BEST_CONFIGS = {
    "disney":      dict(rho=1.0,  lam_penalty=100,  template_type="high",     graph_type="original", kappa=5.0,  gamma=1.0, d_hidden=256, n_hidden_layers=3),
    "enron":       dict(rho=1.0,  lam_penalty=100,  template_type="low",      graph_type="affinity", kappa=1.0,  gamma=1.0, d_hidden=256, n_hidden_layers=2),
    "weibo":       dict(rho=1.0,  lam_penalty=30,   template_type="high",     graph_type="original", kappa=2.0,  gamma=0.5, d_hidden=64,  n_hidden_layers=2),
    "reddit":      dict(rho=1.0,  lam_penalty=100,  template_type="low",      graph_type="affinity", kappa=1.0,  gamma=1.0, d_hidden=64,  n_hidden_layers=2),
    "amazon":      dict(rho=0.7,  lam_penalty=50,   template_type="zero",     graph_type="original", kappa=1.0,  gamma=0.5, d_hidden=128, n_hidden_layers=2),
    "yelpchi":     dict(rho=1.0,  lam_penalty=10,   template_type="low",      graph_type="original", kappa=5.0,  gamma=0.5, d_hidden=128, n_hidden_layers=2),
    "blogcatalog": dict(rho=0.25, lam_penalty=100,  template_type="affinity", graph_type="affinity", kappa=0.1,  gamma=0.2, d_hidden=128, n_hidden_layers=2),
    "facebook":    dict(rho=0.25, lam_penalty=5,    template_type="affinity", graph_type="original", kappa=10.0, gamma=0.5, d_hidden=64,  n_hidden_layers=2),
    "acm":         dict(rho=1.0,  lam_penalty=2000, template_type="affinity", graph_type="affinity", kappa=0.1,  gamma=0.2, d_hidden=128, n_hidden_layers=2),
    "t_finance":   dict(rho=0.75, lam_penalty=100,  template_type="affinity", graph_type="original", kappa=1.0,  gamma=0.5, d_hidden=128, n_hidden_layers=2),
    "elliptic":    dict(rho=1.0,  lam_penalty=50,   template_type="zero",     graph_type="original", kappa=5.0,  gamma=0.5, d_hidden=64,  n_hidden_layers=2),
    "elliptic_plus_plus": dict(rho=0.5, lam_penalty=10, template_type="zero", graph_type="original", kappa=5.0,  gamma=0.5, d_hidden=64,  n_hidden_layers=2),
    "dgraph":      dict(rho=0.0,  lam_penalty=10,   template_type="zero",     graph_type="original", kappa=1.0,  gamma=0.5, d_hidden=256, n_hidden_layers=3, labeled_only=True),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--methods", type=str, default="path_energy_v2,path_energy,control_energy,magnitude",
                        help="Comma-separated scoring methods")
    args = parser.parse_args()

    ds = args.dataset.lower()
    if ds not in BEST_CONFIGS:
        print(f"Unknown dataset: {ds}. Available: {list(BEST_CONFIGS.keys())}")
        sys.exit(1)

    best = BEST_CONFIGS[ds]
    data = load_data(ds if ds != "yelpchi" else "YelpChi")
    methods = [m.strip() for m in args.methods.split(",")]

    results = []
    for method in methods:
        aucs = []
        t0 = time.time()
        for trial in range(args.trials):
            torch.manual_seed(trial)
            np.random.seed(trial)
            cfg = SOCGADConfig(
                dataset=ds, **best,
                lr=0.001, epochs=args.epochs,
                score_method=method, score_K=20,
            )
            r = evaluate_single_trial(data, cfg, device=args.device)
            aucs.append(r.auc)
            print(f"  {method} trial {trial}: AUC={r.auc:.4f}", flush=True)
        elapsed = time.time() - t0
        rec = {
            "dataset": ds, "score_method": method,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "trials": aucs, "elapsed_s": elapsed,
            "params": {**best, "epochs": args.epochs, "score_method": method},
        }
        results.append(rec)
        print(f"{ds:15s} {method:16s}  AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}  ({elapsed:.0f}s)")

    if args.output:
        with open(args.output, "w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
