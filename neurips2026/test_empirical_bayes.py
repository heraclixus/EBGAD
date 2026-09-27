"""Test empirical Bayes hypothesis: does lowest training loss predict best AUC?

For each dataset, sweep prior configs, record training loss AND scoring AUC
for both CE and CE ratio. Then check correlation.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse, json, sys, time, os
import numpy as np
import torch
from soc.soc_gad import SOCGADConfig
from soc.soc_trainer import SOCTrainer
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc

sys.stdout.reconfigure(line_buffering=True)


# Compact sweep grids per dataset — focus on prior knobs
GRIDS = {
    "enron": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.1, 1.0, 5.0, 10.0],
        "gamma": [0.2, 0.5, 1.0, 2.0],
        "template_type": ["zero", "low", "high", "affinity"],
        "graph_type": ["original", "affinity"],
    },
    "weibo": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.1, 1.0, 5.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low", "high"],
        "graph_type": ["original"],
    },
    "reddit": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.1, 1.0, 5.0],
        "gamma": [0.5, 1.0, 2.0],
        "template_type": ["zero", "low", "high", "affinity"],
        "graph_type": ["original", "affinity"],
    },
    "yelpchi": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.1, 1.0, 5.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low", "high"],
        "graph_type": ["original"],
    },
    "facebook": {
        "rho": [0.0, 0.25, 0.5, 1.0],
        "kappa": [1.0, 5.0, 10.0, 20.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low", "high", "affinity"],
        "graph_type": ["original"],
    },
    "elliptic": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.01, 0.1, 1.0, 5.0],
        "gamma": [0.1, 0.5, 1.0],
        "template_type": ["zero", "low", "high"],
        "graph_type": ["original"],
    },
    "t_finance": {
        "rho": [0.0, 0.5, 0.75, 1.0],
        "kappa": [0.1, 1.0, 5.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low", "affinity"],
        "graph_type": ["original"],
    },
    "acm": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.01, 0.1, 1.0],
        "gamma": [0.1, 0.2, 0.5],
        "template_type": ["zero", "low", "affinity"],
        "graph_type": ["original", "affinity"],
    },
}

# Fixed training params
FIXED = dict(
    d_hidden=64, n_hidden_layers=2, lr=0.001, epochs=200, patience=50,
    parameterization="score", time_weighting="importance",
    lam_penalty=50, alpha=2.0, laplacian_variant="sym",
)


def grid_product(grid):
    import itertools
    keys = sorted(grid.keys())
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ds = args.dataset.lower()
    if ds not in GRIDS:
        print("Available: %s" % list(GRIDS.keys()))
        sys.exit(1)

    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    configs = list(grid_product(GRIDS[ds]))
    print("Dataset: %s | %d configs" % (ds, len(configs)))

    out_path = args.output or "results/empirical_bayes_%s.jsonl" % ds
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "w")

    for i, prior in enumerate(configs):
        torch.manual_seed(0)
        np.random.seed(0)

        cfg = SOCGADConfig(dataset=ds, score_method="control_energy", score_K=1,
                           **FIXED, **prior)
        trainer_cfg = cfg.to_trainer_config()
        trainer = SOCTrainer(trainer_cfg)

        # Train and capture best loss
        trainer.train(data, device=args.device)
        bl = trainer.best_loss if hasattr(trainer, 'best_loss') else float('inf')
        best_loss = bl if np.isfinite(bl) else 1e30  # json can't handle inf/nan

        # Score with both CE and CE ratio
        eval_data = data
        x_test = eval_data.x.to(args.device).float()

        ce_scores = compute_anomaly_scores(trainer, x_test, K=1, method="control_energy")
        cr_scores = compute_anomaly_scores(trainer, x_test, K=1, method="ce_ratio")

        y_ce, s_ce = mask_labeled(eval_data.y, ce_scores.cpu().detach())
        y_cr, s_cr = mask_labeled(eval_data.y, cr_scores.cpu().detach())

        auc_ce = eval_roc_auc(y_ce, s_ce) if y_ce is not None else 0.0
        auc_cr = eval_roc_auc(y_cr, s_cr) if y_cr is not None else 0.0

        rec = {
            "dataset": ds,
            "prior": prior,
            "best_loss": float(best_loss),
            "auc_ce": float(auc_ce),
            "auc_ce_ratio": float(auc_cr),
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

        if (i + 1) % 20 == 0 or i == len(configs) - 1:
            print("[%d/%d] loss=%.4f  auc_ce=%.3f  auc_cr=%.3f  %s" %
                  (i + 1, len(configs), best_loss, auc_ce, auc_cr, prior))

    fout.close()
    print("Saved to %s" % out_path)

    # Quick correlation analysis
    import json as j
    records = [j.loads(l) for l in open(out_path)]
    losses = [r["best_loss"] for r in records if not np.isnan(r["best_loss"])]
    aucs_ce = [r["auc_ce"] for r in records if not np.isnan(r["best_loss"])]
    aucs_cr = [r["auc_ce_ratio"] for r in records if not np.isnan(r["best_loss"])]

    if len(losses) > 5:
        from scipy.stats import spearmanr
        corr_ce, p_ce = spearmanr(losses, aucs_ce)
        corr_cr, p_cr = spearmanr(losses, aucs_cr)
        print("\nSpearman correlation (loss vs AUC):")
        print("  CE:       r=%.3f  p=%.4f  (negative = low loss predicts high AUC)" % (corr_ce, p_ce))
        print("  CE ratio: r=%.3f  p=%.4f" % (corr_cr, p_cr))

        # Best-loss config
        best_idx = np.argmin(losses)
        print("\nBest-loss config: loss=%.4f  auc_ce=%.3f  auc_cr=%.3f" %
              (losses[best_idx], aucs_ce[best_idx], aucs_cr[best_idx]))
        print("Best-AUC(CE) config: auc=%.3f  loss=%.4f" %
              (max(aucs_ce), losses[np.argmax(aucs_ce)]))
        print("Best-AUC(CR) config: auc=%.3f  loss=%.4f" %
              (max(aucs_cr), losses[np.argmax(aucs_cr)]))


if __name__ == "__main__":
    main()
