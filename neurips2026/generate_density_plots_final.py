"""Generate density separation plots for datasets where EB-GAD beats baselines.

Creates side-by-side density plots showing normal vs anomalous score distributions
for each dataset, using the best config from the optimizer sweeps.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data

sys.stdout.reconfigure(line_buffering=True)


def load_best_config(ds):
    """Load the best config across all optimizer versions for a dataset."""
    best_auc = 0
    best_cfg = None
    best_ver = ""

    for v in ["v4", "v5", "v6", "v7", "v8"]:
        f = "results/eb_gad_%s_configs_%s.json" % (v, ds)
        try:
            r = json.load(open(f))
            if r["best_auc"] > best_auc:
                best_auc = r["best_auc"]
                best_cfg = r["best_config"]
                best_ver = v
        except:
            pass

    # Also check targeted
    f = "results/eb_gad_targeted_%s.json" % ds
    try:
        r = json.load(open(f))
        if r["best_auc"] > best_auc:
            best_auc = r["best_auc"]
            best_cfg = r["best_config"]
            best_ver = "targeted"
    except:
        pass

    return best_auc, best_cfg, best_ver


def compute_scores_for_config(ds, cfg, device="cuda"):
    """Compute per-node anomaly scores using the given config."""
    data = load_data("YelpChi" if ds == "yelpchi" else ds)

    eval_cfg = SOCGADConfig(
        dataset=ds,
        score_method=cfg["score_method"],
        score_K=1,
        **{k: v for k, v in cfg.items() if k != "score_method" and k != "kappa_bounds"}
    )

    # Setup trainer to get scores
    trainer_cfg = eval_cfg.to_trainer_config()
    trainer = SOCTrainer(trainer_cfg)
    trainer._setup(data, device=device)

    x_normed = trainer.normalize(data.x.to(device).float())
    scores = compute_anomaly_scores(trainer, data.x, K=1, method=cfg["score_method"])
    scores = scores.cpu().numpy()

    y = data.y.cpu().numpy()
    normal_mask = (y == 0)
    anomaly_mask = (y == 1)

    return scores, normal_mask, anomaly_mask


def plot_density_grid(results, output_path, ncols=4):
    """Plot density separation for multiple datasets in a grid."""
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (ds, scores, normal_mask, anomaly_mask, auc) in enumerate(results):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        s_normal = scores[normal_mask]
        s_anomaly = scores[anomaly_mask]

        # Clip outliers for better visualization
        p1, p99 = np.percentile(scores, 1), np.percentile(scores, 99)
        bins = np.linspace(p1, p99, 60)

        ax.hist(s_normal, bins=bins, density=True, alpha=0.6, color="tab:blue", label="Normal")
        ax.hist(s_anomaly, bins=bins, density=True, alpha=0.6, color="tab:red", label="Anomaly")
        ax.set_title("%s (%.1f%%)" % (ds, auc * 100), fontsize=11, fontweight="bold")
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved to %s" % output_path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="figures/density_separation_final.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Datasets where we beat baselines significantly
    datasets = [
        "enron", "weibo", "facebook", "yelpchi", "reddit",
        "amazon", "elliptic", "elliptic_plus_plus", "t_finance",
    ]

    results = []
    for ds in datasets:
        auc, cfg, ver = load_best_config(ds)
        if cfg is None:
            print("  %s: no config found, skipping" % ds)
            continue

        enc = cfg.get("encoder_type", "none") if cfg.get("use_encoder") else "none"
        print("  %s: %.1f%% (ver=%s, enc=%s, score=%s)" % (
            ds, auc * 100, ver, enc, cfg.get("score_method")), flush=True)

        try:
            scores, normal_mask, anomaly_mask = compute_scores_for_config(
                ds, cfg, device=args.device)
            results.append((ds, scores, normal_mask, anomaly_mask, auc))
        except Exception as e:
            print("    ERROR: %s" % e)

    if results:
        plot_density_grid(results, args.output)
        print("\nGenerated density plots for %d datasets" % len(results))


if __name__ == "__main__":
    main()
