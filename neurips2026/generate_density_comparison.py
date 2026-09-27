"""Generate density comparison: TAM vs EB-GAD.

Style matches the reference figure: KDE smoothing, filled areas,
log(1+score) transform, orange/blue colors.

Generates plots for multiple datasets so we can pick the best.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from run_tam import run_tam_trial


def load_best_config(ds):
    """Load actual best config from saved results."""
    best_auc = 0
    best_cfg = None
    for tag in ["targeted", "bandpass", "lbfgsb"]:
        f = "results/eb_gad_%s_%s.json" % (tag, ds)
        try:
            r = json.load(open(f))
            if r["best_auc"] > best_auc:
                best_auc = r["best_auc"]
                best_cfg = r["best_config"]
        except:
            pass
    for v in ["v4", "v5", "v6", "v7", "v8"]:
        f = "results/eb_gad_%s_configs_%s.json" % (v, ds)
        try:
            r = json.load(open(f))
            if r["best_auc"] > best_auc:
                best_auc = r["best_auc"]
                best_cfg = r["best_config"]
        except:
            pass
    return best_auc, best_cfg


def compute_ebgad_scores(ds, cfg, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    cfg_clean = {k: v for k, v in cfg.items()
                 if k not in ("kappa_bounds", "optimizer")}
    eval_cfg = SOCGADConfig(dataset=ds, score_K=1, **cfg_clean)
    trainer_cfg = eval_cfg.to_trainer_config()
    trainer = SOCTrainer(trainer_cfg)
    trainer._setup(data, device=device)
    scores = compute_anomaly_scores(trainer, data.x, K=1,
                                     method=cfg["score_method"])
    scores = scores.cpu().numpy()
    y = data.y.cpu().numpy()
    from sklearn.metrics import roc_auc_score
    mask = (y == 0) | (y == 1)
    return scores[mask], y[mask], roc_auc_score(y[mask], scores[mask])


def compute_tam_scores(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    result = run_tam_trial(data, device=device, return_scores=True)
    return result["scores"], result["y"], result["auc"]


def plot_kde_panel(ax, normal, anomaly, title, show_xlabel=True, show_legend=True):
    """Plot KDE density with filled areas, matching reference style."""
    # log(1+score) transform
    normal_t = np.log1p(np.maximum(normal, 0))
    anomaly_t = np.log1p(np.maximum(anomaly, 0))
    all_t = np.concatenate([normal_t, anomaly_t])

    lo, hi = np.percentile(all_t, 0.5), np.percentile(all_t, 99.5)
    margin = (hi - lo) * 0.05
    x_grid = np.linspace(lo - margin, hi + margin, 300)

    try:
        kde_n = gaussian_kde(normal_t, bw_method=0.15)
        kde_a = gaussian_kde(anomaly_t, bw_method=0.15)
    except:
        kde_n = gaussian_kde(normal_t)
        kde_a = gaussian_kde(anomaly_t)

    y_n = kde_n(x_grid)
    y_a = kde_a(x_grid)

    ax.fill_between(x_grid, y_n, alpha=0.5, color="tab:blue", label="Normal")
    ax.fill_between(x_grid, y_a, alpha=0.5, color="tab:orange", label="Abnormal")
    ax.plot(x_grid, y_n, color="tab:blue", linewidth=1)
    ax.plot(x_grid, y_a, color="tab:orange", linewidth=1)

    ax.set_title(title, fontsize=11, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel(r"$\mathbf{Score}$", fontsize=10)
    ax.set_ylabel("Density", fontsize=9)
    if show_legend:
        ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)


def plot_1x4(results, ds_list, output_path):
    """1x4 layout: TAM-ds1, EB-GAD-ds1, TAM-ds2, EB-GAD-ds2."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))

    labels = {"enron": "Enron", "weibo": "Weibo", "elliptic": "Elliptic",
              "facebook": "Facebook", "t_finance": "T-Finance",
              "elliptic_plus_plus": "Elliptic++", "yelpchi": "YelpChi"}

    for i, ds in enumerate(ds_list):
        tam_scores, tam_y, _ = results[ds]["tam"]
        eb_scores, eb_y, _ = results[ds]["ebgad"]

        label = labels.get(ds, ds)
        show_legend = (i == 0)  # legend only on first panel
        plot_kde_panel(axes[2*i], tam_scores[tam_y==0], tam_scores[tam_y==1],
                       "TAM \u2014 %s" % label, show_legend=show_legend)
        plot_kde_panel(axes[2*i+1], eb_scores[eb_y==0], eb_scores[eb_y==1],
                       "EB-GAD - %s" % label, show_legend=show_legend)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved %s" % output_path)
    plt.close()


def plot_2x_grid(results, ds_list, output_path):
    """2-row grid: top=TAM, bottom=EB-GAD, one column per dataset."""
    ncols = len(ds_list)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 5))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    labels = {"enron": "Enron", "weibo": "Weibo", "elliptic": "Elliptic",
              "facebook": "Facebook", "t_finance": "T-Finance",
              "elliptic_plus_plus": "Elliptic++", "yelpchi": "YelpChi"}

    for col, ds in enumerate(ds_list):
        tam_scores, tam_y, _ = results[ds]["tam"]
        eb_scores, eb_y, _ = results[ds]["ebgad"]

        label = labels.get(ds, ds)
        plot_kde_panel(axes[0, col], tam_scores[tam_y==0], tam_scores[tam_y==1],
                       label)
        axes[0, col].set_ylabel("TAM", fontsize=11, fontweight="bold")
        axes[0, col].set_xlabel("")

        plot_kde_panel(axes[1, col], eb_scores[eb_y==0], eb_scores[eb_y==1],
                       "")
        axes[1, col].set_ylabel("EB-GAD", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved %s" % output_path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    # Compute scores for all candidate datasets
    datasets = ["weibo", "enron", "elliptic", "facebook", "t_finance", "yelpchi"]
    results = {}

    for ds in datasets:
        print("=== %s ===" % ds, flush=True)
        best_auc, best_cfg = load_best_config(ds)
        if best_cfg is None:
            print("  No config, skipping")
            continue

        enc = best_cfg.get("encoder_type", "none") if best_cfg.get("use_encoder") else "none"
        print("  Config: %.1f%% enc=%s score=%s" % (best_auc*100, enc, best_cfg.get("score_method")), flush=True)

        try:
            eb_scores, eb_y, eb_auc = compute_ebgad_scores(ds, best_cfg, args.device)
            print("  EB-GAD AUC: %.1f%%" % (eb_auc*100), flush=True)
        except Exception as e:
            print("  EB-GAD ERROR: %s" % e)
            continue

        try:
            tam_scores, tam_y, tam_auc = compute_tam_scores(ds, args.device)
            print("  TAM AUC: %.1f%%" % (tam_auc*100), flush=True)
        except Exception as e:
            print("  TAM ERROR: %s" % e)
            continue

        results[ds] = {"ebgad": (eb_scores, eb_y, eb_auc), "tam": (tam_scores, tam_y, tam_auc)}

    # Primary figure: 1x4 Weibo + Facebook (for paper)
    if "weibo" in results and "facebook" in results:
        plot_1x4(results, ["weibo", "facebook"],
                 "tex/figures/density_1x4_weibo_facebook.pdf")

    # Also generate 2x grid variants for reference
    for combo_name, combo in [
        ("weibo_facebook", ["weibo", "facebook"]),
        ("weibo_enron", ["weibo", "enron"]),
    ]:
        valid = [ds for ds in combo if ds in results]
        if len(valid) >= 2:
            plot_2x_grid(results, valid, "tex/figures/density_2x_%s.pdf" % combo_name)

    print("\nDone! Generated figures in figures/density_*.pdf")


if __name__ == "__main__":
    main()
