"""Generate prior selection diagnostic figures.

1. Spectral energy distribution S_j vs lambda_j — annotated by winning score
2. Feature homophily per dataset — annotated by winning template
3. PCA explained variance — annotated by whether PCA helps

These justify the practical guide in §4.2.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import compute_spectral_energy
from data_utils import load_data


# Best config metadata
BEST_INFO = {
    "enron":    {"score": "R", "template": "high", "proj": "sage", "d": 18},
    "weibo":    {"score": "J*", "template": "low", "proj": "pca", "d": 400},
    "reddit":   {"score": "J*", "template": "low", "proj": "mlp", "d": 64},
    "amazon":   {"score": "J*", "template": "low", "proj": "pca", "d": 25},
    "yelpchi":  {"score": "J*", "template": "low", "proj": "pca", "d": 32},
    "blogcatalog": {"score": "J*", "template": "low", "proj": "pca", "d": 8189},
    "facebook": {"score": "R", "template": "low", "proj": "pca", "d": 576},
    "acm":      {"score": "C", "template": "low", "proj": "pca", "d": 8337},
    "elliptic": {"score": "R", "template": "low", "proj": "none", "d": 166},
    "elliptic_plus_plus": {"score": "R", "template": "low", "proj": "none", "d": 182},
    "t_finance": {"score": "R", "template": "low", "proj": "none", "d": 10},
}

LABELS = {"enron": "Enron", "weibo": "Weibo", "reddit": "Reddit",
          "amazon": "Amazon", "yelpchi": "YelpChi", "blogcatalog": "BlogCat",
          "facebook": "Facebook", "acm": "ACM", "elliptic": "Elliptic",
          "elliptic_plus_plus": "Elliptic++", "t_finance": "T-Finance"}


def compute_homophily(data):
    """Mean cosine similarity between connected nodes."""
    x = data.x.float()
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    src, dst = data.edge_index[0], data.edge_index[1]

    # Sample if too many edges
    if len(src) > 500000:
        idx = np.random.choice(len(src), 500000, replace=False)
        src, dst = src[idx], dst[idx]

    sims = (x_norm[src] * x_norm[dst]).sum(dim=1)
    return float(sims.mean()), float(sims.std())


def compute_spectral_concentration(S):
    """Gini coefficient of spectral energy — 0=uniform, 1=concentrated."""
    s = np.sort(S)
    n = len(s)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * s) / (n * np.sum(s))) - (n + 1) / n)


def plot_spectral_energy_annotated(all_data, output_dir):
    """Plot S_j/d vs lambda_j, colored by winning score type."""
    score_colors = {"J*": "tab:blue", "R": "tab:red", "C": "tab:green"}
    datasets = [ds for ds in LABELS if ds in all_data]

    ncols = min(4, len(datasets))
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows))
    axes = np.atleast_2d(axes)

    for idx, ds in enumerate(datasets):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        info = all_data[ds]
        score = BEST_INFO.get(ds, {}).get("score", "J*")
        color = score_colors.get(score, "gray")

        ax.scatter(info["lam_L"], info["S"] / info["d"], s=6, alpha=0.5, c=color)
        ax.set_title("%s (best: %s)" % (LABELS[ds], score), fontsize=10, fontweight="bold")
        ax.set_xlabel(r"$\lambda_j$", fontsize=9)
        if col == 0:
            ax.set_ylabel(r"$S_j/d$", fontsize=9)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Hide unused
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "spectral_energy_annotated.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def plot_homophily_vs_template(homophily_data, output_dir):
    """Bar plot of homophily per dataset, colored by winning template."""
    datasets = sorted(homophily_data.keys(), key=lambda ds: -homophily_data[ds][0])
    vals = [homophily_data[ds][0] for ds in datasets]
    colors = ["tab:orange" if BEST_INFO.get(ds, {}).get("template") == "high"
              else "tab:blue" for ds in datasets]
    labels_ds = [LABELS.get(ds, ds) for ds in datasets]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    bars = ax.bar(range(len(datasets)), vals, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(labels_ds, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean neighbor cosine similarity", fontsize=10)
    ax.set_title("Feature homophily by dataset", fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color="tab:blue"), Patch(color="tab:orange")],
              ["Low-pass template wins", "High-pass template wins"],
              fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "homophily_vs_template.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def plot_pca_variance(pca_data, output_dir):
    """Cumulative explained variance, colored by whether PCA helps."""
    datasets = [ds for ds in LABELS if ds in pca_data]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for ds in datasets:
        info = pca_data[ds]
        proj = BEST_INFO.get(ds, {}).get("proj", "none")
        style = "-" if proj == "pca" else "--"
        color = "tab:blue" if proj == "pca" else ("tab:red" if proj == "none" else "tab:green")
        cumvar = info["cumvar"]
        k = min(100, len(cumvar))
        ax.plot(range(1, k+1), cumvar[:k], style, linewidth=1.5, alpha=0.7,
                label="%s (d=%d, %s)" % (LABELS[ds], info["d"], proj))

    ax.axhline(y=0.9, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of PCA components", fontsize=10)
    ax.set_ylabel("Cumulative explained variance", fontsize=10)
    ax.set_title("PCA explained variance by dataset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(output_dir, "pca_variance.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def plot_summary_scatter(homophily_data, spectral_data, output_dir):
    """Scatter: homophily vs spectral concentration, annotated by winning choices."""
    datasets = [ds for ds in LABELS if ds in homophily_data and ds in spectral_data]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    score_markers = {"J*": "o", "R": "s", "C": "D"}
    score_colors = {"J*": "tab:blue", "R": "tab:red", "C": "tab:green"}

    for ds in datasets:
        h = homophily_data[ds][0]
        g = spectral_data[ds]
        score = BEST_INFO.get(ds, {}).get("score", "J*")
        ax.scatter(h, g, s=80, marker=score_markers.get(score, "o"),
                   c=score_colors.get(score, "gray"), edgecolors="black", linewidth=0.5,
                   zorder=5)
        ax.annotate(LABELS[ds], (h, g), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
                              markersize=8, label='$J^*$ wins'),
                       Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:red',
                              markersize=8, label='$R$ wins'),
                       Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:green',
                              markersize=8, label='$C$ wins')]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Feature homophily (mean neighbor cosine sim)", fontsize=10)
    ax.set_ylabel("Spectral energy concentration (Gini)", fontsize=10)
    ax.set_title("Dataset characteristics vs.\ optimal scoring mode", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "score_selection_scatter.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved %s" % path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="tex/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["enron", "weibo", "reddit", "amazon", "yelpchi", "blogcatalog",
                "facebook", "acm", "elliptic", "elliptic_plus_plus", "t_finance"]

    spectral_all = {}
    homophily_all = {}
    pca_all = {}
    gini_all = {}

    for ds in datasets:
        print("=== %s ===" % ds, flush=True)
        data = load_data("YelpChi" if ds == "yelpchi" else ds)
        n, d = data.x.shape

        # 1. Spectral energy
        cfg = SOCTrainerConfig(
            kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0, alpha=2.0, T=1.0,
            template_type="low", graph_type="original", gamma=1.0,
            normalize_mode="zscore", prior_mean_mode="stationary",
            laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True,
        )
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=args.device)
            x_normed = trainer.normalize(data.x.to(args.device).float())
            V_np = trainer.V.cpu().numpy()
            lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
            delta = (x_normed - trainer.template).cpu().numpy()
            S = compute_spectral_energy(delta, V_np)
            spectral_all[ds] = {"lam_L": lam_L, "S": S, "d": d}
            gini_all[ds] = compute_spectral_concentration(S)
            print("  Spectral Gini: %.3f" % gini_all[ds], flush=True)
        except Exception as e:
            print("  Spectral failed: %s" % str(e)[:50], flush=True)

        # 2. Homophily
        try:
            h_mean, h_std = compute_homophily(data)
            homophily_all[ds] = (h_mean, h_std)
            print("  Homophily: %.3f ± %.3f" % (h_mean, h_std), flush=True)
        except Exception as e:
            print("  Homophily failed: %s" % str(e)[:50], flush=True)

        # 3. PCA variance
        try:
            x = data.x.float()
            x_centered = x - x.mean(dim=0)
            U, S_pca, Vt = torch.linalg.svd(x_centered, full_matrices=False)
            explained = (S_pca ** 2) / (S_pca ** 2).sum()
            cumvar = explained.cumsum(0).cpu().numpy()
            pca_all[ds] = {"cumvar": cumvar, "d": d}
            k90 = int((cumvar < 0.9).sum()) + 1
            print("  PCA: d=%d, 90%% variance at k=%d" % (d, k90), flush=True)
        except Exception as e:
            print("  PCA failed: %s" % str(e)[:50], flush=True)

    # Generate plots
    if spectral_all:
        plot_spectral_energy_annotated(spectral_all, args.output_dir)

    if homophily_all:
        plot_homophily_vs_template(homophily_all, args.output_dir)

    if pca_all:
        plot_pca_variance(pca_all, args.output_dir)

    if homophily_all and gini_all:
        plot_summary_scatter(homophily_all, gini_all, args.output_dir)

    # Print summary table
    print("\n=== SUMMARY ===")
    print("%-14s | %6s | %5s | %5s | %6s | %4s" % (
        "Dataset", "Homoph", "Gini", "d", "k@90%", "Best"))
    print("-" * 55)
    for ds in datasets:
        h = homophily_all.get(ds, (0, 0))[0]
        g = gini_all.get(ds, 0)
        d = BEST_INFO.get(ds, {}).get("d", 0)
        k90 = int((pca_all[ds]["cumvar"] < 0.9).sum()) + 1 if ds in pca_all else 0
        score = BEST_INFO.get(ds, {}).get("score", "?")
        print("%-14s | %6.3f | %5.3f | %5d | %6d | %4s" % (ds, h, g, d, k90, score))

    print("\nDone!")


if __name__ == "__main__":
    main()
