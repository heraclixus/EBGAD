"""Generate score density separation plots for all datasets.

Produces per-dataset density plots showing normal vs anomalous score
distributions under the unsupervised pipeline (ML-selected config + KS score).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def ks_null(scores):
    s = scores[scores > 0]
    if len(s) < 10: return 0.0
    mu, var = s.mean(), s.var()
    if var < 1e-12 or mu < 1e-12: return 0.0
    k = max(0.01, 2*mu**2/var); a = var/(2*mu)
    try: return kstest(s/a, 'chi2', args=(k,))[0]
    except: return 0.0


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = data.edge_index.shape[1] // 2 / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean()), density


def get_density_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in ALL_GAMMAS if g <= max_gamma]
    if not nearest: nearest = [ALL_GAMMAS[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


def run_pipeline(ds, device="cpu"):
    """Run unsupervised pipeline, return (scores, y, mask, score_name, auc, config)."""
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d_raw = data.x.shape[1]
    pca_dim = 64 if d_raw > 100 else None

    h, density = compute_graph_stats(data)
    density_range = get_density_range(h, density)

    best_ml = -np.inf
    best_result = None

    for gamma in density_range:
        cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
                   normalize_mode="zscore", prior_mean_mode="stationary",
                   laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
                   epochs=0, normalize_features=True,
                   template_type="low", graph_type="original", gamma=gamma)
        if pca_dim and d_raw > pca_dim:
            cfg.update(dict(use_encoder=True, encoder_type="pca",
                           encoder_hid_dim=pca_dim, encoder_num_layers=1,
                           encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                           encoder_alpha=1.0, encoder_weight_decay=0.0))
        trainer = SOCTrainer(SOCTrainerConfig(**cfg))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        x = trainer.x_T.cpu().numpy()
        tmpl = trainer.template.cpu().numpy()
        n, d = x.shape
        delta = x - tmpl
        S = compute_spectral_energy(delta, V)

        for kk in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kk, d)
            q = Q_rho_eigenvalues(lam_L, rho, kk)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml = ml
                # Compute scores
                q_safe = np.maximum(q, 1e-12)
                delta_hat = V.T @ delta
                w = np.sqrt(q_safe)[:, None] * delta_hat
                je = np.sum((V @ w)**2, axis=1)
                tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
                jr = je / tot
                best_result = {
                    "je": je, "jr": jr, "gamma": gamma,
                    "rho": rho, "kappa": kk,
                }

    je = best_result["je"]
    jr = best_result["jr"]
    ks_je = ks_null(je[mask])
    ks_jr = ks_null(jr[mask])
    if ks_je > ks_jr:
        scores, sname = je, "$J^\\star$"
    else:
        scores, sname = jr, "$R$"
    auc = roc_auc_score(y[mask], scores[mask])

    return scores[mask], y[mask], sname, auc, best_result


def plot_density(scores, y, sname, auc, ds_name, ax):
    """Plot score density for normal vs anomalous on given axis."""
    normal = scores[y == 0]
    anomaly = scores[y == 1]

    # Use percentile clipping for better visualization
    clip_max = np.percentile(scores, 98)
    clip_min = np.percentile(scores, 1)
    normal = normal[(normal >= clip_min) & (normal <= clip_max)]
    anomaly = anomaly[(anomaly >= clip_min) & (anomaly <= clip_max)]

    bins = np.linspace(clip_min, clip_max, 80)
    ax.hist(normal, bins=bins, density=True, alpha=0.6, color="#2196F3", label="Normal")
    ax.hist(anomaly, bins=bins, density=True, alpha=0.6, color="#F44336", label="Anomaly")
    ax.set_title(f"{ds_name}", fontsize=10)
    ax.set_xlabel("Score", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog",
                     "facebook", "acm", "elliptic", "elliptic_plus_plus", "t_finance"]
    else:
        datasets = [args.dataset]

    os.makedirs("figures", exist_ok=True)
    results = {}

    for ds in datasets:
        print(f"  {ds}...", end="", flush=True)
        try:
            scores, y, sname, auc, cfg = run_pipeline(ds, args.device)
            results[ds] = (scores, y, sname, auc)
            print(f" {sname} AUC={auc:.1f}%")
        except Exception as e:
            print(f" ERROR: {str(e)[:60]}")

    # Individual plots
    for ds, (scores, y, sname, auc) in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        plot_density(scores, y, sname, auc, ds, ax)
        fig.tight_layout()
        fig.savefig(f"figures/density_{ds}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figures/density_{ds}.pdf")

    # Combined 2x5 grid (all datasets)
    if len(results) >= 10:
        fig, axes = plt.subplots(2, 5, figsize=(20, 6))
        ds_order = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog",
                     "facebook", "acm", "elliptic", "elliptic_plus_plus", "t_finance"]
        ds_labels = ["Weibo", "Reddit", "Amazon", "YelpChi", "BlogCatalog",
                      "Facebook", "ACM", "Elliptic", "Elliptic++", "T-Finance"]
        for i, (ds, label) in enumerate(zip(ds_order, ds_labels)):
            if ds in results:
                scores, y, sname, auc = results[ds]
                plot_density(scores, y, sname, auc, label, axes[i//5, i%5])
        fig.tight_layout()
        fig.savefig("figures/density_all_2x5.pdf", bbox_inches="tight")
        plt.close(fig)
        print("  Saved figures/density_all_2x5.pdf")

    # 1x2 for paper Figure 2: Weibo + T-Finance
    fig2_ds = ["weibo", "t_finance"]
    fig2_labels = ["Weibo", "T-Finance"]
    if all(c in results for c in fig2_ds):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for i, (ds, label) in enumerate(zip(fig2_ds, fig2_labels)):
            scores, y, sname, auc = results[ds]
            plot_density(scores, y, sname, auc, label, axes[i])
        fig.tight_layout()
        fig.savefig("figures/density_weibo_tfinance.pdf", bbox_inches="tight")
        plt.close(fig)
        print("  Saved figures/density_weibo_tfinance.pdf")


if __name__ == "__main__":
    main()
