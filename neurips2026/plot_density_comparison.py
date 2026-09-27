"""Generate TAM vs EB-GAD density comparison plots for Weibo and Amazon.

Style: smooth KDE, filled areas, blue/orange colors.
Weibo: log(1+score) transform.
Amazon: raw scores.
Outputs: 1x4 and 2x2 versions.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kstest
from sklearn.metrics import roc_auc_score

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


def run_ebgad(ds, device="cpu"):
    """Run EB-GAD pipeline, return (scores, y, mask, score_name)."""
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
    best_scores = None

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
                q_safe = np.maximum(q, 1e-12)
                delta_hat = V.T @ delta
                w = np.sqrt(q_safe)[:, None] * delta_hat
                je = np.sum((V @ w)**2, axis=1)
                tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
                jr = je / tot
                best_scores = {"je": je, "jr": jr}

    je, jr = best_scores["je"], best_scores["jr"]
    ks_je = ks_null(je[mask])
    ks_jr = ks_null(jr[mask])
    if ks_je > ks_jr:
        return je, y, mask, "$J^\\star$"
    else:
        return jr, y, mask, "$R$"


def run_tam(ds, device="cpu"):
    """Run TAM, return (scores, y_eval)."""
    from run_tam import run_tam_trial
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    result = run_tam_trial(data, device=device, return_scores=True, verbose=True)
    return result["scores"], result["y"]


def plot_kde(scores, y, mask, title, ax, transform=None):
    """Plot smooth KDE density."""
    s = scores[mask]
    labels = y[mask]
    normal = s[labels == 0]
    anomaly = s[labels == 1]

    if transform == "log1p":
        normal = np.log1p(normal)
        anomaly = np.log1p(anomaly)

    # Clip extreme outliers
    all_vals = np.concatenate([normal, anomaly])
    lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    normal = normal[(normal >= lo) & (normal <= hi)]
    anomaly = anomaly[(anomaly >= lo) & (anomaly <= hi)]

    x_grid = np.linspace(lo, hi, 500)

    try:
        kde_n = gaussian_kde(normal, bw_method=0.3)
        ax.fill_between(x_grid, kde_n(x_grid), alpha=0.5, color="#5B9BD5", label="Normal")
        ax.plot(x_grid, kde_n(x_grid), color="#5B9BD5", linewidth=1)
    except:
        ax.hist(normal, bins=60, density=True, alpha=0.5, color="#5B9BD5", label="Normal")

    try:
        kde_a = gaussian_kde(anomaly, bw_method=0.3)
        ax.fill_between(x_grid, kde_a(x_grid), alpha=0.5, color="#ED7D31", label="Abnormal")
        ax.plot(x_grid, kde_a(x_grid), color="#ED7D31", linewidth=1)
    except:
        ax.hist(anomaly, bins=60, density=True, alpha=0.5, color="#ED7D31", label="Abnormal")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=11)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    # Run EB-GAD on both datasets
    print("Running EB-GAD on Weibo...", flush=True)
    eb_weibo_scores, eb_weibo_y, eb_weibo_mask, eb_weibo_sname = run_ebgad("weibo", args.device)
    print(f"  {eb_weibo_sname} AUC={roc_auc_score(eb_weibo_y[eb_weibo_mask], eb_weibo_scores[eb_weibo_mask])*100:.1f}%")

    print("Running EB-GAD on Amazon...", flush=True)
    eb_amazon_scores, eb_amazon_y, eb_amazon_mask, eb_amazon_sname = run_ebgad("amazon", args.device)
    print(f"  {eb_amazon_sname} AUC={roc_auc_score(eb_amazon_y[eb_amazon_mask], eb_amazon_scores[eb_amazon_mask])*100:.1f}%")

    # Run TAM on both datasets
    print("Running TAM on Weibo...", flush=True)
    try:
        tam_weibo_scores, tam_weibo_y = run_tam("weibo", args.device)
        print(f"  TAM AUC={roc_auc_score(tam_weibo_y, tam_weibo_scores)*100:.1f}%")
        have_tam_weibo = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  TAM Weibo failed: {e}")
        have_tam_weibo = False

    print("Running TAM on Amazon...", flush=True)
    try:
        tam_amazon_scores, tam_amazon_y = run_tam("amazon", args.device)
        print(f"  TAM AUC={roc_auc_score(tam_amazon_y, tam_amazon_scores)*100:.1f}%")
        have_tam_amazon = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  TAM Amazon failed: {e}")
        have_tam_amazon = False

    # Helper: wrap TAM scores (already masked) into full-array format for plot_kde
    def _plot_tam(scores, y, title, ax, **kw):
        """TAM returns pre-masked arrays; wrap for plot_kde."""
        mask = np.ones(len(y), dtype=bool)
        plot_kde(scores, y, mask, title, ax, **kw)

    # === 1x4 version ===
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))

    if have_tam_weibo:
        _plot_tam(tam_weibo_scores, tam_weibo_y,
                  "TAM \u2014 Weibo", axes[0], transform="log1p")
        axes[0].legend(fontsize=11, loc="upper left")

    plot_kde(eb_weibo_scores, eb_weibo_y, eb_weibo_mask,
             "EB-GAD \u2014 Weibo", axes[1], transform="log1p")

    if have_tam_amazon:
        _plot_tam(tam_amazon_scores, tam_amazon_y,
                  "TAM \u2014 Amazon", axes[2], transform="log1p")

    plot_kde(eb_amazon_scores, eb_amazon_y, eb_amazon_mask,
             "EB-GAD \u2014 Amazon", axes[3], transform="log1p")

    fig.tight_layout()
    fig.savefig("figures/density_comparison_1x4.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved figures/density_comparison_1x4.pdf")

    # === 2x2 version ===
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    if have_tam_weibo:
        _plot_tam(tam_weibo_scores, tam_weibo_y,
                  "TAM \u2014 Weibo", axes[0, 0], transform="log1p")
        axes[0, 0].legend(fontsize=11, loc="upper left")

    plot_kde(eb_weibo_scores, eb_weibo_y, eb_weibo_mask,
             "EB-GAD \u2014 Weibo", axes[0, 1], transform="log1p")

    if have_tam_amazon:
        _plot_tam(tam_amazon_scores, tam_amazon_y,
                  "TAM \u2014 Amazon", axes[1, 0], transform="log1p")

    plot_kde(eb_amazon_scores, eb_amazon_y, eb_amazon_mask,
             "EB-GAD \u2014 Amazon", axes[1, 1], transform="log1p")

    fig.tight_layout()
    fig.savefig("figures/density_comparison_2x2.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved figures/density_comparison_2x2.pdf")


if __name__ == "__main__":
    main()
