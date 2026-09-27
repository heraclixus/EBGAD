"""Generate ML contour plots for many (dataset, profile) combinations.

Sweeps over datasets × templates × graph_types × gammas,
producing individual contour PNGs. Pick the best-looking ones afterward.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    marginal_log_likelihood_stationary,
    compute_spectral_energy,
    solve_rho_newton,
)

DS_NAMES = {
    "weibo": "Weibo", "facebook": "Facebook", "amazon": "Amazon",
    "enron": "Enron", "elliptic": "Elliptic", "yelpchi": "YelpChi",
    "t_finance": "T-Finance", "acm": "ACM", "reddit": "Reddit",
    "blogcatalog": "BlogCatalog",
}

DATASETS = ["weibo", "facebook", "amazon", "enron", "reddit",
            "yelpchi", "acm", "blogcatalog", "elliptic", "t_finance"]

PROFILES = []
for tmpl in ["zero", "low", "affinity"]:
    for graph in ["original", "affinity"]:
        for gamma in [0.1, 0.5, 1.0, 2.0]:
            for pca in [None, 32, 128]:
                PROFILES.append({
                    "template_type": tmpl,
                    "graph_type": graph,
                    "gamma": gamma,
                    "pca": pca,
                })


def build_trainer(ds_name, profile, device="cpu"):
    cfg = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        d_hidden=64, n_hidden_layers=2, epochs=0,
        normalize_features=True, laplacian_variant="sym",
        prior_mean_mode="stationary",
        template_type=profile["template_type"],
        graph_type=profile["graph_type"],
        gamma=profile["gamma"],
    )
    if profile["pca"] is not None:
        cfg["use_encoder"] = True
        cfg["encoder_type"] = "pca"
        cfg["encoder_hid_dim"] = profile["pca"]
        cfg["encoder_num_layers"] = 1
        cfg["encoder_dropout"] = 0.0
        cfg["encoder_lr"] = 0.0
        cfg["encoder_epochs"] = 0
        cfg["encoder_alpha"] = 1.0
        cfg["encoder_weight_decay"] = 0.0

    trainer_cfg = SOCTrainerConfig(**cfg)
    trainer = SOCTrainer(trainer_cfg)
    load_name = ("YelpChi" if ds_name == "yelpchi" else
                 "Facebook" if ds_name == "facebook" else ds_name)
    data = load_data(load_name)
    trainer.train(data, device=device)
    return trainer, data


def compute_contour(trainer, data, device="cpu"):
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    template = trainer.template.cpu().numpy()

    x = data.x.float()
    if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
        x = trainer.normalizer(x)
    if hasattr(trainer, 'encoder') and trainer.encoder is not None:
        with torch.no_grad():
            x = trainer.encoder(x.to(device), data.edge_index.to(device)).cpu()
    x_np = x.numpy()
    n, d = x_np.shape
    delta = x_np - template
    S = compute_spectral_energy(delta, V)

    rho_grid = np.linspace(0.01, 0.99, 60)
    kappa_grid = np.logspace(-2, 1.5, 60)

    ML = np.zeros((len(rho_grid), len(kappa_grid)))
    for i, rho in enumerate(rho_grid):
        for j, kappa in enumerate(kappa_grid):
            q = Q_rho_eigenvalues(lam_L, rho, kappa, nu=1.0)
            ML[i, j] = marginal_log_likelihood_stationary(S, q, n, d)

    # Profile-Newton path
    path_kappa = np.logspace(-2, 1.5, 40)
    path_rho = []
    best_ml, best_rho, best_kappa = -np.inf, 0.5, 1.0
    for kp in path_kappa:
        rs = solve_rho_newton(S, lam_L, kp, d)
        path_rho.append(rs)
        q = Q_rho_eigenvalues(lam_L, rs, kp)
        ml = marginal_log_likelihood_stationary(S, q, n, d)
        if ml > best_ml:
            best_ml, best_rho, best_kappa = ml, rs, kp

    return (rho_grid, kappa_grid, ML,
            np.array(path_rho), path_kappa, best_rho, best_kappa)


def score_contour(ML):
    """Score how 'interesting' a contour is. Higher = more structure."""
    ml_norm = (ML - ML.max()) / max(ML.max() - ML.min(), 1e-6) * 100
    midrange = np.mean((ml_norm > -80) & (ml_norm < -20))
    grad_r = np.diff(ml_norm, axis=0)
    grad_k = np.diff(ml_norm, axis=1)
    grad_mag = np.mean(np.abs(grad_r)) + np.mean(np.abs(grad_k))
    return midrange * 10 + grad_mag


def newton_trajectory(S, lam_L, kappa, d, rho_init=0.1, n_steps=10):
    """Run Newton, return iterates and convergence info."""
    a = (kappa**2 + lam_L)**1.0 - 1.0
    rho = np.clip(rho_init, 1e-6, 1.0 - 1e-6)
    iterates = [float(rho)]
    for _ in range(n_steps):
        q = rho * a + 1.0
        q_safe = np.maximum(q, 1e-12)
        grad = np.sum(a * (-S / 2.0 + d / (2.0 * q_safe)))
        hess = -np.sum(d * a**2 / (2.0 * q_safe**2))
        if abs(hess) < 1e-15:
            break
        rho_new = float(np.clip(rho - grad / hess, 0.0, 1.0))
        iterates.append(rho_new)
        if abs(rho_new - rho) < 1e-12:
            break
        rho = rho_new
    return iterates


def score_convergence(iterates):
    """Score how 'interesting' a Newton convergence is.

    Best: interior rho* (0.1-0.9) reached in 3-5 steps.
    Boring: boundary rho* (0 or 1) reached in 1 step.
    """
    rho_star = iterates[-1]
    n_iters = len(iterates) - 1  # exclude initial point

    # Interior optimum is more interesting
    interior_score = 1.0 - 2 * abs(rho_star - 0.5)  # 1 at 0.5, 0 at 0/1

    # 3-5 iterations is ideal (shows convergence without being slow)
    iter_score = max(0, min(n_iters, 5) - 1) / 4.0  # 0 for 1 iter, 1 for 5

    # Check for clear convergence (errors should decrease)
    errors = [abs(r - rho_star) for r in iterates]
    if len(errors) > 2 and errors[0] > 1e-3:
        # Ratio of successive errors (quadratic = ratio decreases)
        ratios = []
        for i in range(1, len(errors)-1):
            if errors[i] > 1e-14 and errors[i-1] > 1e-14:
                ratios.append(errors[i] / errors[i-1])
        convergence_quality = 1.0 if ratios and max(ratios) < 0.5 else 0.5
    else:
        convergence_quality = 0.0

    return interior_score * 3 + iter_score * 2 + convergence_quality


def main(device="cpu", chunk=0, n_chunks=1):
    out_dir = os.path.join("figures",
                           "contour_sweep")
    os.makedirs(out_dir, exist_ok=True)

    all_configs = []
    for ds in DATASETS:
        for p_idx, profile in enumerate(PROFILES):
            all_configs.append((ds, p_idx, profile))

    my_configs = [c for i, c in enumerate(all_configs)
                  if i % n_chunks == chunk]

    print(f"=== Contour sweep: {len(my_configs)} configs "
          f"(chunk {chunk}/{n_chunks}) ===", flush=True)

    results = []
    for idx, (ds, p_idx, profile) in enumerate(my_configs):
        pca_str = f"pca{profile['pca']}" if profile['pca'] else "nopca"
        tag = (f"{ds}_{profile['template_type']}_{profile['graph_type']}_"
               f"g{profile['gamma']}_{pca_str}")

        fname = f"{tag}.png"
        fpath = os.path.join(out_dir, fname)

        if os.path.exists(fpath):
            continue

        print(f"  [{idx+1}/{len(my_configs)}] {tag}...", end="", flush=True)
        try:
            trainer, data = build_trainer(ds, profile, device)
            (rho_grid, kappa_grid, ML,
             path_rho, path_kappa, best_rho, best_kappa) = \
                compute_contour(trainer, data, device)

            interest = score_contour(ML)

            # Newton convergence from rho=0.1 at the best kappa
            V = trainer.V.cpu().numpy()
            lam_L_np = trainer.graph_eigen.lam_L.cpu().numpy()
            x_feat = data.x.float()
            if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
                x_feat = trainer.normalizer(x_feat)
            if hasattr(trainer, 'encoder') and trainer.encoder is not None:
                with torch.no_grad():
                    x_feat = trainer.encoder(
                        x_feat.to(device),
                        data.edge_index.to(device)).cpu()
            delta_np = x_feat.numpy() - trainer.template.cpu().numpy()
            S_np = compute_spectral_energy(delta_np, V)
            d_feat = delta_np.shape[1]

            newton_iters = newton_trajectory(
                S_np, lam_L_np, best_kappa, d_feat, rho_init=0.1)
            conv_score = score_convergence(newton_iters)

            # Plot contour
            fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
            ax = axes[0]
            ml_norm = (ML - ML.max()) / max(ML.max() - ML.min(), 1e-6) * 100
            K, R = np.meshgrid(kappa_grid, rho_grid)
            levels = np.linspace(-100, 0, 20)
            cf = ax.contourf(K, R, ml_norm, levels=levels,
                             cmap="RdYlBu_r", extend="min")
            ax.contour(K, R, ml_norm, levels=levels[::2],
                       colors="k", linewidths=0.2, alpha=0.3)
            ax.plot(path_kappa, path_rho, 'w-', lw=1.5, alpha=0.8)
            ax.plot(path_kappa, path_rho, 'k--', lw=0.7, alpha=0.4)
            ax.plot(best_kappa, best_rho, 'w*', ms=8,
                    markeredgecolor='k', markeredgewidth=0.6, zorder=5)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\kappa$", fontsize=8)
            ax.set_ylabel(r"$\rho$", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(
                f"{DS_NAMES.get(ds,ds)} | {profile['template_type']}, "
                f"{profile['graph_type']}, $\\gamma$={profile['gamma']}, "
                f"{pca_str}\ncontour={interest:.1f}, "
                f"conv={conv_score:.1f}",
                fontsize=6)
            fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)

            # Newton convergence panel
            ax2 = axes[1]
            rho_star = newton_iters[-1]
            errors = [abs(r - rho_star) + 1e-16 for r in newton_iters]
            ax2.semilogy(range(len(errors)), errors, 'o-',
                        color='#d62728', ms=4, lw=1.2)
            ax2.set_xlabel("Newton iteration", fontsize=8)
            ax2.set_ylabel(r"$|\rho_k - \rho^\star|$", fontsize=8)
            ax2.set_title(
                f"$\\rho^*$={rho_star:.3f}, $\\kappa^*$={best_kappa:.2f}\n"
                f"{len(newton_iters)-1} iters",
                fontsize=7)
            ax2.set_xlim(-0.3, max(len(newton_iters), 6) + 0.3)

            plt.tight_layout()
            fig.savefig(fpath, dpi=150)
            plt.close()

            total_score = interest + conv_score
            print(f" rho*={best_rho:.2f} kappa*={best_kappa:.1f} "
                  f"contour={interest:.1f} conv={conv_score:.1f} "
                  f"total={total_score:.1f}", flush=True)

            results.append({
                "ds": ds, "tag": tag,
                "contour_score": interest,
                "conv_score": conv_score,
                "total_score": total_score,
                "best_rho": best_rho, "best_kappa": best_kappa,
                "n_newton_iters": len(newton_iters) - 1,
                "file": fname,
            })
        except Exception as e:
            print(f" ERROR: {str(e)[:50]}", flush=True)

    # Save results summary
    import json
    summary_path = os.path.join(out_dir, f"summary_{chunk}.json")
    with open(summary_path, "w") as f:
        json.dump(sorted(results, key=lambda x: -x["total_score"]),
                  f, indent=2)

    # Print top-10 by total score (contour + convergence)
    print("\n=== Top 10 (contour + convergence) ===")
    for r in sorted(results, key=lambda x: -x["total_score"])[:10]:
        print(f"  total={r['total_score']:5.1f} "
              f"(ctr={r['contour_score']:.1f} "
              f"conv={r['conv_score']:.1f}) "
              f"rho*={r['best_rho']:.2f} "
              f"iters={r['n_newton_iters']} "
              f"{r['tag']}")

    # Print top-5 best convergence
    print("\n=== Top 5 convergence ===")
    for r in sorted(results, key=lambda x: -x["conv_score"])[:5]:
        print(f"  conv={r['conv_score']:.1f} "
              f"rho*={r['best_rho']:.2f} "
              f"iters={r['n_newton_iters']} "
              f"{r['tag']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    main(args.device, args.chunk, args.n_chunks)
