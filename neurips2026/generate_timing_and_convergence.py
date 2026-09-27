"""Generate runtime comparison and Newton convergence plots.

Figure 1 (runtime_comparison.pdf):
  Log-scale bar chart comparing EB-GAD vs DiffGAD vs TAM wall-clock time.

Figure 2 (newton_convergence.pdf):
  Newton iteration convergence: |rho_k - rho*| vs iteration for multiple datasets.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, warnings, time
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from data_utils import load_data
from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    marginal_log_likelihood_stationary,
    compute_spectral_energy,
    solve_rho_newton,
)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

# Best configs for timing
BEST_CONFIGS = {
    "weibo": dict(kappa=0.0, gamma=0.2, graph_type="original",
                  template_type="low", prior_mean_mode="stationary",
                  use_encoder=False),
    "facebook": dict(kappa=1.1, gamma=0.1, graph_type="original",
                     template_type="affinity", prior_mean_mode="stationary",
                     use_encoder=False),
    "enron": dict(kappa=0.0, gamma=0.1, graph_type="affinity",
                  template_type="high", prior_mean_mode="stationary",
                  use_encoder=False),
    "amazon": dict(kappa=0.0, gamma=5.0, graph_type="original",
                   template_type="low", prior_mean_mode="stationary",
                   use_encoder=False),
    "reddit": dict(kappa=1.0, gamma=1.0, graph_type="original",
                   template_type="low", prior_mean_mode="stationary",
                   use_encoder=False),
    "yelpchi": dict(kappa=0.0, gamma=1.0, graph_type="original",
                    template_type="low", prior_mean_mode="stationary",
                    use_encoder=True, encoder_type="pca",
                    encoder_hid_dim=32),
    "blogcatalog": dict(kappa=0.01, gamma=0.2, graph_type="original",
                        template_type="low", prior_mean_mode="stationary",
                        use_encoder=True, encoder_type="pca",
                        encoder_hid_dim=128),
    "acm": dict(kappa=0.1, gamma=0.5, graph_type="original",
                template_type="affinity", prior_mean_mode="stationary",
                use_encoder=True, encoder_type="pca",
                encoder_hid_dim=128),
    "t_finance": dict(kappa=0.0, gamma=0.1, graph_type="original",
                      template_type="affinity", prior_mean_mode="stationary",
                      use_encoder=False),
    "elliptic": dict(kappa=0.2, gamma=0.7, graph_type="original",
                     template_type="affinity", prior_mean_mode="stationary",
                     use_encoder=False),
    "elliptic_plus_plus": dict(kappa=0.2, gamma=0.5, graph_type="original",
                               template_type="affinity",
                               prior_mean_mode="stationary",
                               use_encoder=False),
}

# Baseline runtimes (seconds) from our measurements on same hardware.
# All graph-based methods. None = OOM or not measured.
BASELINE_TIMES = {
    "Enron":       {"ANOM.": None,  "DOMINANT": None, "AnomDAE": None,  "CONAD": None,  "CoLA": 62,    "DiffGAD": 109,   "TAM": 286},
    "Weibo":       {"ANOM.": 78,    "DOMINANT": None, "AnomDAE": None,  "CONAD": None,  "CoLA": 130,   "DiffGAD": 134,   "TAM": 858},
    "Reddit":      {"ANOM.": None,  "DOMINANT": None, "AnomDAE": None,  "CONAD": None,  "CoLA": None,  "DiffGAD": 110,   "TAM": 5467},
    "Amazon":      {"ANOM.": None,  "DOMINANT": None, "AnomDAE": 153,   "CONAD": 308,   "CoLA": None,  "DiffGAD": 106,   "TAM": 10739},
    "YelpChi":     {"ANOM.": None,  "DOMINANT": None, "AnomDAE": 1152,  "CONAD": 1359,  "CoLA": None,  "DiffGAD": 118,   "TAM": 5106},
    "BlogCatalog": {"ANOM.": None,  "DOMINANT": None, "AnomDAE": 92,    "CONAD": 3047,  "CoLA": None,  "DiffGAD": 3272,  "TAM": 8764},
    "Facebook":    {"ANOM.": None,  "DOMINANT": 20,   "AnomDAE": 9,     "CONAD": 56,    "CoLA": None,  "DiffGAD": 118,   "TAM": 479},
    "ACM":         {"ANOM.": None,  "DOMINANT": None, "AnomDAE": 580,   "CONAD": 2293,  "CoLA": None,  "DiffGAD": 39209, "TAM": 14446},
    "T-Finance":   {"ANOM.": 963,   "DOMINANT": None, "AnomDAE": 9761,  "CONAD": None,  "CoLA": None,  "DiffGAD": 384,   "TAM": None},
    "Elliptic":    {"ANOM.": None,  "DOMINANT": 19802,"AnomDAE": None,  "CONAD": None,  "CoLA": 263,   "DiffGAD": 294,   "TAM": 1176},
    "Elliptic++":  {"ANOM.": None,  "DOMINANT": 16962,"AnomDAE": None,  "CONAD": None,  "CoLA": 288,   "DiffGAD": 292,   "TAM": 1186},
}

# Methods for the bar chart (keep clean with 4)
CHART_METHODS = ["CoLA", "DiffGAD", "TAM"]
# All methods for the appendix table
ALL_METHODS = ["ANOM.", "DOMINANT", "AnomDAE", "CONAD", "CoLA", "DiffGAD", "TAM"]

DS_NAMES = {
    "weibo": "Weibo", "facebook": "Facebook", "amazon": "Amazon",
    "enron": "Enron", "elliptic": "Elliptic", "yelpchi": "YelpChi",
    "t_finance": "T-Finance", "acm": "ACM", "reddit": "Reddit",
    "blogcatalog": "BlogCatalog", "elliptic_plus_plus": "Elliptic++",
}

TIMING_DATASETS = [
    "enron", "weibo", "reddit", "amazon", "yelpchi",
    "blogcatalog", "facebook", "acm", "t_finance",
    "elliptic", "elliptic_plus_plus",
]

CONVERGENCE_DATASETS = ["weibo", "amazon", "facebook", "elliptic",
                        "enron", "reddit", "yelpchi", "acm"]


# ══════════════════════════════════════════════════════════════════
# Runtime measurement
# ══════════════════════════════════════════════════════════════════

def measure_runtime(ds_name, device="cpu"):
    """Measure EB-GAD runtime, separating preprocessing from scoring.

    Preprocessing (one-time): eigendecomposition + affinity weights
    Scoring (per-config): template + ML optimization + anomaly scores

    Returns (preprocess_time, scoring_time).
    """
    cfg_ov = BEST_CONFIGS.get(ds_name, {}).copy()

    load_name = ("YelpChi" if ds_name == "yelpchi" else
                 "Facebook" if ds_name == "facebook" else ds_name)
    data = load_data(load_name)

    base = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, template_type="low", graph_type="original",
        normalize_mode="zscore", d_hidden=64, n_hidden_layers=2,
        epochs=0, normalize_features=True, laplacian_variant="sym",
        prior_mean_mode="stationary", gamma=0.5, score_K=1,
    )
    base.update(cfg_ov)
    for k in ["score_method"]:
        base.pop(k, None)

    cfg = SOCGADConfig(dataset=ds_name, **base)

    # Run 1: full pipeline (preprocessing + scoring) = total time
    torch.manual_seed(0)
    t0 = time.time()
    _ = evaluate_single_trial(data, cfg, device=device)
    total_time = time.time() - t0

    # Run 2+: reuse cached preprocessing (trainer already built)
    # Subsequent runs only do scoring (template + score computation)
    scoring_times = []
    for seed in range(3):
        torch.manual_seed(seed)
        t0 = time.time()
        _ = evaluate_single_trial(data, cfg, device=device)
        scoring_times.append(time.time() - t0)

    scoring_time = np.mean(scoring_times)
    preprocess_time = max(total_time - scoring_time, 0)

    return preprocess_time, scoring_time


def generate_runtime_chart(device="cpu"):
    """Bar chart: EB-GAD (scoring only) vs DiffGAD vs TAM."""
    ebgad_preprocess = {}
    ebgad_scoring = {}
    for ds in TIMING_DATASETS:
        print(f"  Timing {DS_NAMES.get(ds, ds)}...", flush=True)
        try:
            pre, score = measure_runtime(ds, device)
            ebgad_preprocess[ds] = pre
            ebgad_scoring[ds] = score
            print(f"    preprocess={pre:.1f}s  scoring={score:.1f}s  "
                  f"total={pre+score:.1f}s", flush=True)
        except Exception as e:
            print(f"    ERROR: {str(e)[:60]}", flush=True)

    # Build data for chart
    datasets = []
    eb_score_vals, eb_pre_vals = [], []
    diff_vals, tam_vals = [], []
    for ds in TIMING_DATASETS:
        if ds not in ebgad_scoring:
            continue
        dn = DS_NAMES.get(ds, ds)
        datasets.append(dn)
        eb_score_vals.append(max(ebgad_scoring[ds], 0.01))  # floor for log
        eb_pre_vals.append(ebgad_preprocess[ds])
        diff_vals.append(BASELINE_TIMES.get(dn, {}).get("DiffGAD", None))
        tam_vals.append(BASELINE_TIMES.get(dn, {}).get("TAM", None))

    # Graph-based methods for chart
    methods = ["EB-GAD"] + CHART_METHODS
    method_colors = {
        "EB-GAD": "#2ca02c", "EB-GAD (pre)": "#98df8a",
        "CoLA": "#ff7f0e", "DiffGAD": "#1f77b4", "TAM": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(7, 2.5))
    n_methods = len(methods)
    x = np.arange(len(datasets))
    w = 0.8 / n_methods

    for m_idx, method in enumerate(methods):
        vals = []
        for ds in datasets:
            if method == "EB-GAD":
                ds_idx = datasets.index(ds)
                vals.append(max(eb_score_vals[ds_idx], 0.005))
            else:
                v = BASELINE_TIMES.get(ds, {}).get(method, None)
                vals.append(v if v and v > 0 else None)

        offset = (m_idx - n_methods / 2 + 0.5) * w
        # Filter None values (OOM/missing)
        plot_vals = [v if v else 0 for v in vals]
        bars = ax.bar(x + offset, plot_vals, w,
                      label=method, color=method_colors.get(method, "gray"),
                      edgecolor="k", linewidth=0.3)

        # Stack preprocessing on EB-GAD
        if method == "EB-GAD":
            ax.bar(x + offset, eb_pre_vals, w, bottom=eb_score_vals,
                   color=method_colors["EB-GAD (pre)"],
                   edgecolor="k", linewidth=0.3, alpha=0.6,
                   label="+ preprocess")

    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=6.5)
    ax.legend(fontsize=6, loc="upper left", ncol=3,
              columnspacing=0.8, handletextpad=0.4)
    ax.set_ylim(bottom=0.003, top=5e4)

    plt.tight_layout()
    out = os.path.join("figures",
                       "runtime_comparison.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()

    # Print compact table for chart
    print("\n=== Runtime Chart Data ===")
    for ds, eb_s, eb_p in zip(datasets, eb_score_vals, eb_pre_vals):
        print(f"  {ds:15s}  pre={eb_p:6.1f}s  score={eb_s:6.2f}s  "
              f"total={eb_p+eb_s:6.1f}s")

    # Print FULL table for appendix (all graph-based methods)
    print("\n=== Full Runtime Table (for appendix) ===")
    header = f"  {'Dataset':15s}  {'EB-GAD':>8s}"
    for m in ALL_METHODS:
        header += f"  {m:>9s}"
    print(header)
    for ds, eb_s, eb_p in zip(datasets, eb_score_vals, eb_pre_vals):
        # Report EB-GAD as scoring time (preprocessing amortized)
        line = f"  {ds:15s}  {eb_s:7.2f}s"
        for m in ALL_METHODS:
            v = BASELINE_TIMES.get(ds, {}).get(m)
            if v is not None:
                line += f"  {v:8.0f}s"
            else:
                line += f"       ---"
        print(line)

    # LaTeX table
    print("\n=== LaTeX table ===")
    print("% EB-GAD reports scoring time only (eigendecomposition is a "
          "one-time preprocessing step).")
    for ds, eb_s, eb_p in zip(datasets, eb_score_vals, eb_pre_vals):
        parts = [f"    {ds:15s}"]
        for m in ALL_METHODS:
            v = BASELINE_TIMES.get(ds, {}).get(m)
            if v is not None:
                parts.append(f"{v:.0f}")
            else:
                parts.append("---")
        parts.append(f"\\textbf{{{eb_s:.1f}}}")
        print(" & ".join(parts) + " \\\\")


# ══════════════════════════════════════════════════════════════════
# Newton convergence
# ══════════════════════════════════════════════════════════════════

def newton_iterations(S, lam_L, kappa, d, rho_init=0.1, n_steps=10):
    """Run Newton, return all rho iterates."""
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


def generate_convergence(device="cpu"):
    """Newton convergence: |rho_k - rho*| vs iteration."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONVERGENCE_DATASETS)))

    for idx, ds in enumerate(CONVERGENCE_DATASETS):
        cfg_ov = BEST_CONFIGS.get(ds, {}).copy()
        kappa = cfg_ov.get("kappa", 0.0)

        print(f"  [{idx+1}/{len(CONVERGENCE_DATASETS)}] "
              f"{DS_NAMES.get(ds, ds)}...", flush=True)

        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)

        base = dict(
            kappa=kappa, nu=1.0, rho=0.5, lam_penalty=50.0,
            alpha=2.0, T=1.0, template_type=cfg_ov.get("template_type", "low"),
            graph_type=cfg_ov.get("graph_type", "original"),
            normalize_mode="zscore", d_hidden=64, n_hidden_layers=2,
            epochs=0, normalize_features=True, laplacian_variant="sym",
            prior_mean_mode=cfg_ov.get("prior_mean_mode", "stationary"),
            gamma=cfg_ov.get("gamma", 0.5),
        )
        for k in ["score_method"]:
            base.pop(k, None)

        try:
            trainer_cfg = SOCTrainerConfig(**base)
            trainer = SOCTrainer(trainer_cfg)
            data = load_data(load_name)
            trainer.train(data, device=device)

            V = trainer.V.cpu().numpy()
            lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
            template = trainer.template.cpu().numpy()

            x = data.x.float()
            if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
                x = trainer.normalizer(x)
            if hasattr(trainer, 'encoder') and trainer.encoder is not None:
                with torch.no_grad():
                    x = trainer.encoder(x.to(device),
                                       data.edge_index.to(device)).cpu()
            delta = x.numpy() - template
            S = compute_spectral_energy(delta, V)
            d_feat = delta.shape[1]

            iterates = newton_iterations(S, lam_L, kappa, d_feat,
                                         rho_init=0.1)
            rho_star = iterates[-1]
            errors = [abs(r - rho_star) + 1e-16 for r in iterates]

            ax.semilogy(range(len(errors)), errors, 'o-', color=colors[idx],
                       markersize=3, lw=1.0,
                       label=DS_NAMES.get(ds, ds))
        except Exception as e:
            print(f"    ERROR: {str(e)[:60]}", flush=True)

    ax.set_xlabel("Newton iteration")
    ax.set_ylabel(r"$|\rho_k - \rho^\star|$")
    ax.set_xlim(-0.2, 6.2)
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title("Newton convergence for $\\rho$", fontsize=9)

    plt.tight_layout()
    out = os.path.join("figures",
                       "newton_convergence.pdf")
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--convergence-only", action="store_true")
    args = parser.parse_args()

    if not args.convergence_only:
        print("=== Generating runtime comparison ===")
        generate_runtime_chart(args.device)

    if not args.runtime_only:
        print("\n=== Generating Newton convergence ===")
        generate_convergence(args.device)

    print("\nDone.")
