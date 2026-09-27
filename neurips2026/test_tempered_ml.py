"""Tempered ML for gamma selection.

Instead of max_γ ML(γ), use max_γ ML(γ)^{1/T} = max_γ ML(γ)/T.

Higher T = flatter landscape = less sensitive to ML differences between gammas.
This prevents ML from locking onto the moderate-gamma attractor.

Combined with density-adjusted range restriction and KS score selection.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
GAMMA_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # no 0.7
TEMPS = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]  # T=1 is standard ML

REPORTED = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1, "yelpchi": 72.0,
    "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}
BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 78.1, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 83.5,
}


def ks_null(scores):
    s = scores[scores > 0]
    if len(s) < 10:
        return 0.0
    mu, var = np.mean(s), np.var(s)
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k = max(0.01, 2 * mu**2 / var)
    a = var / (2 * mu)
    try:
        stat, _ = kstest(s / a, 'chi2', args=(k,))
        return float(stat)
    except:
        return 0.0


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = data.edge_index.shape[1] // 2 / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean()), density


def get_gamma_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in GAMMA_GRID if g <= max_gamma]
    if not nearest:
        nearest = [GAMMA_GRID[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


def run_gamma(data, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="low", graph_type="original", gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl.numpy()
    S = compute_spectral_energy(delta, V)
    best_ml, rho, kappa = -np.inf, 0.5, 0.0
    for k in KAPPAS:
        r = solve_rho_newton(S, lam_L, k, d_eff)
        q = Q_rho_eigenvalues(lam_L, r, k)
        ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
        if ml > best_ml:
            best_ml, rho, kappa = ml, r, k
    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
    V_t = torch.from_numpy(V).float()
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()
    return je, jr, best_ml, rho


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None
    h, density = compute_graph_stats(data)
    gamma_range = get_gamma_range(h, density)

    # Compute scores at all gammas in range
    gamma_results = {}
    for gamma in gamma_range:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            auc = roc_auc_score(y[mask], sel[mask])
            gamma_results[gamma] = {
                "auc": auc, "ml": ml, "rho": rho,
                "score": "J*" if use_je else "R",
            }
        except:
            pass

    if not gamma_results:
        return None

    print("  h=%.3f dens=%.1f range=%s" % (h, density, gamma_range), flush=True)
    for g, r in sorted(gamma_results.items()):
        print("    g=%s: %s=%.1f%% ml=%.0f rho=%.3f" %
              (g, r["score"], r["auc"]*100, r["ml"], r["rho"]), flush=True)

    # Test each temperature
    results = {}
    for T in TEMPS:
        # Tempered ML: pick gamma with highest ML/T (= highest ML when T=1)
        # For negative ML, dividing by T makes values less negative (closer to 0)
        # So higher T makes ALL gammas look similar, reducing ML's preference
        best_tempered = -np.inf
        best_gamma = list(gamma_results.keys())[0]
        for g, r in gamma_results.items():
            tempered = r["ml"] / T
            if tempered > best_tempered:
                best_tempered = tempered
                best_gamma = g

        sel = gamma_results[best_gamma]
        results[T] = {
            "gamma": best_gamma, "auc": sel["auc"],
            "score": sel["score"], "rho": sel["rho"],
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    all_results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                all_results[ds] = r
                for T in TEMPS:
                    info = r[T]
                    print("  T=%5.1f: g=%s %s=%.1f%% rho=%.3f" %
                          (T, info["gamma"], info["score"], info["auc"]*100, info["rho"]),
                          flush=True)
        except Exception as e:
            print("  ERROR: %s" % str(e)[:80])

    if len(all_results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s" % "Dataset", end="")
        for T in TEMPS:
            print("  T=%-5s" % ("%.0f" % T), end="")
        print("  Base")
        for ds in datasets:
            if ds in all_results:
                print("%-15s" % ds, end="")
                for T in TEMPS:
                    auc = all_results[ds][T]["auc"] * 100
                    print(" %5.1f%%" % auc, end="")
                print(" %5.1f%%" % BASELINES.get(ds, 0))

        print("\nBeats baseline:")
        for T in TEMPS:
            cnt = sum(1 for ds, r in all_results.items()
                     if r[T]["auc"]*100 >= BASELINES.get(ds, 0))
            print("  T=%.0f: %d/%d" % (T, cnt, len(all_results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/tempered_%s.json" % args.dataset
               if args.dataset else "results/tempered.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
