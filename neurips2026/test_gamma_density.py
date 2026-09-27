"""Gamma selection: density-adjusted homophily center.

gamma_center = (0.5 + h) / sqrt(max(density, 10) / 10)

Dense + low-h -> small gamma (strong smoothing, many noisy neighbors)
Sparse + high-h -> large gamma (weak smoothing, features already smooth)

Search 3 nearest grid points to center, ML picks within that range.
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
GAMMA_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # no 0.7 (ML attractor)
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
    n_edges = edge_index.shape[1] // 2
    density = n_edges / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    homophily = float(cos.mean())
    return homophily, density


def get_gamma_range(h, density):
    """Compute gamma center from homophily + density, return nearest 3 grid points."""
    # Density-adjusted center
    # Dense graphs (density > 10): divide by sqrt(density/10), pushing gamma smaller
    # Sparse graphs (density <= 10): no adjustment (factor = 1.0)
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor

    # Clamp to reasonable range
    gamma_center = np.clip(gamma_center, 0.05, 10.0)

    # Include gammas up to 2x center (prevents ML from picking too-large gamma)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in GAMMA_GRID if g <= max_gamma]
    if not nearest:
        nearest = [GAMMA_GRID[0]]  # fallback to smallest
    # Keep at most 3 options
    if len(nearest) > 3:
        # Pick 3 closest to center
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return gamma_center, nearest


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
    gamma_center, gamma_range = get_gamma_range(h, density)

    print("  h=%.3f dens=%.1f -> center=%.3f range=%s" %
          (h, density, gamma_center, gamma_range), flush=True)

    best_ml = -np.inf
    best_result = None

    # Also run all gammas for oracle comparison
    all_results = {}
    for gamma in GAMMA_GRID:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            auc = roc_auc_score(y[mask], sel[mask])
            in_range = "*" if gamma in gamma_range else " "
            print("   %sg=%s: %s=%.1f%% ml=%.0f rho=%.3f" %
                  (in_range, gamma, "J*" if use_je else "R", auc*100, ml, rho),
                  flush=True)

            all_results[gamma] = {"auc": auc, "ml": ml, "rho": rho,
                                   "score": "J*" if use_je else "R"}

            if gamma in gamma_range and ml > best_ml:
                best_ml = ml
                best_result = all_results[gamma].copy()
                best_result["gamma"] = gamma
        except:
            pass

    oracle = max(all_results.values(), key=lambda r: r["auc"]) if all_results else None
    oracle_gamma = [g for g, r in all_results.items() if r["auc"] == oracle["auc"]][0] if oracle else None

    return {
        "selected": best_result,
        "oracle_auc": oracle["auc"] if oracle else 0,
        "oracle_gamma": oracle_gamma,
        "h": h, "density": density, "gamma_center": gamma_center,
        "gamma_range": gamma_range,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r and r["selected"]:
                sel = r["selected"]
                rep = REPORTED.get(ds, 0)
                base = BASELINES.get(ds, 0)
                print("  Selected: %s g=%s %.1f%% (range=%s center=%.2f)" %
                      (sel["score"], sel["gamma"], sel["auc"]*100,
                       r["gamma_range"], r["gamma_center"]))
                print("  Oracle: g=%s %.1f%%" % (r["oracle_gamma"], r["oracle_auc"]*100))
                print("  vsRep=%+.1f vsBase=%+.1f" %
                      (sel["auc"]*100 - rep, sel["auc"]*100 - base))
                results[ds] = r
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %5s %5s %6s %5s %6s %6s %7s %7s" %
              ("Dataset", "h", "dens", "center", "g*", "Sel", "Oracle", "vsRep", "vsBase"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                sel = r["selected"]
                rep = REPORTED.get(ds, 0)
                _, base = BASELINES.get(ds, ("?", 0))
                print("%-15s %5.3f %5.1f %6.3f %5s %5.1f%% %5.1f%% %+6.1f %+6.1f" %
                      (ds, r["h"], r["density"], r["gamma_center"],
                       sel["gamma"], sel["auc"]*100, r["oracle_auc"]*100,
                       sel["auc"]*100 - rep, sel["auc"]*100 - base))
        beats = sum(1 for ds, r in results.items()
                   if r["selected"]["auc"]*100 >= BASELINES[ds])
        within5 = sum(1 for r in results.values()
                     if abs(r["selected"]["auc"] - r["oracle_auc"]) < 0.05)
        print("\n  Beats baseline: %d/%d  Within 5pp of oracle: %d/%d" %
              (beats, len(results), within5, len(results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/gamma_density_%s.json" % args.dataset
               if args.dataset else "results/gamma_density.json")
    with open(outfile, "w") as f:
        json.dump({ds: {"h": r["h"], "density": r["density"],
                        "center": r["gamma_center"], "range": r["gamma_range"],
                        "sel_gamma": r["selected"]["gamma"],
                        "sel_auc": r["selected"]["auc"],
                        "oracle_auc": r["oracle_auc"]}
                   for ds, r in results.items()}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
