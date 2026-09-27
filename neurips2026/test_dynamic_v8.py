"""Dynamic framework v8: Fixed-Gamma ML pipeline comparison.

Hypothesis: At fixed finite Gamma, ML selects different (gamma, rho, kappa)
than at Gamma=inf. The precision floor from finite Gamma may encourage
higher rho (more graph trust) without over-tightening.

Runs the FULL ML pipeline (density-adjusted gamma, optimize rho/kappa)
at several fixed Gamma values and compares to stationary.

This tells us: does fixing Gamma change ML's structural decisions?
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    marginal_log_likelihood_dynamic, time_dependent_precision,
    compute_spectral_energy, solve_rho_newton, solve_rho_dynamic,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
GAMMA_T_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 70.6, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 63.1,
}
CURRENT = {
    "weibo": 95.1, "reddit": 61.9, "amazon": 75.8, "yelpchi": 70.7,
    "blogcatalog": 74.8, "facebook": 85.9, "acm": 78.4,
    "elliptic": 73.4, "elliptic_plus_plus": 72.6, "t_finance": 82.8,
}


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


def score_nodes(delta, V, q, Gamma_t):
    if Gamma_t is None or Gamma_t > 1e5:
        qt = np.maximum(q, 1e-12)
    else:
        qt = time_dependent_precision(q, Gamma_t)
    delta_hat = V.T @ delta
    w = np.sqrt(qt)[:, None] * delta_hat
    je = np.sum((V @ w)**2, axis=1)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot
    return je, jr


def run_pipeline_at_gamma_t(gamma_cache, density_range, Gamma_t, y, mask):
    """Run the full ML pipeline at a fixed Gamma_t.
    Returns the ML-best config and its scores.
    """
    is_stat = (Gamma_t is None)
    best_ml = -np.inf
    best_result = None

    for gamma in density_range:
        gc = gamma_cache[gamma]
        V, lam_L, delta, S, n, d = gc["V"], gc["lam_L"], gc["delta"], gc["S"], gc["n"], gc["d"]

        for kk in KAPPAS:
            if is_stat:
                rho = solve_rho_newton(S, lam_L, kk, d)
                q = Q_rho_eigenvalues(lam_L, rho, kk)
                ml = marginal_log_likelihood_stationary(S, q, n, d)
            else:
                rho = solve_rho_dynamic(S, lam_L, kk, Gamma_t, d)
                q = Q_rho_eigenvalues(lam_L, rho, kk)
                ml = marginal_log_likelihood_dynamic(S, q, Gamma_t, n, d)

            if ml > best_ml:
                best_ml = ml
                je, jr = score_nodes(delta, V, q, Gamma_t)
                if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                    continue
                auc_je = roc_auc_score(y[mask], je[mask])
                auc_jr = roc_auc_score(y[mask], jr[mask])
                ks_je = ks_null(je[mask])
                ks_jr = ks_null(jr[mask])
                ks_pick = "J*" if ks_je > ks_jr else "R"
                auc_ks = auc_je if ks_pick == "J*" else auc_jr
                best_result = {
                    "ml": ml, "gamma": gamma, "rho": rho, "kappa": kk,
                    "auc_ks": auc_ks, "ks_pick": ks_pick,
                    "auc_je": auc_je, "auc_jr": auc_jr,
                    "ks_val": max(ks_je, ks_jr),
                }

    return best_result


def run_dataset(ds, device="cpu"):
    print("\n=== %s ===" % ds, flush=True)
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d_raw = data.x.shape[1]
    pca_dim = 64 if d_raw > 100 else None

    h, density = compute_graph_stats(data)
    density_range = get_density_range(h, density)
    print("  h=%.3f density=%.1f gamma_range=%s" % (h, density, density_range), flush=True)

    # Precompute
    gamma_cache = {}
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
        gamma_cache[gamma] = {"V": V, "lam_L": lam_L, "delta": delta, "S": S, "n": n, "d": d}

    # Run pipeline at each Gamma_t
    results = {}
    print("  --- Fixed-Gamma ML pipeline ---", flush=True)

    # Stationary
    r = run_pipeline_at_gamma_t(gamma_cache, density_range, None, y, mask)
    if r:
        results["inf"] = r
        print("  G=inf:  %s=%.1f%% g=%s rho=%.3f kap=%.1f" % (
            r["ks_pick"], r["auc_ks"]*100, r["gamma"], r["rho"], r["kappa"]), flush=True)

    # Finite Gamma
    for Gt in GAMMA_T_VALUES:
        r = run_pipeline_at_gamma_t(gamma_cache, density_range, Gt, y, mask)
        if r:
            results[str(Gt)] = r
            print("  G=%4s: %s=%.1f%% g=%s rho=%.3f kap=%.1f" % (
                Gt, r["ks_pick"], r["auc_ks"]*100, r["gamma"], r["rho"], r["kappa"]), flush=True)

    # Summary
    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)
    stat_auc = results.get("inf", {}).get("auc_ks", 0)

    print("  ---")
    print("  Stationary:  %.1f%%" % (stat_auc*100))

    # Find best dynamic (highest AUC among finite Gamma)
    best_dyn = max(
        [(k, v) for k, v in results.items() if k != "inf"],
        key=lambda x: x[1]["auc_ks"], default=(None, None))

    if best_dyn[1]:
        print("  Best dynamic: %.1f%%  G=%s (rho=%.3f kap=%.1f)" % (
            best_dyn[1]["auc_ks"]*100, best_dyn[0],
            best_dyn[1]["rho"], best_dyn[1]["kappa"]))
        print("  Dyn vs Stat: %+.1f%%" % (best_dyn[1]["auc_ks"]*100 - stat_auc*100))

    # Check if ML-selected params differ across Gamma values
    print("  --- Parameter variation across Gamma ---")
    for Gt_str in ["inf"] + [str(g) for g in GAMMA_T_VALUES]:
        if Gt_str in results:
            r = results[Gt_str]
            print("    G=%5s: g=%s rho=%.4f kap=%4.1f → %s %.1f%%" % (
                Gt_str, r["gamma"], r["rho"], r["kappa"],
                r["ks_pick"], r["auc_ks"]*100))

    print("  Baseline: %.1f%%  Current: %.1f%%" % (base, curr))

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog",
                     "facebook", "acm", "elliptic", "elliptic_plus_plus", "t_finance"]
    else:
        datasets = [args.dataset]

    summary = {}
    for ds in datasets:
        try:
            summary[ds] = run_dataset(ds, args.device)
        except Exception as e:
            import traceback
            traceback.print_exc()

    if len(summary) > 1:
        print("\n" + "="*80)
        print("SUMMARY: Does finite Gamma change ML's parameter selection?")
        print("="*80)
        print("%-18s %6s | %s" % ("Dataset", "G=inf", " | ".join("G=%-4s" % g for g in GAMMA_T_VALUES)))
        for ds in datasets:
            if ds in summary:
                vals = []
                for Gt_str in ["inf"] + [str(g) for g in GAMMA_T_VALUES]:
                    if Gt_str in summary[ds]:
                        vals.append("%.1f" % (summary[ds][Gt_str]["auc_ks"]*100))
                    else:
                        vals.append("  -  ")
                print("%-18s %s" % (ds, " | ".join("%6s" % v for v in vals)))

        # Show rho variation
        print("\nRho values (does Gamma change graph trust?):")
        print("%-18s %6s | %s" % ("Dataset", "G=inf", " | ".join("G=%-4s" % g for g in GAMMA_T_VALUES)))
        for ds in datasets:
            if ds in summary:
                vals = []
                for Gt_str in ["inf"] + [str(g) for g in GAMMA_T_VALUES]:
                    if Gt_str in summary[ds]:
                        vals.append("%.3f" % summary[ds][Gt_str]["rho"])
                    else:
                        vals.append("  -  ")
                print("%-18s %s" % (ds, " | ".join("%6s" % v for v in vals)))


if __name__ == "__main__":
    main()
