"""Dynamic framework v4: ML within each Gamma, KS across Gamma.

Key insight: ML always prefers Gamma=inf in joint optimization. But at each
fixed Gamma, ML selects DIFFERENT (rho, kappa). The finite-Gamma precision
floor may let ML choose higher rho (more graph trust) without over-tightening.

Approach:
  For each (gamma, Gamma):
    1. ML optimizes (rho, kappa) conditioned on this Gamma
    2. Compute J* and R scores at the ML-optimal params
    3. Compute KS null-deviation for both scores
  Select the (gamma, Gamma) with highest max(KS_J*, KS_R)
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
GAMMA_T_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]  # finite Gamma values

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
    """Per-node J* and R scores."""
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

    # Precompute eigenpairs/template/residuals for each gamma
    gamma_data = {}
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
        gamma_data[gamma] = {"V": V, "lam_L": lam_L, "delta": delta, "S": S, "n": n, "d": d}

    # For each (gamma, Gamma_t): ML selects (rho, kappa), then compute KS
    # Gamma_t = None means stationary
    all_configs = []

    for gamma in density_range:
        gd = gamma_data[gamma]
        V, lam_L, delta, S = gd["V"], gd["lam_L"], gd["delta"], gd["S"]
        n, d = gd["n"], gd["d"]

        for Gamma_t in GAMMA_T_GRID + [None]:
            is_stat = (Gamma_t is None)

            # ML-optimize (rho, kappa) for this fixed Gamma
            best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
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
                    best_ml, best_rho, best_kappa = ml, rho, kk

            # Score at ML-optimal params
            q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
            je, jr = score_nodes(delta, V, q, Gamma_t)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue

            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            ks_best = max(ks_je, ks_jr)
            ks_pick = "J*" if ks_je > ks_jr else "R"
            auc_ks = auc_je if ks_pick == "J*" else auc_jr

            Gt_str = "inf" if is_stat else str(Gamma_t)
            print("    g=%s G=%5s: J*=%.1f%% R=%.1f%% KS(%s)=%.4f rho=%.3f kap=%.1f" %
                  (gamma, Gt_str, auc_je*100, auc_jr*100, ks_pick, ks_best,
                   best_rho, best_kappa), flush=True)

            all_configs.append({
                "gamma": gamma, "Gamma_t": Gamma_t, "Gt_str": Gt_str,
                "rho": best_rho, "kappa": best_kappa, "ml": best_ml,
                "auc_je": auc_je, "auc_jr": auc_jr,
                "ks_je": ks_je, "ks_jr": ks_jr, "ks_best": ks_best,
                "ks_pick": ks_pick, "auc_ks": auc_ks,
                "is_stat": is_stat,
            })

    if not all_configs:
        print("  No valid configs!")
        return None

    # === Selection strategies ===
    # 1. KS across all (gamma, Gamma) -- the proposed method
    ks_selected = max(all_configs, key=lambda c: c["ks_best"])

    # 2. ML across all -- always picks stationary (baseline)
    ml_selected = max(all_configs, key=lambda c: c["ml"])

    # 3. Oracle (best AUC)
    oracle = max(all_configs, key=lambda c: max(c["auc_je"], c["auc_jr"]))
    oracle_auc = max(oracle["auc_je"], oracle["auc_jr"])

    # 4. Stationary-only ML (current pipeline)
    stat_configs = [c for c in all_configs if c["is_stat"]]
    stat_ml = max(stat_configs, key=lambda c: c["ml"]) if stat_configs else ks_selected

    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)

    print("  ---")
    print("  KS-select:    %.1f%%  g=%s G=%s %s rho=%.3f kap=%.1f (KS=%.4f)" % (
        ks_selected["auc_ks"]*100, ks_selected["gamma"], ks_selected["Gt_str"],
        ks_selected["ks_pick"], ks_selected["rho"], ks_selected["kappa"],
        ks_selected["ks_best"]))
    print("  ML-select:    %.1f%%  g=%s G=%s %s rho=%.3f kap=%.1f" % (
        ml_selected["auc_ks"]*100, ml_selected["gamma"], ml_selected["Gt_str"],
        ml_selected["ks_pick"], ml_selected["rho"], ml_selected["kappa"]))
    print("  Stat-ML:      %.1f%%  g=%s rho=%.3f kap=%.1f" % (
        stat_ml["auc_ks"]*100, stat_ml["gamma"], stat_ml["rho"], stat_ml["kappa"]))
    print("  Oracle:       %.1f%%  g=%s G=%s rho=%.3f kap=%.1f" % (
        oracle_auc*100, oracle["gamma"], oracle["Gt_str"],
        oracle["rho"], oracle["kappa"]))
    print("  Baseline:     %.1f%%  Current: %.1f%%" % (base, curr))
    print("  KS vs Stat:  %+.1f%%  KS vs Current: %+.1f%%  KS vs Base: %+.1f%%" % (
        ks_selected["auc_ks"]*100 - stat_ml["auc_ks"]*100,
        ks_selected["auc_ks"]*100 - curr,
        ks_selected["auc_ks"]*100 - base))

    return {
        "ks": ks_selected["auc_ks"]*100,
        "ks_cfg": "g=%s G=%s %s rho=%.3f kap=%.1f" % (
            ks_selected["gamma"], ks_selected["Gt_str"],
            ks_selected["ks_pick"], ks_selected["rho"], ks_selected["kappa"]),
        "ks_Gamma": ks_selected["Gt_str"],
        "stat": stat_ml["auc_ks"]*100,
        "oracle": oracle_auc*100,
    }


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
        print("%-20s %7s %7s %7s %6s %6s  %s" % (
            "Dataset", "KS-Dyn", "StatML", "Oracle", "Base", "KSvSt", "Gamma"))
        for ds in datasets:
            if ds in summary:
                s = summary[ds]
                diff = s["ks"] - s["stat"]
                print("%-20s %6.1f%% %6.1f%% %6.1f%% %5.1f%% %+5.1f%%  G=%s" % (
                    ds, s["ks"], s["stat"], s["oracle"],
                    BASELINES.get(ds, 0), diff, s["ks_Gamma"]))


if __name__ == "__main__":
    main()
