"""Dynamic framework v2: Current pipeline + Gamma as only new parameter.

Uses the proven density-adjusted gamma range, eta=0 (hard matching),
KS score selection. Adds Gamma (time horizon) to the ML search.

Compares:
  1. Stationary (Gamma=inf) -- current pipeline
  2. Dynamic (finite Gamma) -- new
  3. ML selects best (gamma, Gamma, kappa) jointly
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
    compute_spectral_energy, solve_rho_newton,
    time_dependent_precision, marginal_log_likelihood_dynamic,
    solve_rho_dynamic,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
# Gamma_t grid: from finite to stationary
GAMMA_T_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

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


def score_nodes_dynamic(delta, V, q, Gamma_t):
    """Per-node scores with time-dependent precision (eta=0)."""
    qt = time_dependent_precision(q, Gamma_t)
    qt_safe = np.maximum(qt, 1e-12)
    delta_hat = V.T @ delta
    w = np.sqrt(qt_safe)[:, None] * delta_hat
    je = np.sum((V @ w)**2, axis=1)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot
    return je, jr


def score_nodes_stationary(delta, V, q):
    """Per-node scores with stationary precision."""
    q_safe = np.maximum(q, 1e-12)
    delta_hat = V.T @ delta
    w = np.sqrt(q_safe)[:, None] * delta_hat
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

    # --- Compute eigenpairs/template/residuals for density-adjusted gammas ---
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

    # --- Sweep (gamma, Gamma_t, kappa), optimize rho, track ML-best and AUC-best ---
    ml_best = {"ml": -np.inf, "auc_je": 0, "auc_jr": 0, "cfg": "",
               "je": None, "jr": None}
    auc_best = {"auc": 0, "cfg": ""}

    for gamma in density_range:
        gd = gamma_data[gamma]
        V, lam_L, delta, S = gd["V"], gd["lam_L"], gd["delta"], gd["S"]
        n, d = gd["n"], gd["d"]

        # Stationary (Gamma=inf) + Dynamic (finite Gamma)
        for Gamma_t in GAMMA_T_GRID + [None]:  # None = stationary
            is_stat = (Gamma_t is None)
            best_ml_this, best_rho, best_kappa = -np.inf, 0.5, 0.0

            for kk in KAPPAS:
                if is_stat:
                    rho = solve_rho_newton(S, lam_L, kk, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = marginal_log_likelihood_stationary(S, q, n, d)
                else:
                    rho = solve_rho_dynamic(S, lam_L, kk, Gamma_t, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = marginal_log_likelihood_dynamic(S, q, Gamma_t, n, d)

                if ml > best_ml_this:
                    best_ml_this, best_rho, best_kappa = ml, rho, kk

            q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
            if is_stat:
                je, jr = score_nodes_stationary(delta, V, q)
            else:
                je, jr = score_nodes_dynamic(delta, V, q, Gamma_t)

            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue

            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            ks_pick = "J*" if ks_je > ks_jr else "R"
            auc_ks = auc_je if ks_pick == "J*" else auc_jr
            auc_oracle = max(auc_je, auc_jr)

            Gt_str = "inf" if is_stat else str(Gamma_t)
            cfg_str = "g=%s G=%s %s rho=%.3f kap=%.1f" % (
                gamma, Gt_str, ks_pick, best_rho, best_kappa)

            # Track oracle (best AUC)
            if auc_oracle > auc_best["auc"]:
                auc_best = {"auc": auc_oracle, "cfg": cfg_str.replace(ks_pick,
                            "J*" if auc_je > auc_jr else "R")}

            # Track ML-selected
            if best_ml_this > ml_best["ml"]:
                ml_best = {"ml": best_ml_this, "auc_je": auc_je, "auc_jr": auc_jr,
                           "cfg": cfg_str, "je": je, "jr": jr,
                           "ks_pick": ks_pick, "auc_ks": auc_ks}

            print("    g=%s G=%s: J*=%.1f%% R=%.1f%% KS=%s=%.1f%% rho=%.3f kap=%.1f ml=%.0f" %
                  (gamma, Gt_str, auc_je*100, auc_jr*100, ks_pick, auc_ks*100,
                   best_rho, best_kappa, best_ml_this), flush=True)

    # --- Final scoring with ML-selected config + KS ---
    ml_auc = ml_best["auc_ks"]
    ml_cfg = ml_best["cfg"]

    # Also compute stationary-only ML-select for comparison
    stat_ml_best = {"ml": -np.inf, "auc": 0, "cfg": ""}
    for gamma in density_range:
        gd = gamma_data[gamma]
        V, lam_L, delta, S = gd["V"], gd["lam_L"], gd["delta"], gd["S"]
        n, d = gd["n"], gd["d"]
        best_ml_s, rho_s, kap_s = -np.inf, 0.5, 0.0
        for kk in KAPPAS:
            r = solve_rho_newton(S, lam_L, kk, d)
            q = Q_rho_eigenvalues(lam_L, r, kk)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml_s: best_ml_s, rho_s, kap_s = ml, r, kk
        if best_ml_s > stat_ml_best["ml"]:
            q = Q_rho_eigenvalues(lam_L, rho_s, kap_s)
            je, jr = score_nodes_stationary(delta, V, q)
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            sel = je if ks_je > ks_jr else jr
            auc = roc_auc_score(y[mask], sel[mask])
            stat_ml_best = {"ml": best_ml_s, "auc": auc,
                            "cfg": "g=%s G=inf rho=%.3f kap=%.1f" % (gamma, rho_s, kap_s)}

    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)
    print("  ---")
    print("  Oracle (AUC):      %.1f%%  %s" % (auc_best["auc"]*100, auc_best["cfg"]))
    print("  ML-select (dyn):   %.1f%%  %s" % (ml_auc*100, ml_cfg))
    print("  ML-select (stat):  %.1f%%  %s" % (stat_ml_best["auc"]*100, stat_ml_best["cfg"]))
    print("  Baseline:          %.1f%%  Current pipeline: %.1f%%" % (base, curr))
    print("  Dyn vs Stat ML:   %+.1f%%" % (ml_auc*100 - stat_ml_best["auc"]*100))
    print("  Dyn ML vs Base:   %+.1f%%" % (ml_auc*100 - base))

    return {
        "oracle": auc_best["auc"]*100,
        "ml_dyn": ml_auc*100,
        "ml_dyn_cfg": ml_cfg,
        "ml_stat": stat_ml_best["auc"]*100,
        "ml_stat_cfg": stat_ml_best["cfg"],
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
            print("  FAILED: %s" % str(e)[:100])

    if len(summary) > 1:
        print("\n" + "="*80)
        print("%-20s %8s %8s %8s %8s  %s" % (
            "Dataset", "Oracle", "DynML", "StatML", "Base", "DynML config"))
        for ds in datasets:
            if ds in summary:
                s = summary[ds]
                print("%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%  %s" % (
                    ds, s["oracle"], s["ml_dyn"], s["ml_stat"],
                    BASELINES.get(ds, 0), s["ml_dyn_cfg"]))


if __name__ == "__main__":
    main()
