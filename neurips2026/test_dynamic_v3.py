"""Dynamic framework v3: ML for model params, KS for Gamma selection.

Two-stage approach:
  Stage 1: ML selects (gamma, rho, kappa) from stationary model (proven)
  Stage 2: At selected (gamma*, rho*, kappa*), sweep Gamma and pick
           the Gamma that maximizes KS null-deviation of the score.

This separates what ML is good at (model structure) from what it can't
do (scoring sharpness via time horizon).
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
    time_dependent_precision,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
GAMMA_T_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 1e6]  # 1e6 = stationary

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
    """KS departure from chi-squared null."""
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
    """Per-node J* and R scores at given Gamma."""
    qt = time_dependent_precision(q, Gamma_t)
    qt_safe = np.maximum(qt, 1e-12)
    delta_hat = V.T @ delta
    w = np.sqrt(qt_safe)[:, None] * delta_hat
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

    # ===== STAGE 1: ML selects (gamma, rho, kappa) from stationary model =====
    print("  --- Stage 1: ML model selection (stationary) ---", flush=True)
    best_ml = -np.inf
    best_gamma, best_rho, best_kappa = None, 0.5, 0.0
    best_V, best_lam_L, best_delta, best_S = None, None, None, None
    best_n, best_d = 0, 0

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
                best_gamma, best_rho, best_kappa = gamma, rho, kk
                best_V, best_lam_L, best_delta, best_S = V, lam_L, delta, S
                best_n, best_d = n, d

    print("  Stage 1 result: g=%s rho=%.3f kap=%.1f ml=%.0f" %
          (best_gamma, best_rho, best_kappa, best_ml), flush=True)

    # ===== STAGE 2: KS selects Gamma at fixed (gamma*, rho*, kappa*) =====
    print("  --- Stage 2: KS Gamma selection ---", flush=True)
    q_star = Q_rho_eigenvalues(best_lam_L, best_rho, best_kappa)

    best_ks = -1
    best_Gamma_t = 1e6
    best_scores = None
    best_sname = "J*"

    # Also track oracle for comparison
    oracle_auc = 0
    oracle_cfg = ""

    for Gamma_t in GAMMA_T_GRID:
        je, jr = score_nodes(best_delta, best_V, q_star, Gamma_t)
        if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
            continue

        auc_je = roc_auc_score(y[mask], je[mask])
        auc_jr = roc_auc_score(y[mask], jr[mask])
        ks_je = ks_null(je[mask])
        ks_jr = ks_null(jr[mask])

        # KS picks score mode
        ks_pick = "J*" if ks_je > ks_jr else "R"
        ks_val = max(ks_je, ks_jr)
        auc_ks = auc_je if ks_pick == "J*" else auc_jr

        Gt_str = "inf" if Gamma_t > 1e5 else str(Gamma_t)
        print("    G=%5s: J*=%.1f%% R=%.1f%% KS(%s)=%.4f  auc_ks=%.1f%%" %
              (Gt_str, auc_je*100, auc_jr*100, ks_pick, ks_val, auc_ks*100), flush=True)

        # KS selects Gamma
        if ks_val > best_ks:
            best_ks = ks_val
            best_Gamma_t = Gamma_t
            best_scores = je if ks_pick == "J*" else jr
            best_sname = ks_pick

        # Oracle tracking
        best_auc_this = max(auc_je, auc_jr)
        if best_auc_this > oracle_auc:
            oracle_auc = best_auc_this
            oracle_cfg = "G=%s %s" % (Gt_str, "J*" if auc_je > auc_jr else "R")

    # Final result
    Gt_final = "inf" if best_Gamma_t > 1e5 else str(best_Gamma_t)
    final_auc = roc_auc_score(y[mask], best_scores[mask])

    # Stationary baseline (Gamma=inf at same params)
    je_stat, jr_stat = score_nodes(best_delta, best_V, q_star, 1e6)
    ks_je_s = ks_null(je_stat[mask])
    ks_jr_s = ks_null(jr_stat[mask])
    sel_stat = je_stat if ks_je_s > ks_jr_s else jr_stat
    auc_stat = roc_auc_score(y[mask], sel_stat[mask])

    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)
    print("  ---")
    print("  KS-selected:     %.1f%%  G=%s %s (KS=%.4f)" %
          (final_auc*100, Gt_final, best_sname, best_ks))
    print("  Stationary:      %.1f%%" % (auc_stat*100))
    print("  Oracle:          %.1f%%  %s" % (oracle_auc*100, oracle_cfg))
    print("  Baseline:        %.1f%%  Current: %.1f%%" % (base, curr))
    print("  KS vs Stat:     %+.1f%%" % (final_auc*100 - auc_stat*100))
    print("  KS vs Current:  %+.1f%%" % (final_auc*100 - curr))
    print("  KS vs Base:     %+.1f%%" % (final_auc*100 - base))

    return {
        "ks_auc": final_auc*100,
        "ks_Gamma": Gt_final,
        "stat_auc": auc_stat*100,
        "oracle_auc": oracle_auc*100,
        "baseline": base,
        "current": curr,
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
        print("%-20s %8s %8s %8s %6s %6s  %s" % (
            "Dataset", "KS-Dyn", "Static", "Oracle", "Base", "Diff", "Gamma"))
        for ds in datasets:
            if ds in summary:
                s = summary[ds]
                diff = s["ks_auc"] - s["stat_auc"]
                print("%-20s %7.1f%% %7.1f%% %7.1f%% %5.1f%% %+5.1f%%  G=%s" % (
                    ds, s["ks_auc"], s["stat_auc"], s["oracle_auc"],
                    s["baseline"], diff, s["ks_Gamma"]))


if __name__ == "__main__":
    main()
