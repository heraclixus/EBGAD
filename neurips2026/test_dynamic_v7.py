"""Dynamic framework v7: Tail-excess criterion + per-gamma Gamma selection.

Hypotheses:
  H1: Right-tail excess (fraction of scores above null's 95th/99th pctile)
      is a better unsupervised criterion than KS for selecting Gamma.
  H2: Selecting Gamma independently per gamma, then using ML across gammas,
      may work better than global selection.

Pipeline:
  For each gamma in density range:
    For each Gamma in grid + inf:
      ML optimizes (rho, kappa)
      Compute scores, fit chi-squared null, measure tail excess
    Pick Gamma by tail excess (within this gamma)
    Record ML for the best Gamma's config
  Pick gamma by ML across the per-gamma winners

Also tests: KS, skewness, kurtosis as alternative criteria.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest, chi2

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


def tail_excess(scores, pctile=95):
    """Fraction of scores exceeding the chi-squared null's pctile-th percentile.
    Higher = more anomalies detected = better.
    """
    s = scores[scores > 0]
    if len(s) < 20: return 0.0
    mu, var = s.mean(), s.var()
    if var < 1e-12 or mu < 1e-12: return 0.0
    k = max(0.01, 2*mu**2/var)
    a = var/(2*mu)
    try:
        threshold = chi2.ppf(pctile/100.0, k) * a
        return float(np.mean(s > threshold))
    except:
        return 0.0


def score_skewness(scores):
    """Skewness of positive scores. Higher = more right-tail weight."""
    s = scores[scores > 0]
    if len(s) < 10: return 0.0
    mu, std = s.mean(), s.std()
    if std < 1e-12: return 0.0
    return float(np.mean(((s - mu) / std) ** 3))


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

    # For each gamma, sweep Gamma and compute all criteria
    per_gamma_results = {}

    for gamma in density_range:
        gc = gamma_cache[gamma]
        V, lam_L, delta, S, n, d = gc["V"], gc["lam_L"], gc["delta"], gc["S"], gc["n"], gc["d"]

        gamma_configs = []
        for Gt in GAMMA_T_GRID + [None]:
            is_stat = (Gt is None)
            best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
            for kk in KAPPAS:
                if is_stat:
                    rho = solve_rho_newton(S, lam_L, kk, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = marginal_log_likelihood_stationary(S, q, n, d)
                else:
                    rho = solve_rho_dynamic(S, lam_L, kk, Gt, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = marginal_log_likelihood_dynamic(S, q, Gt, n, d)
                if ml > best_ml:
                    best_ml, best_rho, best_kappa = ml, rho, kk

            q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
            je, jr = score_nodes(delta, V, q, Gt)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue

            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])

            # KS-based score selection
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            ks_pick = "J*" if ks_je > ks_jr else "R"
            sel_scores = je[mask] if ks_pick == "J*" else jr[mask]
            auc_ks = auc_je if ks_pick == "J*" else auc_jr

            # Compute criteria on KS-selected scores
            te95 = tail_excess(sel_scores, 95)
            te99 = tail_excess(sel_scores, 99)
            skew = score_skewness(sel_scores)
            ks_val = max(ks_je, ks_jr)

            Gt_str = "inf" if is_stat else str(Gt)
            print("    g=%s G=%5s: %s=%.1f%% KS=%.4f TE95=%.4f TE99=%.4f skew=%.2f rho=%.3f kap=%.1f" %
                  (gamma, Gt_str, ks_pick, auc_ks*100, ks_val, te95, te99, skew,
                   best_rho, best_kappa), flush=True)

            gamma_configs.append({
                "Gt": Gt, "Gt_str": Gt_str, "is_stat": is_stat,
                "rho": best_rho, "kappa": best_kappa, "ml": best_ml,
                "auc_ks": auc_ks, "ks_pick": ks_pick, "ks_val": ks_val,
                "te95": te95, "te99": te99, "skew": skew,
                "auc_je": auc_je, "auc_jr": auc_jr,
            })

        per_gamma_results[gamma] = gamma_configs

    # ===== Selection strategies =====
    print("\n  --- Per-gamma Gamma selection, then ML across gammas ---", flush=True)

    criteria = {
        "ML": lambda c: c["ml"],
        "KS": lambda c: c["ks_val"],
        "TE95": lambda c: c["te95"],
        "TE99": lambda c: c["te99"],
        "Skew": lambda c: c["skew"],
    }

    results = {}
    for crit_name, crit_fn in criteria.items():
        # For each gamma: pick Gamma by criterion
        gamma_winners = {}
        for gamma in density_range:
            configs = per_gamma_results.get(gamma, [])
            if not configs:
                continue
            winner = max(configs, key=crit_fn)
            gamma_winners[gamma] = winner

        if not gamma_winners:
            continue

        # Pick gamma by ML among the per-gamma winners
        best_gamma = max(gamma_winners.keys(), key=lambda g: gamma_winners[g]["ml"])
        final = gamma_winners[best_gamma]

        results[crit_name] = {
            "auc": final["auc_ks"],
            "gamma": best_gamma,
            "Gt_str": final["Gt_str"],
            "rho": final["rho"],
            "kappa": final["kappa"],
            "ks_pick": final["ks_pick"],
        }

        print("  %-6s: %.1f%%  g=%s G=%s %s rho=%.3f kap=%.1f" % (
            crit_name, final["auc_ks"]*100, best_gamma, final["Gt_str"],
            final["ks_pick"], final["rho"], final["kappa"]))

    # Also: global selection (all configs, pick by each criterion, then KS for score)
    print("\n  --- Global selection (across all gamma x Gamma) ---", flush=True)
    all_configs = []
    for gamma in density_range:
        for c in per_gamma_results.get(gamma, []):
            c["gamma"] = gamma
            all_configs.append(c)

    for crit_name, crit_fn in criteria.items():
        winner = max(all_configs, key=crit_fn)
        print("  %-6s: %.1f%%  g=%s G=%s %s rho=%.3f kap=%.1f" % (
            crit_name, winner["auc_ks"]*100, winner["gamma"], winner["Gt_str"],
            winner["ks_pick"], winner["rho"], winner["kappa"]))

    # Oracle
    oracle = max(all_configs, key=lambda c: max(c["auc_je"], c["auc_jr"]))
    oracle_auc = max(oracle["auc_je"], oracle["auc_jr"])

    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)
    stat_ml = results.get("ML", {}).get("auc", 0) * 100 if "ML" in results else 0
    print("\n  Oracle:      %.1f%%  g=%s G=%s" % (oracle_auc*100, oracle["gamma"], oracle["Gt_str"]))
    print("  Baseline:    %.1f%%  Current: %.1f%%" % (base, curr))

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
        print("SUMMARY: Per-gamma Gamma selection (by criterion), then ML across gammas")
        print("="*80)
        for crit in ["ML", "KS", "TE95", "TE99", "Skew"]:
            print("\n--- %s for Gamma, ML for gamma ---" % crit)
            print("%-18s %7s %6s  %s" % ("Dataset", "AUC", "Base", "Config"))
            for ds in datasets:
                if ds in summary and crit in summary[ds]:
                    r = summary[ds][crit]
                    print("%-18s %6.1f%% %5.1f%%  g=%s G=%s %s rho=%.3f" % (
                        ds, r["auc"]*100, BASELINES.get(ds, 0),
                        r["gamma"], r["Gt_str"], r["ks_pick"], r["rho"]))


if __name__ == "__main__":
    main()
