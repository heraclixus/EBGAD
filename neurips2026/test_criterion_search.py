"""Systematic criterion search on the restricted v2 grid.

Grid: original graph, {low, affinity} template, 7 gammas, PCA=64 if d>100.
For each candidate (template × gamma × {J*, R}), compute many criteria.
Test which criterion best selects the oracle config.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis, spearmanr, iqr

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
GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]
TEMPLATES = ["low", "affinity"]

ORACLE_AUROCS = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1,
    "yelpchi": 72.0, "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}


def gini(s):
    s = np.sort(s)
    n = len(s)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) / (n * np.sum(s) + 1e-12)) - (n + 1) / n)


def compute_all_criteria(scores, ml_val):
    """Compute every criterion we can think of for a score array."""
    s = scores
    n = len(s)
    s_sorted = np.sort(s)

    g = gini(s)

    # Trimmed Gini (remove top 1%)
    k99 = max(1, int(n * 0.99))
    tg = gini(s_sorted[:k99])

    # Trimmed Gini (remove top 0.1%)
    k999 = max(1, int(n * 0.999))
    tg2 = gini(s_sorted[:k999])

    # Gini robustness
    g_robust = tg / (g + 1e-12)

    # Skewness, kurtosis
    sk = float(skew(s))
    ku = float(kurtosis(s))

    # Percentile ratios
    p50 = np.percentile(s, 50)
    p90 = np.percentile(s, 90)
    p95 = np.percentile(s, 95)
    p99 = np.percentile(s, 99)
    p999 = np.percentile(s, 99.9)
    p95_p50 = p95 / (p50 + 1e-12)
    p99_p50 = p99 / (p50 + 1e-12)
    p99_p90 = p99 / (p90 + 1e-12)

    # Tail mass at different thresholds
    mu, sigma = np.mean(s), np.std(s)
    tail_2s = float(np.mean(s > mu + 2 * sigma))
    tail_3s = float(np.mean(s > mu + 3 * sigma))
    tail_5s = float(np.mean(s > mu + 5 * sigma))

    # Max gap (normalized)
    gaps = np.diff(s_sorted)
    rng = s_sorted[-1] - s_sorted[0]
    max_gap = float(np.max(gaps) / (rng + 1e-12)) if rng > 0 else 0

    # IQR ratio (tail spread vs bulk spread)
    iqr_val = float(iqr(s))
    tail_spread = p99 - p90
    iqr_ratio = tail_spread / (iqr_val + 1e-12)

    # Coefficient of variation
    cv = float(sigma / (mu + 1e-12))

    # Bimodality coefficient
    bimod = (sk**2 + 1) / (ku + 3 + 1e-12)

    # Score entropy (discretize into 100 bins)
    hist, _ = np.histogram(s, bins=100, density=True)
    hist = hist / (hist.sum() + 1e-12)
    entropy = float(-np.sum(hist[hist > 0] * np.log(hist[hist > 0])))

    # Negative entropy (higher = more concentrated = better)
    neg_entropy = -entropy

    return {
        "gini": g, "trimmed_gini": tg, "trimmed_gini_999": tg2,
        "gini_robust": g_robust,
        "skewness": sk, "kurtosis": ku,
        "p95_p50": p95_p50, "p99_p50": p99_p50, "p99_p90": p99_p90,
        "tail_2s": tail_2s, "tail_3s": tail_3s, "tail_5s": tail_5s,
        "max_gap": max_gap, "iqr_ratio": iqr_ratio,
        "cv": cv, "bimodality": bimod, "neg_entropy": neg_entropy,
        "ml": ml_val,
        # Combined criteria
        "ml_x_tgini": ml_val * tg if ml_val > 0 else ml_val / (tg + 1e-12),
        "ml_x_gini": ml_val * g if ml_val > 0 else ml_val / (g + 1e-12),
    }


def run_config(data, tmpl, gamma, pca_dim, device="cpu"):
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type=tmpl, graph_type="original", gamma=gamma,
    )
    if pca_dim is not None and data.x.shape[1] > pca_dim:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=pca_dim, encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))
    elif pca_dim is not None:
        return None
    try:
        trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl_t = trainer.template.cpu()
        x = trainer.x_T.cpu()
        n, d = x.shape
        delta = x.numpy() - tmpl_t.numpy()
        S = compute_spectral_energy(delta, V)
        best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa
        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t = torch.from_numpy(V).float()
        with torch.no_grad():
            je = precision_energy_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()
        return {"je": je, "jr": jr, "ml": best_ml, "rho": best_rho}
    except:
        return None


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    candidates = []
    for tmpl in TEMPLATES:
        for gamma in GAMMAS:
            result = run_config(data, tmpl, gamma, pca_dim, device)
            if result is None:
                continue
            je, jr = result["je"], result["jr"]
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue

            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            tag = "%s g=%s" % (tmpl, gamma)

            for sname, scores, auc in [("J*", je, auc_je), ("R", jr, auc_jr)]:
                crit = compute_all_criteria(scores[mask], result["ml"])
                crit["auc"] = auc
                crit["tag"] = "%s %s" % (sname, tag)
                crit["rho"] = result["rho"]
                candidates.append(crit)

            print("    %s: J*=%.1f%% R=%.1f%% ml=%.0f rho=%.3f" %
                  (tag, auc_je*100, auc_jr*100, result["ml"], result["rho"]),
                  flush=True)

    if not candidates:
        return None

    oracle_auc = max(c["auc"] for c in candidates)
    oracle_tag = max(candidates, key=lambda c: c["auc"])["tag"]

    # Test every criterion
    all_criteria = [
        "gini", "trimmed_gini", "trimmed_gini_999", "gini_robust",
        "skewness", "kurtosis",
        "p95_p50", "p99_p50", "p99_p90",
        "tail_2s", "tail_3s", "tail_5s",
        "iqr_ratio", "cv", "bimodality", "neg_entropy",
        "ml", "ml_x_tgini", "ml_x_gini",
    ]
    # Also add: max_gap (lower = better)
    all_criteria_inv = ["max_gap"]

    results = {"oracle_auc": oracle_auc, "oracle_tag": oracle_tag,
               "n_candidates": len(candidates)}

    for cname in all_criteria:
        best = max(candidates, key=lambda c: c[cname])
        results[cname] = {"auc": best["auc"], "tag": best["tag"], "val": best[cname]}

    for cname in all_criteria_inv:
        best = min(candidates, key=lambda c: c[cname])
        results[cname] = {"auc": best["auc"], "tag": best["tag"], "val": best[cname]}

    # Also: top-5 by ML, then pick by trimmed_gini
    ml_sorted = sorted(candidates, key=lambda c: c["ml"], reverse=True)
    for topk in [3, 5, 10]:
        pool = ml_sorted[:min(topk, len(candidates))]
        best = max(pool, key=lambda c: c["trimmed_gini"])
        results["ml_top%d_tgini" % topk] = {"auc": best["auc"], "tag": best["tag"]}

    # Top-5 by ML, then pick by tail_3s
    for topk in [3, 5, 10]:
        pool = ml_sorted[:min(topk, len(candidates))]
        best = max(pool, key=lambda c: c["tail_3s"])
        results["ml_top%d_tail3s" % topk] = {"auc": best["auc"], "tag": best["tag"]}

    # Rank product: ML rank × trimmed_gini rank
    from scipy.stats import rankdata
    ml_vals = np.array([c["ml"] for c in candidates])
    tg_vals = np.array([c["trimmed_gini"] for c in candidates])
    ml_ranks = rankdata(-ml_vals)
    tg_ranks = rankdata(-tg_vals)
    rp = ml_ranks * tg_ranks
    best_idx = np.argmin(rp)
    results["rank_prod"] = {"auc": candidates[best_idx]["auc"],
                            "tag": candidates[best_idx]["tag"]}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    ds = args.dataset
    print("=== %s ===" % ds, flush=True)
    r = run_dataset(ds, args.device)

    if r:
        rep = ORACLE_AUROCS.get(ds, 0)
        print("\nOracle: %.1f%% (%s)  Reported: %.1f%%  Candidates: %d" %
              (r["oracle_auc"]*100, r["oracle_tag"], rep, r["n_candidates"]))

        all_names = [
            "gini", "trimmed_gini", "trimmed_gini_999", "gini_robust",
            "skewness", "kurtosis",
            "p95_p50", "p99_p50", "p99_p90",
            "tail_2s", "tail_3s", "tail_5s",
            "max_gap", "iqr_ratio", "cv", "bimodality", "neg_entropy",
            "ml", "ml_x_tgini", "ml_x_gini",
            "ml_top3_tgini", "ml_top5_tgini", "ml_top10_tgini",
            "ml_top3_tail3s", "ml_top5_tail3s", "ml_top10_tail3s",
            "rank_prod",
        ]

        print("\n%-20s %7s %7s  %s" % ("Criterion", "AUC", "vsOrac", "Config"))
        for c in all_names:
            if c in r:
                info = r[c]
                auc = info["auc"] * 100
                gap = auc - r["oracle_auc"] * 100
                tag = info["tag"]
                marker = " ***" if abs(gap) < 3 else ""
                print("%-20s %6.1f%% %+6.1f   %s%s" % (c, auc, gap, tag, marker))

        os.makedirs("results", exist_ok=True)
        with open("results/criterion_search_%s.json" % ds, "w") as f:
            json.dump(r, f, indent=2, default=str)


if __name__ == "__main__":
    main()
