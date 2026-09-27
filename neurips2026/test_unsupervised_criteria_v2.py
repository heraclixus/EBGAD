"""Test multiple unsupervised selection criteria across the full config grid.

For each dataset, run all configs (PCA × graph × template × gamma),
compute J* and R, then evaluate which selection criterion best identifies
the oracle config.

Criteria tested:
1. Gini coefficient (baseline)
2. Trimmed Gini (Gini after removing top 1% of scores)
3. Gini robustness ratio (trimmed / full)
4. Maximum gap in sorted scores
5. Tail mass fraction (fraction above mean + 3*std)
6. p95/p50 ratio
7. p99/p50 ratio
8. Excess kurtosis
9. ML (marginal likelihood, for comparison)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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

ORACLE_AUROCS = {
    "enron": 81.3, "weibo": 95.8, "reddit": 62.6, "amazon": 78.1,
    "yelpchi": 72.0, "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}


def gini(scores):
    s = np.sort(scores)
    n = len(s)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) / (n * np.sum(s) + 1e-12)) - (n + 1) / n)


def trimmed_gini(scores, trim_frac=0.01):
    """Gini after removing top trim_frac of scores."""
    s = np.sort(scores)
    k = max(1, int(len(s) * (1 - trim_frac)))
    return gini(s[:k])


def max_gap(scores):
    """Largest gap in sorted scores, normalized by range."""
    s = np.sort(scores)
    gaps = np.diff(s)
    rng = s[-1] - s[0]
    if rng < 1e-12:
        return 0.0
    return float(np.max(gaps) / rng)


def tail_mass(scores, k=3):
    """Fraction of scores above mean + k*std."""
    threshold = np.mean(scores) + k * np.std(scores)
    return float(np.mean(scores > threshold))


def percentile_ratio(scores, hi=95, lo=50):
    p_hi = np.percentile(scores, hi)
    p_lo = np.percentile(scores, lo)
    return float(p_hi / (p_lo + 1e-12))


def run_config(data, template_type, gamma, pca_dim, graph_type="original", device="cpu"):
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type=template_type, graph_type=graph_type, gamma=gamma,
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
        tmpl = trainer.template.cpu()
        x = trainer.x_T.cpu()
        x_np = x.numpy()
        n, d = x_np.shape
        delta = x_np - tmpl.numpy()
        S = compute_spectral_energy(delta, V)

        best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa

        if best_rho < 0.01:
            return None

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t = torch.from_numpy(V).float()

        with torch.no_grad():
            je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()

        return {"je": je, "jr": jr, "ml": best_ml, "rho": best_rho, "kappa": best_kappa}
    except Exception as e:
        return None


def compute_criteria(scores):
    """Compute all unsupervised criteria for a score array."""
    g = gini(scores)
    tg = trimmed_gini(scores, 0.01)
    return {
        "gini": g,
        "trimmed_gini": tg,
        "gini_robust": tg / (g + 1e-12),  # close to 1 = robust, <<1 = fragile
        "max_gap": max_gap(scores),
        "tail_mass": tail_mass(scores, 3),
        "p95_p50": percentile_ratio(scores, 95, 50),
        "p99_p50": percentile_ratio(scores, 99, 50),
    }


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]

    pca_candidates = [None]
    if d > 32:
        pca_candidates.append(16)
    if d > 100:
        pca_candidates.append(64)
    if d > 200:
        pca_candidates.append(128)

    candidates = []
    for pca_dim in pca_candidates:
        for graph in ["original", "affinity"]:
            for tmpl in ["low", "affinity", "high"]:
                for gamma in [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]:
                    result = run_config(data, tmpl, gamma, pca_dim, graph, device)
                    if result is not None:
                        je, jr = result["je"], result["jr"]
                        if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                            continue
                        auc_je = roc_auc_score(y[mask], je[mask])
                        auc_jr = roc_auc_score(y[mask], jr[mask])

                        pca_tag = "p%d" % pca_dim if pca_dim else "noP"
                        tag = "%s+%s g=%s %s" % (graph[:3], tmpl, gamma, pca_tag)

                        for sname, scores, auc in [("J*", je, auc_je), ("R", jr, auc_jr)]:
                            crit = compute_criteria(scores[mask])
                            crit["ml"] = result["ml"]
                            crit["auc"] = auc
                            crit["tag"] = "%s %s" % (sname, tag)
                            crit["score_name"] = sname
                            crit["tmpl"] = tmpl
                            crit["graph"] = graph
                            crit["gamma"] = gamma
                            crit["pca"] = pca_dim
                            crit["rho"] = result["rho"]
                            candidates.append(crit)

                        print("    %s: J*=%.1f%% R=%.1f%%" % (tag, auc_je*100, auc_jr*100),
                              flush=True)

    if not candidates:
        return None

    oracle_auc = max(c["auc"] for c in candidates)
    n = len(candidates)

    # Compute ranks for combined criteria (rank 0 = best)
    from scipy.stats import rankdata
    ml_vals = np.array([c["ml"] for c in candidates])
    tg_vals = np.array([c["trimmed_gini"] for c in candidates])
    gini_vals = np.array([c["gini"] for c in candidates])
    tm_vals = np.array([c["tail_mass"] for c in candidates])

    # Ranks (higher value = lower rank number = better)
    ml_ranks = rankdata(-ml_vals)
    tg_ranks = rankdata(-tg_vals)
    gini_ranks = rankdata(-gini_vals)
    tm_ranks = rankdata(-tm_vals)

    # Normalize each criterion to [0, 1] for weighted sums
    def normalize(vals):
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-12:
            return np.zeros_like(vals)
        return (vals - lo) / (hi - lo)

    ml_norm = normalize(ml_vals)
    tg_norm = normalize(tg_vals)
    gini_norm = normalize(gini_vals)
    tm_norm = normalize(tm_vals)

    # === Single criteria ===
    criteria_names = ["gini", "trimmed_gini", "gini_robust", "max_gap",
                      "tail_mass", "p95_p50", "p99_p50", "ml"]

    results = {"oracle": oracle_auc, "n_candidates": n}

    for cname in criteria_names:
        if cname == "max_gap":
            best = min(candidates, key=lambda c: c[cname])
        elif cname == "gini_robust":
            best = max(candidates, key=lambda c: c[cname])
        else:
            best = max(candidates, key=lambda c: c[cname])
        results[cname + "_auc"] = best["auc"]
        results[cname + "_tag"] = best["tag"]

    # === Combined criteria ===

    # C1: Top-10 by ML, then pick highest trimmed Gini
    top_k = 10
    ml_order = np.argsort(-ml_vals)
    top_ml_idx = ml_order[:min(top_k, n)]
    best_idx = top_ml_idx[np.argmax(tg_vals[top_ml_idx])]
    results["ml_top10_tgini_auc"] = candidates[best_idx]["auc"]
    results["ml_top10_tgini_tag"] = candidates[best_idx]["tag"]

    # C2: Top-10 by trimmed Gini, then pick highest ML
    tg_order = np.argsort(-tg_vals)
    top_tg_idx = tg_order[:min(top_k, n)]
    best_idx = top_tg_idx[np.argmax(ml_vals[top_tg_idx])]
    results["tgini_top10_ml_auc"] = candidates[best_idx]["auc"]
    results["tgini_top10_ml_tag"] = candidates[best_idx]["tag"]

    # C3: Rank product (trimmed Gini rank × ML rank), pick lowest
    rank_prod = tg_ranks * ml_ranks
    best_idx = np.argmin(rank_prod)
    results["rank_prod_tg_ml_auc"] = candidates[best_idx]["auc"]
    results["rank_prod_tg_ml_tag"] = candidates[best_idx]["tag"]

    # C4: Weighted sum: 0.5*ML_norm + 0.5*trimmedGini_norm
    combo = 0.5 * ml_norm + 0.5 * tg_norm
    best_idx = np.argmax(combo)
    results["avg_ml_tgini_auc"] = candidates[best_idx]["auc"]
    results["avg_ml_tgini_tag"] = candidates[best_idx]["tag"]

    # C5: Weighted sum: 0.3*ML + 0.7*trimmedGini
    combo = 0.3 * ml_norm + 0.7 * tg_norm
    best_idx = np.argmax(combo)
    results["w37_ml_tgini_auc"] = candidates[best_idx]["auc"]
    results["w37_ml_tgini_tag"] = candidates[best_idx]["tag"]

    # C6: Weighted sum: 0.7*ML + 0.3*trimmedGini
    combo = 0.7 * ml_norm + 0.3 * tg_norm
    best_idx = np.argmax(combo)
    results["w73_ml_tgini_auc"] = candidates[best_idx]["auc"]
    results["w73_ml_tgini_tag"] = candidates[best_idx]["tag"]

    # C7: Top-10 by ML, then pick highest Gini (original, not trimmed)
    best_idx = top_ml_idx[np.argmax(gini_vals[top_ml_idx])]
    results["ml_top10_gini_auc"] = candidates[best_idx]["auc"]
    results["ml_top10_gini_tag"] = candidates[best_idx]["tag"]

    # C8: Rank product (Gini rank × ML rank)
    rank_prod2 = gini_ranks * ml_ranks
    best_idx = np.argmin(rank_prod2)
    results["rank_prod_g_ml_auc"] = candidates[best_idx]["auc"]
    results["rank_prod_g_ml_tag"] = candidates[best_idx]["tag"]

    # C9: Top-20% by ML, then pick highest trimmed Gini
    top_pct = max(1, n // 5)
    top_ml_pct_idx = ml_order[:top_pct]
    best_idx = top_ml_pct_idx[np.argmax(tg_vals[top_ml_pct_idx])]
    results["ml_top20p_tgini_auc"] = candidates[best_idx]["auc"]
    results["ml_top20p_tgini_tag"] = candidates[best_idx]["tag"]

    # C10: Top-20% by ML, then pick highest tail_mass
    best_idx = top_ml_pct_idx[np.argmax(tm_vals[top_ml_pct_idx])]
    results["ml_top20p_tmass_auc"] = candidates[best_idx]["auc"]
    results["ml_top20p_tmass_tag"] = candidates[best_idx]["tag"]

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
        print("\n=== RESULTS ===")
        print("Oracle: %.1f%%  Reported: %.1f%%  Candidates: %d" %
              (r["oracle"]*100, rep, r["n_candidates"]))

        all_criteria = [
            "gini", "trimmed_gini", "gini_robust", "max_gap",
            "tail_mass", "p95_p50", "p99_p50", "ml",
            "ml_top10_tgini", "tgini_top10_ml", "rank_prod_tg_ml",
            "avg_ml_tgini", "w37_ml_tgini", "w73_ml_tgini",
            "ml_top10_gini", "rank_prod_g_ml",
            "ml_top20p_tgini", "ml_top20p_tmass",
        ]
        print("\n%-20s %7s %7s  %s" % ("Criterion", "AUC", "vsOracle", "Config"))
        for c in all_criteria:
            auc = r[c + "_auc"] * 100
            gap = auc - r["oracle"] * 100
            tag = r[c + "_tag"]
            marker = " ***" if abs(gap) < 3 else ""
            print("%-20s %6.1f%% %+6.1f   %s%s" % (c, auc, gap, tag, marker))

        os.makedirs("results", exist_ok=True)
        with open("results/criteria_v2_%s.json" % ds, "w") as f:
            json.dump(r, f, indent=2, default=str)


if __name__ == "__main__":
    main()
