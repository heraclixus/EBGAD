"""Compute unsupervised graph/feature statistics to predict J* vs R.

For each dataset, run a few representative configs, compute J* and R,
then measure graph/feature properties that might predict which score wins.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis, spearmanr

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

# Ground truth from paper's appendix table
REPORTED_SCORE = {
    "enron": "R", "weibo": "J*", "reddit": "J*", "amazon": "J*",
    "yelpchi": "J*", "blogcatalog": "J*", "facebook": "R", "acm": "C",
    "elliptic": "R", "elliptic_plus_plus": "R", "t_finance": "R",
}

CONFIGS = [
    {"template_type": "low", "graph_type": "original", "gamma": 0.5},
    {"template_type": "affinity", "graph_type": "original", "gamma": 0.5},
    {"template_type": "low", "graph_type": "original", "gamma": 0.1},
    {"template_type": "low", "graph_type": "original", "gamma": 2.0},
]


def run_config(data, cfg, pca_dim, device="cpu"):
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        **cfg,
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
        n, d = x.shape
        delta = x.numpy() - tmpl.numpy()
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
            je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()
        return {
            "je": je, "jr": jr, "rho": best_rho, "kappa": best_kappa,
            "lam_L": lam_L, "V": V, "delta": delta, "S": S, "ml": best_ml,
        }
    except:
        return None


def analyze_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    x_raw = data.x.float().numpy()
    n, d = x_raw.shape

    # === Graph statistics (cheap) ===
    edge_index = data.edge_index.numpy()
    degrees = np.bincount(edge_index[0], minlength=n) + np.bincount(edge_index[1], minlength=n)
    avg_deg = float(degrees.mean())
    deg_cv = float(degrees.std() / (degrees.mean() + 1e-12))
    deg_skew = float(skew(degrees.astype(float)))
    n_edges = edge_index.shape[1] // 2
    e_per_n = n_edges / n

    # === Feature statistics (cheap) ===
    feat_norms = np.linalg.norm(x_raw, axis=1)
    feat_cv = float(feat_norms.std() / (feat_norms.mean() + 1e-12))
    feat_skew_val = float(skew(feat_norms))

    # Feature-neighbor similarity (one-class homophily proxy)
    # Sample edges for speed
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    src, dst = edge_index[0, idx], edge_index[1, idx]
    x_t = torch.from_numpy(x_raw).float()
    src_feat = x_t[src]
    dst_feat = x_t[dst]
    cos_sim = torch.nn.functional.cosine_similarity(src_feat, dst_feat, dim=1)
    edge_homophily = float(cos_sim.mean())
    edge_homophily_std = float(cos_sim.std())

    # === Run configs and get scores ===
    pca_dim = 64 if d > 100 else None
    best_je_auc, best_jr_auc = 0, 0
    best_result = None

    for cfg in CONFIGS:
        result = run_config(data, cfg, pca_dim, device)
        if result is None or np.any(np.isnan(result["je"])):
            continue
        auc_je = roc_auc_score(y[mask], result["je"][mask])
        auc_jr = roc_auc_score(y[mask], result["jr"][mask])
        if max(auc_je, auc_jr) > max(best_je_auc, best_jr_auc):
            best_je_auc = auc_je
            best_jr_auc = auc_jr
            best_result = result

    if best_result is None:
        return None

    je, jr = best_result["je"], best_result["jr"]
    lam_L = best_result["lam_L"]
    delta = best_result["delta"]
    S = best_result["S"]
    rho = best_result["rho"]

    # === Residual statistics ===
    delta_norms = np.linalg.norm(delta, axis=1)
    delta_cv = float(delta_norms.std() / (delta_norms.mean() + 1e-12))
    delta_skew_val = float(skew(delta_norms))
    delta_kurtosis = float(kurtosis(delta_norms))

    # Degree-residual correlation (unsupervised)
    corr_deg_delta = float(spearmanr(degrees, delta_norms)[0])

    # Score correlations
    corr_je_jr = float(spearmanr(je[mask], jr[mask])[0])
    delta_energy = je / (jr + 1e-12)
    corr_je_delta_e = float(spearmanr(je[mask], delta_energy[mask])[0])

    # === Spectral statistics ===
    n_modes = len(lam_L)
    # Spectral gap
    lam_sorted = np.sort(lam_L)
    spectral_gap = float(lam_sorted[1] - lam_sorted[0]) if n_modes > 1 else 0
    # Eigenvalue spread
    lam_range = float(lam_sorted[-1] - lam_sorted[0])
    # Spectral energy concentration
    S_norm = S / (S.sum() + 1e-12)
    spectral_entropy = float(-np.sum(S_norm * np.log(S_norm + 1e-12)))
    # Low-freq energy fraction
    k10 = max(1, n_modes // 10)
    low_freq_frac = float(S[:k10].sum() / (S.sum() + 1e-12))
    # High-freq energy fraction
    high_freq_frac = float(S[-k10:].sum() / (S.sum() + 1e-12))

    # === Feature-graph alignment ===
    # How much feature variance is explained by graph smoothing?
    # Ratio of smoothed feature norm to raw feature norm
    # (already have delta = x - template, so ||template|| / ||x|| measures alignment)
    tmpl_norms = np.linalg.norm(best_result["V"] @ (best_result["V"].T @ delta), axis=1)
    graph_alignment = float(np.mean(tmpl_norms / (delta_norms + 1e-12)))

    oracle = REPORTED_SCORE.get(ds, "?")
    je_wins = best_je_auc > best_jr_auc

    return {
        # Graph
        "n": n, "d": d, "e_per_n": e_per_n, "avg_deg": avg_deg,
        "deg_cv": deg_cv, "deg_skew": deg_skew,
        "edge_homophily": edge_homophily, "edge_homophily_std": edge_homophily_std,
        # Features
        "feat_cv": feat_cv, "feat_skew": feat_skew_val,
        # Residuals
        "delta_cv": delta_cv, "delta_skew": delta_skew_val, "delta_kurtosis": delta_kurtosis,
        "corr_deg_delta": corr_deg_delta,
        # Scores
        "corr_je_jr": corr_je_jr, "corr_je_delta_e": corr_je_delta_e,
        "rho": rho,
        # Spectral
        "spectral_gap": spectral_gap, "lam_range": lam_range,
        "spectral_entropy": spectral_entropy,
        "low_freq_frac": low_freq_frac, "high_freq_frac": high_freq_frac,
        "graph_alignment": graph_alignment,
        # Truth
        "auc_je": best_je_auc, "auc_jr": best_jr_auc,
        "je_wins": je_wins, "oracle": oracle,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else [
        "enron", "weibo", "reddit", "amazon", "yelpchi",
        "blogcatalog", "facebook", "acm",
        "elliptic", "elliptic_plus_plus", "t_finance",
    ]

    results = {}
    for ds in datasets:
        print("=== %s ===" % ds, flush=True)
        try:
            r = analyze_dataset(ds, args.device)
            if r:
                results[ds] = r
                print("  J*=%.1f%% R=%.1f%% winner=%s reported=%s" %
                      (r["auc_je"]*100, r["auc_jr"]*100,
                       "J*" if r["je_wins"] else "R", r["oracle"]))
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(results) > 1:
        # Separate J*-wins vs R-wins
        # Use reported oracle (paper table), not our empirical winner
        je_ds = [ds for ds, r in results.items() if r["oracle"] in ("J*",)]
        r_ds = [ds for ds, r in results.items() if r["oracle"] in ("R",)]

        stats = [
            "e_per_n", "avg_deg", "deg_cv", "deg_skew",
            "edge_homophily", "edge_homophily_std",
            "feat_cv", "feat_skew",
            "delta_cv", "delta_skew", "delta_kurtosis",
            "corr_deg_delta", "corr_je_jr", "corr_je_delta_e",
            "rho", "spectral_gap", "lam_range", "spectral_entropy",
            "low_freq_frac", "high_freq_frac", "graph_alignment",
        ]

        print("\n=== SEPARATION ANALYSIS (J* vs R, using reported labels) ===")
        print("J* datasets: %s" % je_ds)
        print("R datasets: %s" % r_ds)
        print("\n%-22s  %10s  %10s  %10s  %s" % ("Statistic", "J*-mean", "R-mean", "Gap", "Sep?"))
        for s in stats:
            je_vals = [results[ds][s] for ds in je_ds if ds in results and s in results[ds]]
            r_vals = [results[ds][s] for ds in r_ds if ds in results and s in results[ds]]
            if not je_vals or not r_vals:
                continue
            je_m, r_m = np.mean(je_vals), np.mean(r_vals)
            # Check if ranges overlap
            je_min, je_max = min(je_vals), max(je_vals)
            r_min, r_max = min(r_vals), max(r_vals)
            overlap = max(0, min(je_max, r_max) - max(je_min, r_min))
            total_range = max(je_max, r_max) - min(je_min, r_min) + 1e-12
            overlap_frac = overlap / total_range
            sep = "CLEAN" if overlap_frac < 0.01 else ("good" if overlap_frac < 0.2 else ("partial" if overlap_frac < 0.5 else "no"))
            print("%-22s  %10.4f  %10.4f  %10.4f  %s" % (s, je_m, r_m, je_m - r_m, sep))

        # Per-dataset detail
        print("\n%-15s %6s %6s %6s %6s %6s %6s %6s %6s %3s" %
              ("Dataset", "e/n", "hom", "fCV", "dCV", "dSkew", "rho", "JRcor", "Jdcor", "Win"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                print("%-15s %6.1f %6.3f %6.2f %6.2f %6.1f %6.3f %+5.2f %+5.2f  %s" %
                      (ds, r["e_per_n"], r["edge_homophily"], r["feat_cv"],
                       r["delta_cv"], r["delta_skew"], r["rho"],
                       r["corr_je_jr"], r["corr_je_delta_e"], r["oracle"]))

    os.makedirs("results", exist_ok=True)
    outfile = "results/score_pred_%s.json" % args.dataset if args.dataset else "results/score_predictors.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
