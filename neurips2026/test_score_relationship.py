"""Analyze the relationship between J*, R, and ||delta||^2.

Key identity: R_i = J*_i / ||delta_i||^2

So J* and R disagree when ||delta||^2 is NOT monotone with J*.
This script measures:
1. Spearman(J*, ||delta||^2) -- if high, J* ≈ rank(R) and they agree
2. Spearman(J*, R) -- direct rank agreement
3. Which score has higher AUROC and why
4. Whether we can predict the winner unsupervised
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

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

# Best configs from sweeps (same as test_ablation_gaps.py)
BEST = {
    "enron":    {"template_type": "high", "graph_type": "affinity", "gamma": 0.1, "pca": None},
    "weibo":    {"template_type": "low", "graph_type": "original", "gamma": 0.2, "pca": 64},
    "reddit":   {"template_type": "affinity", "graph_type": "original", "gamma": 2.0, "pca": None},
    "amazon":   {"template_type": "low", "graph_type": "original", "gamma": 5.0, "pca": 64},
    "yelpchi":  {"template_type": "low", "graph_type": "original", "gamma": 1.0, "pca": 64},
    "blogcatalog": {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": 64},
    "facebook": {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": 16},
    "acm":      {"template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": 128},
    "elliptic": {"template_type": "affinity", "graph_type": "original", "gamma": 0.7, "pca": None},
    "elliptic_plus_plus": {"template_type": "affinity", "graph_type": "original", "gamma": 0.7, "pca": None},
    "t_finance": {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": None},
}


def analyze_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    cfg = BEST[ds]

    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type=cfg["template_type"], graph_type=cfg["graph_type"],
        gamma=cfg["gamma"],
    )
    if cfg.get("pca") and data.x.shape[1] > cfg["pca"]:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=cfg["pca"], encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))

    trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl = trainer.template.cpu()
    # Use trainer's processed features (already normalized + PCA projected)
    x = trainer.x_T.cpu()
    x_np = x.numpy()
    n, d = x_np.shape
    delta = x_np - tmpl.numpy()
    S = compute_spectral_energy(delta, V)

    # ML-optimize rho
    best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
    for kappa in KAPPAS:
        rho = solve_rho_newton(S, lam_L, kappa, d)
        q = Q_rho_eigenvalues(lam_L, rho, kappa)
        ml = marginal_log_likelihood_stationary(S, q, n, d)
        if ml > best_ml:
            best_ml, best_rho, best_kappa = ml, rho, kappa

    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
    V_t = torch.from_numpy(V).float()
    tmpl_t = tmpl.float()
    x_t = x.float()

    with torch.no_grad():
        je = precision_energy_anomaly(x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()

    # Raw residual energy: since R = J* / ||δ||², we get ||δ||² = J* / R
    delta_energy = je / (jr + 1e-12)

    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)

    # AUROCs (for evaluation, not used in selection)
    auc_je = roc_auc_score(y[mask], je[mask])
    auc_jr = roc_auc_score(y[mask], jr[mask])
    auc_delta = roc_auc_score(y[mask], delta_energy[mask])

    # === UNSUPERVISED statistics on score distributions ===
    from scipy.stats import skew, kurtosis

    # Per-score distributional statistics (all unsupervised)
    stats = {}
    for name, scores in [("je", je), ("jr", jr)]:
        s = scores[mask] if mask.sum() > 0 else scores
        stats[f"{name}_skew"] = float(skew(s))
        stats[f"{name}_kurtosis"] = float(kurtosis(s))
        stats[f"{name}_cv"] = float(s.std() / (s.mean() + 1e-12))
        stats[f"{name}_maxmed"] = float(np.max(s) / (np.median(s) + 1e-12))
        stats[f"{name}_excess"] = float(np.mean(s > np.mean(s) + 3*np.std(s)))
        # Gini coefficient (measures inequality/concentration)
        sorted_s = np.sort(s)
        n_s = len(sorted_s)
        index = np.arange(1, n_s + 1)
        stats[f"{name}_gini"] = float((2 * np.sum(index * sorted_s) / (n_s * np.sum(sorted_s) + 1e-12)) - (n_s + 1) / n_s)
        # 95th/50th percentile ratio
        stats[f"{name}_p95_p50"] = float(np.percentile(s, 95) / (np.percentile(s, 50) + 1e-12))

    # Rank correlation between J* and R (unsupervised)
    corr_je_jr = spearmanr(je[mask], jr[mask])[0]

    # Supervised (for evaluation only)
    anom_mask = (y == 1) & mask
    norm_mask = (y == 0) & mask
    magnitude_separation = delta_energy[anom_mask].mean() / (delta_energy[norm_mask].mean() + 1e-12)

    winner = "J*" if auc_je >= auc_jr else "R"

    # === UNSUPERVISED SELECTION RULES ===
    rules = {}

    # Rule 1: Pick whichever has higher skewness (heavier anomaly tail)
    rules["skewness"] = "J*" if stats["je_skew"] > stats["jr_skew"] else "R"

    # Rule 2: Pick whichever has higher kurtosis
    rules["kurtosis"] = "J*" if stats["je_kurtosis"] > stats["jr_kurtosis"] else "R"

    # Rule 3: Pick whichever has higher max/median ratio
    rules["maxmed"] = "J*" if stats["je_maxmed"] > stats["jr_maxmed"] else "R"

    # Rule 4: Pick whichever has higher excess mass (fraction in extreme tail)
    rules["excess"] = "J*" if stats["je_excess"] > stats["jr_excess"] else "R"

    # Rule 5: Pick whichever has higher Gini (more concentrated/outlier-heavy)
    rules["gini"] = "J*" if stats["je_gini"] > stats["jr_gini"] else "R"

    # Rule 6: Pick whichever has higher CV
    rules["cv"] = "J*" if stats["je_cv"] > stats["jr_cv"] else "R"

    # Rule 7: Pick whichever has higher 95th/50th percentile ratio
    rules["p95_p50"] = "J*" if stats["je_p95_p50"] > stats["jr_p95_p50"] else "R"

    # Rule 8: Majority vote across rules 1-7
    votes = list(rules.values())
    rules["majority"] = "J*" if votes.count("J*") > votes.count("R") else "R"

    # Rule 9: Always J* (baseline)
    rules["always_je"] = "J*"

    # Rule 10: Always R (baseline)
    rules["always_jr"] = "R"

    return {
        "auc_je": auc_je, "auc_jr": auc_jr, "auc_delta": auc_delta,
        "winner": winner, "gap": abs(auc_je - auc_jr),
        "corr_je_jr": corr_je_jr,
        "magnitude_separation": magnitude_separation,
        "rho": best_rho, "kappa": best_kappa,
        **stats, "rules": rules,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to run (for parallel SLURM jobs)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(BEST.keys())
    results = {}

    print(f"{'Dataset':15s}  {'J*':>6s} {'R':>6s} {'Win':>3s} {'Gap':>5s}  "
          f"{'skew_J':>7s} {'skew_R':>7s} {'mm_J':>7s} {'mm_R':>7s} "
          f"{'gini_J':>7s} {'gini_R':>7s} {'MagSep':>7s}",
          flush=True)

    for ds in datasets:
        print(f"{ds:15s}  ", end="", flush=True)
        try:
            r = analyze_dataset(ds, args.device)
            results[ds] = r
            print(f"{r['auc_je']*100:5.1f}% {r['auc_jr']*100:5.1f}% "
                  f"{r['winner']:>3s} {r['gap']*100:4.1f}  "
                  f"{r['je_skew']:6.1f} {r['jr_skew']:6.1f} "
                  f"{r['je_maxmed']:6.1f} {r['jr_maxmed']:6.1f} "
                  f"{r['je_gini']:6.3f} {r['jr_gini']:6.3f} "
                  f"{r['magnitude_separation']:6.2f}")
        except Exception as e:
            import traceback
            print(f"ERROR: {str(e)[:80]}")
            traceback.print_exc()

    os.makedirs("results", exist_ok=True)
    outfile = f"results/score_rel_{args.dataset}.json" if args.dataset else "results/score_relationship.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary: evaluate each unsupervised rule
    if len(results) > 1:
        print("\n=== UNSUPERVISED RULE ACCURACY ===")
        rule_names = ["skewness", "kurtosis", "maxmed", "excess", "gini",
                       "cv", "p95_p50", "majority", "always_je", "always_jr"]
        for rule in rule_names:
            correct = sum(1 for r in results.values()
                         if r["rules"][rule] == r["winner"])
            total = len(results)
            # Also compute: what AUROC do we get with this rule?
            selected_aucs = []
            oracle_aucs = []
            for r in results.values():
                pick = r["rules"][rule]
                selected_aucs.append(r["auc_je"] if pick == "J*" else r["auc_jr"])
                oracle_aucs.append(max(r["auc_je"], r["auc_jr"]))
            mean_sel = np.mean(selected_aucs) * 100
            mean_oracle = np.mean(oracle_aucs) * 100
            gap = mean_sel - mean_oracle
            print(f"  {rule:12s}: {correct}/{total} correct  "
                  f"mean_AUROC={mean_sel:.1f}%  oracle={mean_oracle:.1f}%  gap={gap:+.1f}pp")

        # Per-dataset rule selections
        print(f"\n{'Dataset':15s}  {'Oracle':>3s}", end="")
        for rule in rule_names[:8]:
            print(f"  {rule[:6]:>6s}", end="")
        print()
        for ds, r in results.items():
            print(f"{ds:15s}  {r['winner']:>3s}", end="")
            for rule in rule_names[:8]:
                pick = r["rules"][rule]
                ok = "Y" if pick == r["winner"] else "N"
                print(f"  {pick:>2s} {ok}", end="")
            print()


if __name__ == "__main__":
    main()
