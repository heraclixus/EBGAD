"""Isolate template selection: low vs affinity at gamma=1.0.

Run both templates, KS selects score within each, then analyze what
unsupervised statistic predicts which template is better.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest, spearmanr, skew

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
REPORTED = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1, "yelpchi": 72.0,
    "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}
# Ground truth: which template is better (from sweep-best configs)
GT_TMPL = {
    "weibo": "low", "reddit": "affinity", "amazon": "low", "yelpchi": "low",
    "blogcatalog": "affinity", "facebook": "affinity", "acm": "affinity",
    "elliptic": "affinity", "elliptic_plus_plus": "affinity", "t_finance": "affinity",
}


def ks_null_deviation(scores):
    s = scores[scores > 0]
    if len(s) < 10:
        return 0.0
    mu, var = np.mean(s), np.var(s)
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k_est = max(0.01, 2 * mu**2 / var)
    a_est = var / (2 * mu)
    try:
        ks_stat, _ = kstest(s / a_est, 'chi2', args=(k_est,))
        return float(ks_stat)
    except:
        return 0.0


def run_template(data, tmpl, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type=tmpl, graph_type="original", gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl_t = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl_t.numpy()
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
        je = precision_energy_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()

    # KS selects score
    ks_je = ks_null_deviation(je)
    ks_jr = ks_null_deviation(jr)
    use_je = ks_je > ks_jr
    sel_scores = je if use_je else jr
    sel_name = "J*" if use_je else "R"

    # Compute unsupervised statistics for this template config
    delta_norms = np.linalg.norm(delta, axis=1)

    return {
        "je": je, "jr": jr, "sel_scores": sel_scores, "sel_name": sel_name,
        "ks_je": ks_je, "ks_jr": ks_jr, "ks_best": max(ks_je, ks_jr),
        "ml": best_ml, "rho": rho, "kappa": kappa,
        "delta_mean": float(np.mean(delta_norms)),
        "delta_cv": float(np.std(delta_norms) / (np.mean(delta_norms) + 1e-12)),
        "delta_skew": float(skew(delta_norms)),
    }


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    results = {}
    for tmpl in ["low", "affinity"]:
        try:
            r = run_template(data, tmpl, 1.0, pca_dim, device)
            if np.any(np.isnan(r["sel_scores"])):
                continue
            auc_sel = roc_auc_score(y[mask], r["sel_scores"][mask])
            auc_je = roc_auc_score(y[mask], r["je"][mask])
            auc_jr = roc_auc_score(y[mask], r["jr"][mask])
            r["auc_sel"] = auc_sel
            r["auc_je"] = auc_je
            r["auc_jr"] = auc_jr
            results[tmpl] = r
            print("    %s: %s=%.1f%% (J*=%.1f%% R=%.1f%%) KS_best=%.3f ML=%.0f rho=%.3f" %
                  (tmpl, r["sel_name"], auc_sel*100, auc_je*100, auc_jr*100,
                   r["ks_best"], r["ml"], r["rho"]), flush=True)
        except Exception as e:
            print("    %s: ERROR %s" % (tmpl, str(e)[:60]))

    if len(results) < 2:
        return None

    gt = GT_TMPL.get(ds, "?")
    low_auc = results["low"]["auc_sel"]
    aff_auc = results["affinity"]["auc_sel"]
    better = "low" if low_auc >= aff_auc else "affinity"

    return {
        "low": results["low"], "affinity": results["affinity"],
        "better": better, "gt": gt,
        "low_auc": low_auc, "aff_auc": aff_auc,
        "margin": abs(low_auc - aff_auc),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    all_results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                all_results[ds] = r
                print("  Better: %s (low=%.1f%% aff=%.1f%% margin=%.1fpp gt=%s)" %
                      (r["better"], r["low_auc"]*100, r["aff_auc"]*100,
                       r["margin"]*100, r["gt"]))
        except Exception as e:
            print("  ERROR: %s" % str(e)[:80])

    if len(all_results) > 1:
        # Analyze: what predicts template choice?
        print("\n=== TEMPLATE PREDICTION ===")
        print("%-15s %6s %6s %7s %7s %7s %7s %7s %7s %5s" %
              ("Dataset", "low", "aff", "KS_low", "KS_aff", "ML_low", "ML_aff",
               "rho_l", "rho_a", "gt"))
        for ds in datasets:
            if ds not in all_results:
                continue
            r = all_results[ds]
            lo, af = r["low"], r["affinity"]
            print("%-15s %5.1f%% %5.1f%% %7.3f %7.3f %7.0f %7.0f %7.3f %7.3f  %s" %
                  (ds, r["low_auc"]*100, r["aff_auc"]*100,
                   lo["ks_best"], af["ks_best"],
                   lo["ml"], af["ml"],
                   lo["rho"], af["rho"], r["gt"]))

        # Test rules
        print("\n=== RULES ===")
        rules = {
            "ks_best": lambda r: "affinity" if r["affinity"]["ks_best"] > r["low"]["ks_best"] else "low",
            "ml": lambda r: "affinity" if r["affinity"]["ml"] > r["low"]["ml"] else "low",
            "rho_higher": lambda r: "affinity" if r["affinity"]["rho"] > r["low"]["rho"] else "low",
            "always_low": lambda r: "low",
            "always_aff": lambda r: "affinity",
            "delta_cv": lambda r: "affinity" if r["affinity"]["delta_cv"] > r["low"]["delta_cv"] else "low",
        }
        for rname, rule in rules.items():
            correct = 0
            total = 0
            for ds, r in all_results.items():
                pred = rule(r)
                actual = r["better"]
                if pred == actual:
                    correct += 1
                total += 1
            print("  %-15s %d/%d" % (rname, correct, total))

        # Detail for ks_best rule
        print("\n=== KS_best rule detail ===")
        for ds, r in all_results.items():
            lo_ks = r["low"]["ks_best"]
            af_ks = r["affinity"]["ks_best"]
            pred = "affinity" if af_ks > lo_ks else "low"
            actual = r["better"]
            ok = "OK" if pred == actual else "MISS"
            print("  %-15s low_KS=%.3f aff_KS=%.3f pred=%s actual=%s %s" %
                  (ds, lo_ks, af_ks, pred, actual, ok))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/template_rule_%s.json" % args.dataset
               if args.dataset else "results/template_rule.json")
    with open(outfile, "w") as f:
        json.dump({ds: {"low_auc": r["low_auc"], "aff_auc": r["aff_auc"],
                        "better": r["better"], "gt": r["gt"],
                        "low_ks": r["low"]["ks_best"], "aff_ks": r["affinity"]["ks_best"],
                        "low_ml": r["low"]["ml"], "aff_ml": r["affinity"]["ml"],
                        "low_rho": r["low"]["rho"], "aff_rho": r["affinity"]["rho"]}
                   for ds, r in all_results.items()}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
