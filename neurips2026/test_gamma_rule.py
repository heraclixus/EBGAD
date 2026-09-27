"""Test gamma selection rules.

H1: KS of R across gammas (R is magnitude-normalized, more stable)
H2: Penalize gammas where rho < 0.1 (graph not used)
H3: KS of R, but only among gammas where rho > 0.1
H4: ML * rho (penalize low graph trust)
H5: KS_R * rho
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest

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
REPORTED = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1, "yelpchi": 72.0,
    "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}
BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 78.1, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 83.5,
}


def ks_null(scores):
    s = scores[scores > 0]
    if len(s) < 10:
        return 0.0
    mu, var = np.mean(s), np.var(s)
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k = max(0.01, 2 * mu**2 / var)
    a = var / (2 * mu)
    try:
        stat, _ = kstest(s / a, 'chi2', args=(k,))
        return float(stat)
    except:
        return 0.0


def run_gamma(data, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="low", graph_type="original", gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl.numpy()
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
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()
    return je, jr, best_ml, rho


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    configs = []
    for gamma in GAMMAS:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            # KS selects score
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            sel_name = "J*" if use_je else "R"
            auc = roc_auc_score(y[mask], sel[mask])
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])

            configs.append({
                "gamma": gamma, "ml": ml, "rho": rho,
                "ks_je": ks_je, "ks_jr": ks_jr, "ks_best": max(ks_je, ks_jr),
                "auc": auc, "auc_je": auc_je, "auc_jr": auc_jr,
                "sel_name": sel_name,
            })
            print("    g=%s: %s=%.1f%% ml=%.0f rho=%.3f ks_J=%.3f ks_R=%.3f" %
                  (gamma, sel_name, auc*100, ml, rho, ks_je, ks_jr), flush=True)
        except Exception as e:
            print("    g=%s: ERROR %s" % (gamma, str(e)[:60]))

    if not configs:
        return None

    oracle = max(configs, key=lambda c: c["auc"])

    # Test gamma selection rules
    rules = {}

    # ML (baseline)
    rules["ml"] = max(configs, key=lambda c: c["ml"])

    # KS of best score
    rules["ks_best"] = max(configs, key=lambda c: c["ks_best"])

    # KS of R only
    rules["ks_R"] = max(configs, key=lambda c: c["ks_jr"])

    # KS of J* only
    rules["ks_J"] = max(configs, key=lambda c: c["ks_je"])

    # ML * rho
    rules["ml_x_rho"] = max(configs, key=lambda c: c["ml"] * c["rho"] if c["ml"] > 0 else c["ml"] / (c["rho"] + 0.01))

    # KS_R * rho
    rules["ks_R_x_rho"] = max(configs, key=lambda c: c["ks_jr"] * c["rho"])

    # KS_best * rho
    rules["ks_best_x_rho"] = max(configs, key=lambda c: c["ks_best"] * c["rho"])

    # ML, but only among gammas with rho > 0.1
    high_rho = [c for c in configs if c["rho"] > 0.1]
    if high_rho:
        rules["ml_rho>0.1"] = max(high_rho, key=lambda c: c["ml"])
    else:
        rules["ml_rho>0.1"] = rules["ml"]

    # Fixed gamma=1.0
    g10 = [c for c in configs if c["gamma"] == 1.0]
    if g10:
        rules["fixed_1.0"] = g10[0]

    return {"oracle": oracle, "rules": rules, "configs": configs}


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
                rep = REPORTED.get(ds, 0)
                print("  Oracle: %.1f%% (g=%s %s)" %
                      (r["oracle"]["auc"]*100, r["oracle"]["gamma"], r["oracle"]["sel_name"]))
                for rname, rval in r["rules"].items():
                    gap = (rval["auc"] - r["oracle"]["auc"]) * 100
                    mark = "***" if abs(gap) < 3 else ""
                    print("  %-15s %.1f%% (g=%s) gap=%+.1f %s" %
                          (rname, rval["auc"]*100, rval["gamma"], gap, mark))
                all_results[ds] = {
                    "oracle_auc": r["oracle"]["auc"],
                    "oracle_gamma": r["oracle"]["gamma"],
                }
                for rname, rval in r["rules"].items():
                    all_results[ds][rname + "_auc"] = rval["auc"]
                    all_results[ds][rname + "_gamma"] = rval["gamma"]
        except Exception as e:
            print("  ERROR: %s" % str(e)[:80])

    if len(all_results) > 1:
        print("\n=== SUMMARY ===")
        rule_names = ["ml", "ks_best", "ks_R", "ks_J", "ml_x_rho", "ks_R_x_rho",
                      "ks_best_x_rho", "ml_rho>0.1", "fixed_1.0"]
        print("%-15s %6s" % ("Dataset", "Oracle"), end="")
        for rn in rule_names:
            print(" %7s" % rn[:7], end="")
        print()
        for ds in datasets:
            if ds not in all_results:
                continue
            r = all_results[ds]
            print("%-15s %5.1f%%" % (ds, r["oracle_auc"]*100), end="")
            for rn in rule_names:
                auc = r.get(rn + "_auc", 0) * 100
                gap = auc - r["oracle_auc"]*100
                mark = "*" if abs(gap) < 3 else " "
                print(" %5.1f%%%s" % (auc, mark), end="")
            print()

        print("\nWithin 3pp of oracle:")
        for rn in rule_names:
            cnt = sum(1 for r in all_results.values()
                     if abs(r.get(rn+"_auc", 0) - r["oracle_auc"]) < 0.03)
            print("  %-15s %d/%d" % (rn, cnt, len(all_results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/gamma_rule_%s.json" % args.dataset
               if args.dataset else "results/gamma_rule.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
