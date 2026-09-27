"""Posterior Predictive Check for gamma selection.

For each gamma:
1. Fit model (ML-optimize rho, kappa)
2. Generate synthetic from fitted model
3. Compute scores on both real and synthetic
4. Measure discrepancy (KS between real and synthetic scores)
Pick gamma with highest discrepancy = strongest anomaly signal.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

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
    """KS against chi-squared null (for score selection within gamma)."""
    s = scores[scores > 0]
    if len(s) < 10:
        return 0.0
    mu, var = np.mean(s), np.var(s)
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k = max(0.01, 2 * mu**2 / var)
    a = var / (2 * mu)
    try:
        from scipy.stats import kstest
        stat, _ = kstest(s / a, 'chi2', args=(k,))
        return float(stat)
    except:
        return 0.0


def compute_ppc_discrepancy(V, lam_L, tmpl_np, x_np, q, n_synthetic=5):
    """Generate synthetic from fitted model, compute score discrepancy.

    Synthetic: sample z_j ~ N(0, 1/q_j) in eigenspace, transform to node space.
    Then compute J* and R on synthetic. Compare with real scores via KS.
    """
    n, d = x_np.shape
    k = len(q)
    q_safe = np.maximum(q, 1e-12)

    # Real scores
    delta_real = x_np - tmpl_np
    delta_hat_real = V.T @ delta_real  # (k, d)
    w_real = np.sqrt(q_safe)[:, None] * delta_hat_real
    je_real = np.sum((V @ w_real)**2, axis=1)  # J* per node
    tot_real = np.maximum(np.sum(delta_real**2, axis=1), 1e-12)
    jr_real = je_real / tot_real  # R per node

    # Synthetic scores (average over multiple samples)
    je_synth_all = []
    jr_synth_all = []
    rng = np.random.RandomState(42)
    for _ in range(n_synthetic):
        # Sample in eigenspace: z_j ~ N(0, 1/q_j) for each of d features
        z = rng.randn(k, d) / np.sqrt(q_safe[:, None])
        # Transform to node space
        delta_synth = V @ z  # (n, d)
        # Compute J*
        delta_hat_s = V.T @ delta_synth
        w_s = np.sqrt(q_safe)[:, None] * delta_hat_s
        je_s = np.sum((V @ w_s)**2, axis=1)
        tot_s = np.maximum(np.sum(delta_synth**2, axis=1), 1e-12)
        jr_s = je_s / tot_s
        je_synth_all.append(je_s)
        jr_synth_all.append(jr_s)

    je_synth = np.mean(je_synth_all, axis=0)
    jr_synth = np.mean(jr_synth_all, axis=0)

    # KS between real and synthetic score distributions
    ks_je, _ = ks_2samp(je_real, je_synth)
    ks_jr, _ = ks_2samp(jr_real, jr_synth)

    return float(ks_je), float(ks_jr), float(max(ks_je, ks_jr))


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
        q_arr = Q_rho_eigenvalues(lam_L, r, k)
        ml = marginal_log_likelihood_stationary(S, q_arr, n, d_eff)
        if ml > best_ml:
            best_ml, rho, kappa = ml, r, k

    q = Q_rho_eigenvalues(lam_L, rho, kappa)

    # Compute actual scores via torch for consistency
    lam_Q = torch.from_numpy(q).float()
    V_t = torch.from_numpy(V).float()
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()

    # PPC discrepancy
    ppc_je, ppc_jr, ppc_best = compute_ppc_discrepancy(
        V, lam_L, tmpl.numpy(), x.numpy(), q)

    # KS null for score selection
    ks_je = ks_null(je)
    ks_jr = ks_null(jr)

    return {
        "je": je, "jr": jr, "ml": best_ml, "rho": rho, "kappa": kappa,
        "ppc_je": ppc_je, "ppc_jr": ppc_jr, "ppc_best": ppc_best,
        "ks_je": ks_je, "ks_jr": ks_jr,
    }


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
            r = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(r["je"])) or np.any(np.isnan(r["jr"])):
                continue
            # KS selects score
            use_je = r["ks_je"] > r["ks_jr"]
            sel = r["je"] if use_je else r["jr"]
            sel_name = "J*" if use_je else "R"
            auc = roc_auc_score(y[mask], sel[mask])
            auc_je = roc_auc_score(y[mask], r["je"][mask])
            auc_jr = roc_auc_score(y[mask], r["jr"][mask])

            configs.append({
                "gamma": gamma, "ml": r["ml"], "rho": r["rho"],
                "ppc_je": r["ppc_je"], "ppc_jr": r["ppc_jr"], "ppc_best": r["ppc_best"],
                "ks_je": r["ks_je"], "ks_jr": r["ks_jr"],
                "auc": auc, "auc_je": auc_je, "auc_jr": auc_jr,
                "sel_name": sel_name,
            })
            print("    g=%s: %s=%.1f%% ml=%.0f rho=%.3f ppc=%.3f (je=%.3f jr=%.3f)" %
                  (gamma, sel_name, auc*100, r["ml"], r["rho"],
                   r["ppc_best"], r["ppc_je"], r["ppc_jr"]), flush=True)
        except Exception as e:
            print("    g=%s: ERROR %s" % (gamma, str(e)[:60]))

    if not configs:
        return None

    oracle = max(configs, key=lambda c: c["auc"])

    rules = {
        "ml": max(configs, key=lambda c: c["ml"]),
        "ppc_best": max(configs, key=lambda c: c["ppc_best"]),
        "ppc_je": max(configs, key=lambda c: c["ppc_je"]),
        "ppc_jr": max(configs, key=lambda c: c["ppc_jr"]),
        "fixed_1.0": next((c for c in configs if c["gamma"] == 1.0), configs[0]),
    }

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
                print("  Oracle: %.1f%% (g=%s %s)" %
                      (r["oracle"]["auc"]*100, r["oracle"]["gamma"], r["oracle"]["sel_name"]))
                for rname, rval in r["rules"].items():
                    gap = (rval["auc"] - r["oracle"]["auc"]) * 100
                    mark = "***" if abs(gap) < 3 else ""
                    print("  %-12s %.1f%% (g=%s) gap=%+.1f %s" %
                          (rname, rval["auc"]*100, rval["gamma"], gap, mark))
                all_results[ds] = {
                    "oracle_auc": r["oracle"]["auc"],
                    "oracle_gamma": r["oracle"]["gamma"],
                }
                for rname, rval in r["rules"].items():
                    all_results[ds][rname + "_auc"] = rval["auc"]
                    all_results[ds][rname + "_gamma"] = rval["gamma"]
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(all_results) > 1:
        print("\n=== SUMMARY ===")
        rule_names = ["ml", "ppc_best", "ppc_je", "ppc_jr", "fixed_1.0"]
        print("%-15s %6s" % ("Dataset", "Oracle"), end="")
        for rn in rule_names:
            print(" %8s" % rn[:8], end="")
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
                print(" %6.1f%%%s" % (auc, mark), end="")
            print()

        print("\nWithin 3pp of oracle:")
        for rn in rule_names:
            cnt = sum(1 for r in all_results.values()
                     if abs(r.get(rn+"_auc", 0) - r["oracle_auc"]) < 0.03)
            print("  %-12s %d/%d" % (rn, cnt, len(all_results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/ppc_gamma_%s.json" % args.dataset
               if args.dataset else "results/ppc_gamma.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
