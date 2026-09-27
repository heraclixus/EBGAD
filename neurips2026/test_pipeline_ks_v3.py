"""Pipeline v3: KS selects from all (gamma, score) pairs.

No ML for gamma. Run gamma={0.5, 1.0}, compute J* and R at each,
KS picks the best (gamma, score) pair.
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
REPORTED = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1, "yelpchi": 72.0,
    "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}
BASELINES = {
    "weibo": ("ANOM.", 95.0), "reddit": ("CoLA", 60.3), "amazon": ("DIF", 78.1),
    "yelpchi": ("LOF", 57.2), "blogcatalog": ("TAM", 82.5), "facebook": ("TAM", 91.4),
    "acm": ("TAM", 88.8), "elliptic": ("CoLA", 65.3),
    "elliptic_plus_plus": ("CoLA", 62.9), "t_finance": ("ECOD", 83.5),
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

    candidates = []
    for gamma in [0.5, 1.0]:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            ks_je = ks_null_deviation(je[mask])
            ks_jr = ks_null_deviation(jr[mask])
            candidates.append({"tag": "J* g=%s" % gamma, "scores": je, "auc": auc_je,
                               "ks": ks_je, "gamma": gamma, "score": "J*", "rho": rho})
            candidates.append({"tag": "R g=%s" % gamma, "scores": jr, "auc": auc_jr,
                               "ks": ks_jr, "gamma": gamma, "score": "R", "rho": rho})
            print("    g=%s: J*=%.1f%% (KS=%.3f) R=%.1f%% (KS=%.3f) rho=%.3f" %
                  (gamma, auc_je*100, ks_je, auc_jr*100, ks_jr, rho), flush=True)
        except Exception as e:
            print("    g=%s: ERROR %s" % (gamma, str(e)[:60]))

    if not candidates:
        return None

    # KS picks the best candidate
    best = max(candidates, key=lambda c: c["ks"])
    oracle = max(candidates, key=lambda c: c["auc"])

    return {
        "auc_selected": best["auc"],
        "auc_oracle": oracle["auc"],
        "tag": best["tag"], "ks": best["ks"],
        "gamma": best["gamma"], "score": best["score"],
        "oracle_tag": oracle["tag"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                rep = REPORTED.get(ds, 0)
                bname, bauc = BASELINES.get(ds, ("?", 0))
                gap_oracle = (r["auc_selected"] - r["auc_oracle"]) * 100
                print("  KS picks: %.1f%% (%s, KS=%.3f)" %
                      (r["auc_selected"]*100, r["tag"], r["ks"]))
                print("  Oracle:   %.1f%% (%s)" %
                      (r["auc_oracle"]*100, r["oracle_tag"]))
                print("  vs Rep: %+.1fpp  vs %s: %+.1fpp  vs Oracle: %+.1fpp" %
                      (r["auc_selected"]*100 - rep, bname,
                       r["auc_selected"]*100 - bauc, gap_oracle))
                results[ds] = r
        except Exception as e:
            print("  ERROR: %s" % str(e)[:80])

    if len(results) > 1:
        print("\n=== FINAL SUMMARY ===")
        print("%-15s %7s %7s %7s %7s %7s" %
              ("Dataset", "KS-sel", "Oracle", "Report", "vsRep", "vsBase"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                rep = REPORTED.get(ds, 0)
                _, bauc = BASELINES.get(ds, ("?", 0))
                print("%-15s %6.1f%% %6.1f%% %6.1f%% %+6.1f %+6.1f  %s" %
                      (ds, r["auc_selected"]*100, r["auc_oracle"]*100, rep,
                       r["auc_selected"]*100 - rep, r["auc_selected"]*100 - bauc,
                       r["tag"]))
        sel_m = np.mean([r["auc_selected"] for r in results.values()]) * 100
        orc_m = np.mean([r["auc_oracle"] for r in results.values()]) * 100
        beats = sum(1 for ds, r in results.items()
                   if r["auc_selected"]*100 >= BASELINES[ds][1])
        within5 = sum(1 for r in results.values()
                     if abs(r["auc_selected"] - r["auc_oracle"]) < 0.05)
        print("\n  Mean KS: %.1f%%  Mean Oracle: %.1f%%  KS-Oracle gap: %+.1fpp" %
              (sel_m, orc_m, sel_m - orc_m))
        print("  Beats baseline: %d/%d  Within 5pp of oracle: %d/%d" %
              (beats, len(results), within5, len(results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/pipeline_ks_v3_%s.json" % args.dataset
               if args.dataset else "results/pipeline_ks_v3.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
