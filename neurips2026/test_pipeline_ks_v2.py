"""Pipeline v2: ML selects gamma from {0.5, 1.0}, KS selects J* vs R.

Also test gamma={0.2, 0.5, 1.0, 2.0} to see if wider range helps.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest, chi2

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
    return je, jr, best_ml, rho, kappa


def run_dataset(ds, gamma_set, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    # ML selects gamma
    best_ml = -np.inf
    best_gamma = gamma_set[0]
    best_je, best_jr = None, None
    best_rho = 0.5

    for gamma in gamma_set:
        try:
            je, jr, ml, rho, kappa = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            print("    g=%s: ml=%.0f rho=%.3f J*=%.1f%% R=%.1f%%" %
                  (gamma, ml, rho, auc_je*100, auc_jr*100), flush=True)
            if ml > best_ml:
                best_ml = ml
                best_gamma = gamma
                best_je, best_jr = je, jr
                best_rho = rho
        except Exception as e:
            print("    g=%s: ERROR %s" % (gamma, str(e)[:60]), flush=True)

    if best_je is None:
        return None

    # KS selects score
    ks_je = ks_null_deviation(best_je[mask])
    ks_jr = ks_null_deviation(best_jr[mask])
    use_je = ks_je > ks_jr
    scores = best_je if use_je else best_jr
    score_name = "J*" if use_je else "R"

    auc_sel = roc_auc_score(y[mask], scores[mask])
    auc_je = roc_auc_score(y[mask], best_je[mask])
    auc_jr = roc_auc_score(y[mask], best_jr[mask])

    return {
        "auc_selected": auc_sel, "auc_je": auc_je, "auc_jr": auc_jr,
        "auc_oracle": max(auc_je, auc_jr),
        "score_name": score_name, "gamma": best_gamma,
        "ks_je": ks_je, "ks_jr": ks_jr, "rho": best_rho,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())

    # Test multiple gamma sets
    gamma_sets = {
        "g05_10": [0.5, 1.0],
        "g02_05_10_20": [0.2, 0.5, 1.0, 2.0],
    }

    for gname, gset in gamma_sets.items():
        print("\n======= Gamma set: %s =======" % gname, flush=True)
        results = {}
        for ds in datasets:
            print("\n=== %s ===" % ds, flush=True)
            try:
                r = run_dataset(ds, gset, args.device)
                if r:
                    rep = REPORTED.get(ds, 0)
                    bname, bauc = BASELINES.get(ds, ("?", 0))
                    print("  Selected: %.1f%% (%s g=%s) KS_J*=%.3f KS_R=%.3f" %
                          (r["auc_selected"]*100, r["score_name"], r["gamma"],
                           r["ks_je"], r["ks_jr"]))
                    print("  vs Rep: %+.1fpp  vs %s: %+.1fpp" %
                          (r["auc_selected"]*100 - rep, bname, r["auc_selected"]*100 - bauc))
                    results[ds] = r
            except Exception as e:
                print("  ERROR: %s" % str(e)[:80])

        if len(results) > 1:
            print("\n=== SUMMARY %s ===" % gname)
            print("%-15s %7s %6s %6s %7s %7s %5s %5s" %
                  ("Dataset", "Sel", "J*", "R", "Report", "vsBase", "Score", "g"))
            for ds in datasets:
                if ds in results:
                    r = results[ds]
                    rep = REPORTED.get(ds, 0)
                    _, bauc = BASELINES.get(ds, ("?", 0))
                    print("%-15s %6.1f%% %5.1f%% %5.1f%% %6.1f%% %+6.1f  %s  %s" %
                          (ds, r["auc_selected"]*100, r["auc_je"]*100, r["auc_jr"]*100,
                           rep, r["auc_selected"]*100 - bauc,
                           r["score_name"], r["gamma"]))
            beats = sum(1 for ds, r in results.items()
                       if r["auc_selected"]*100 >= BASELINES[ds][1])
            sel_m = np.mean([r["auc_selected"] for r in results.values()]) * 100
            print("  Mean: %.1f%%  Beats baseline: %d/%d" % (sel_m, beats, len(results)))

        os.makedirs("results", exist_ok=True)
        with open("results/pipeline_ks_v2_%s_%s.json" % (gname, args.dataset or "all"), "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
