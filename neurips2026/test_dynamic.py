"""Comprehensive test of the finite-Gamma dynamic framework.

Sweeps ALL parameters in the SOC formulation:
  gamma  - template smoothing bandwidth
  Gamma  - time horizon (NEW dynamical parameter)
  kappa  - spectral cutoff
  rho    - graph trust (optimized per config)
  lambda - endpoint matching strength (eta = 1/lambda isotropic noise floor)

The full SOC precision per mode is:
  p_j = 1 / (eta + (1 - exp(-2*Gamma*q_j)) / q_j)

Special cases:
  eta=0, Gamma=inf  -> p_j = q_j              (stationary, current pipeline)
  eta=0, Gamma<inf  -> p_j = q_j/(1-exp(...)) (pure GOU generative)
  eta>0, Gamma=inf  -> p_j = 1/(eta + 1/q_j)  (stationary + noise floor)
  eta>0, Gamma<inf  -> p_j = 1/(eta + sigma_j) (full SOC)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood, marginal_log_likelihood_stationary,
    marginal_log_likelihood_dynamic, time_dependent_precision,
    predictive_variance, compute_spectral_energy,
    solve_rho_newton, solve_rho_dynamic,
)

# Grid definitions
KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
# Time horizon: from short to stationary
GAMMA_T = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
# Endpoint softness eta = 1/lambda: 0 = hard matching, >0 adds isotropic floor
ETAS = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]

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


def score_nodes_soc(delta, V, q, Gamma, eta):
    """Per-node scores using full SOC precision.

    p_j = 1 / (eta + (1 - exp(-2*Gamma*q_j)) / q_j)
    """
    q_safe = np.maximum(q, 1e-8)
    if Gamma > 100:
        # Stationary limit
        sigma_j = 1.0 / q_safe
    else:
        sigma_j = (1.0 - np.exp(-2.0 * Gamma * q_safe)) / q_safe
    D_j = eta + sigma_j  # covariance per mode
    p_j = 1.0 / np.maximum(D_j, 1e-12)  # precision per mode

    delta_hat = V.T @ delta  # (k, d)
    w = np.sqrt(p_j)[:, None] * delta_hat  # (k, d)
    je = np.sum((V @ w)**2, axis=1)  # (n,)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot
    return je, jr


def ml_soc(S, q, Gamma, eta, n, d):
    """Marginal log-likelihood under full SOC model.

    D_j = eta + (1 - exp(-2*Gamma*q_j)) / q_j
    log p = -1/2 sum S_j/D_j - d/2 sum log(D_j) - nd/2 log(2pi)
    """
    q_safe = np.maximum(q, 1e-8)
    if Gamma > 100:
        sigma_j = 1.0 / q_safe
    else:
        sigma_j = (1.0 - np.exp(-2.0 * Gamma * q_safe)) / q_safe
    D_j = eta + sigma_j
    D_j = np.maximum(D_j, 1e-12)
    ll = -0.5 * np.sum(S / D_j) - (d / 2.0) * np.sum(np.log(D_j)) - (n * d / 2.0) * np.log(2 * np.pi)
    return float(ll)


def solve_rho_soc(S, lam_L, kappa, Gamma, eta, d, nu=1.0):
    """Optimize rho under full SOC ML via bounded search."""
    from scipy.optimize import minimize_scalar
    a = (kappa ** 2 + lam_L) ** nu - 1.0

    def neg_ml(rho):
        q = rho * a + 1.0
        return -ml_soc(S, q, Gamma, eta, len(S), d)

    result = minimize_scalar(neg_ml, bounds=(0.0, 1.0), method='bounded')
    return float(np.clip(result.x, 0.0, 1.0))


def prepare_dataset(ds, device="cpu"):
    """Load data, compute eigenpairs/template/residuals for each gamma."""
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    results_per_gamma = {}
    for gamma in GAMMAS:
        cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
                   normalize_mode="zscore", prior_mean_mode="stationary",
                   laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
                   epochs=0, normalize_features=True,
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
        x = trainer.x_T.cpu().numpy()
        tmpl = trainer.template.cpu().numpy()
        n, d_eff = x.shape
        delta = x - tmpl
        S = compute_spectral_energy(delta, V)
        results_per_gamma[gamma] = {
            "V": V, "lam_L": lam_L, "delta": delta, "S": S,
            "n": n, "d": d_eff,
        }

    return data, y, mask, results_per_gamma


def run_dataset(ds, device="cpu"):
    print("\n=== %s ===" % ds, flush=True)
    data, y, mask, gamma_data = prepare_dataset(ds, device)

    # Track best across all parameter regimes
    best = {"stationary": {"auc": 0, "cfg": ""},
            "dynamic_hard": {"auc": 0, "cfg": ""},   # eta=0, finite Gamma
            "dynamic_soft": {"auc": 0, "cfg": ""},   # eta>0, finite Gamma
            "stationary_soft": {"auc": 0, "cfg": ""}, # eta>0, Gamma=inf
            "overall": {"auc": 0, "cfg": ""}}

    # ML-selected: track config with highest marginal likelihood
    ml_best = {"ml": -np.inf, "auc": 0, "cfg": "", "scores": None}

    total_configs = 0

    for gamma in GAMMAS:
        if gamma not in gamma_data:
            continue
        gd = gamma_data[gamma]
        V, lam_L, delta, S = gd["V"], gd["lam_L"], gd["delta"], gd["S"]
        n, d = gd["n"], gd["d"]

        # Sweep all (Gamma, eta, kappa) combos
        # Include Gamma=999 as stationary proxy
        for Gamma_t in GAMMA_T + [999.0]:
            for eta in ETAS:
                # Optimize rho for each kappa
                best_ml_this, best_rho, best_kappa = -np.inf, 0.5, 0.0
                for kk in KAPPAS:
                    rho = solve_rho_soc(S, lam_L, kk, Gamma_t, eta, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = ml_soc(S, q, Gamma_t, eta, n, d)
                    if ml > best_ml_this:
                        best_ml_this, best_rho, best_kappa = ml, rho, kk

                q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
                je, jr = score_nodes_soc(delta, V, q, Gamma_t, eta)
                if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                    continue

                ks_je = ks_null(je[mask])
                ks_jr = ks_null(jr[mask])
                sel = je if ks_je > ks_jr else jr
                sname = "J*" if ks_je > ks_jr else "R"
                auc = roc_auc_score(y[mask], sel[mask])
                total_configs += 1

                cfg_str = "g=%s G=%s eta=%s %s rho=%.3f kap=%.1f" % (
                    gamma, Gamma_t, eta, sname, best_rho, best_kappa)

                # Classify into regime
                is_stat = (Gamma_t > 100)
                is_soft = (eta > 0)

                if is_stat and not is_soft:
                    regime = "stationary"
                elif not is_stat and not is_soft:
                    regime = "dynamic_hard"
                elif not is_stat and is_soft:
                    regime = "dynamic_soft"
                else:
                    regime = "stationary_soft"

                if auc > best[regime]["auc"]:
                    best[regime] = {"auc": auc, "cfg": cfg_str, "ml": best_ml_this}

                if auc > best["overall"]["auc"]:
                    best["overall"] = {"auc": auc, "cfg": cfg_str, "ml": best_ml_this,
                                       "regime": regime}

                # Track ML-selected (highest marginal likelihood)
                if best_ml_this > ml_best["ml"]:
                    ml_best = {"ml": best_ml_this, "auc": auc, "cfg": cfg_str,
                               "regime": regime}

    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)

    print("  Configs evaluated: %d" % total_configs)
    print("  --- Results by regime ---")
    for regime in ["stationary", "dynamic_hard", "dynamic_soft", "stationary_soft"]:
        r = best[regime]
        if r["auc"] > 0:
            print("  %-18s %.1f%%  %s" % (regime + ":", r["auc"]*100, r["cfg"]))
    print("  ---")
    print("  ORACLE (AUC):     %.1f%%  %s" % (best["overall"]["auc"]*100, best["overall"]["cfg"]))
    print("  ML-SELECTED:      %.1f%%  %s" % (ml_best["auc"]*100, ml_best["cfg"]))
    print("  Baseline:         %.1f%%  Current pipeline: %.1f%%" % (base, curr))
    gap = best["overall"]["auc"]*100 - ml_best["auc"]*100
    print("  Oracle-ML gap:    %.1f%%" % gap)
    diff_base = ml_best["auc"]*100 - base
    diff_curr = ml_best["auc"]*100 - curr
    print("  ML vs Baseline: %+.1f%%  ML vs Current: %+.1f%%" % (diff_base, diff_curr))

    best["ml_selected"] = ml_best
    return best


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
            best = run_dataset(ds, args.device)
            summary[ds] = best
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  FAILED: %s" % str(e)[:100])

    if len(summary) > 1:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("%-20s %8s %8s %8s %6s  %s" % (
            "Dataset", "Oracle", "ML-Sel", "Gap", "Base", "ML config"))
        for ds in datasets:
            if ds in summary:
                b = summary[ds]
                oracle = b["overall"]["auc"]*100
                ml_sel = b["ml_selected"]["auc"]*100
                gap = oracle - ml_sel
                base = BASELINES.get(ds, 0)
                print("%-20s %7.1f%% %7.1f%% %5.1f%% %5.1f%%  %s" % (
                    ds, oracle, ml_sel, gap, base,
                    b["ml_selected"]["cfg"]))


if __name__ == "__main__":
    main()
