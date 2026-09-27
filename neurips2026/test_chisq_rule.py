"""Test chi-squared goodness-of-fit rule for J* vs R selection.

Under the null (no anomalies), J* scores follow a scaled chi-squared distribution.
If J* deviates strongly from this null -> J* captures real anomaly signal -> use J*.
If J* matches the null -> magnitude is random -> use R.

Measure deviation by: KS statistic, Anderson-Darling, or simple tail excess
relative to the fitted chi-squared.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest, chi2, anderson, spearmanr

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
GT = {"weibo":"J*","reddit":"J*","amazon":"J*","yelpchi":"J*","blogcatalog":"J*",
      "facebook":"R","acm":"J*","elliptic":"R","elliptic_plus_plus":"R","t_finance":"R"}
BASELINES = {"weibo":95.0,"reddit":60.3,"amazon":78.1,"yelpchi":57.2,"blogcatalog":82.5,
             "facebook":91.4,"acm":88.8,"elliptic":65.3,"elliptic_plus_plus":62.9,"t_finance":83.5}


def compute_null_deviation(scores):
    """Measure how much J* deviates from a chi-squared null.

    Fit chi-squared to scores via method of moments, then measure tail excess.
    """
    s = scores[scores > 0]
    mu = np.mean(s)
    var = np.var(s)

    # Method of moments: chi2(k) has mean=k, var=2k
    # Scaled chi2: a*chi2(k) has mean=a*k, var=2*a^2*k
    # So k = 2*mu^2/var, a = var/(2*mu)
    if var < 1e-12 or mu < 1e-12:
        return {"ks": 0, "tail_excess": 0, "tail_ratio": 0, "p99_ratio": 1.0, "k_est": 0}

    k_est = 2 * mu**2 / var
    a_est = var / (2 * mu)

    # Normalize scores to standard chi2(k)
    s_norm = s / a_est

    # KS test against chi2(k_est)
    try:
        ks_stat, ks_p = kstest(s_norm, 'chi2', args=(k_est,))
    except:
        ks_stat, ks_p = 0, 1

    # Tail excess: fraction of scores above the 99th percentile of fitted null
    null_p99 = chi2.ppf(0.99, k_est) * a_est
    tail_excess = float(np.mean(scores > null_p99))

    # Expected tail fraction under null: 1%
    # Ratio: observed/expected. If >> 1, real anomalies push the tail.
    tail_ratio = tail_excess / 0.01 if tail_excess > 0 else 0

    # p99 of actual vs p99 of null
    actual_p99 = np.percentile(scores, 99)
    p99_ratio = actual_p99 / (null_p99 + 1e-12)

    return {
        "ks": float(ks_stat),
        "tail_excess": tail_excess,
        "tail_ratio": tail_ratio,
        "p99_ratio": float(p99_ratio),
        "k_est": float(k_est),
    }


def run_gamma(data, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0,nu=1.0,rho=0.5,lam_penalty=50.0,alpha=2.0,T=1.0,
               normalize_mode="zscore",prior_mean_mode="stationary",laplacian_variant="sym",
               d_hidden=64,n_hidden_layers=2,epochs=0,normalize_features=True,
               template_type="low",graph_type="original",gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True,encoder_type="pca",encoder_hid_dim=pca_dim,
                       encoder_num_layers=1,encoder_dropout=0.0,encoder_lr=0.0,
                       encoder_epochs=0,encoder_alpha=1.0,encoder_weight_decay=0.0))
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = list(GT.keys())
    results = {}

    for ds in datasets:
        print("=== %s ===" % ds, flush=True)
        try:
            load_name = ("YelpChi" if ds == "yelpchi" else
                         "Facebook" if ds == "facebook" else ds)
            data = load_data(load_name)
            y = data.y.numpy()
            mask = (y >= 0) & (y <= 1)
            d = data.x.shape[1]
            pca_dim = 64 if d > 100 else None

            # Run at gamma=1.0 (fixed, since ML mostly picks this)
            je, jr, ml, rho = run_gamma(data, 1.0, pca_dim, args.device)

            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                print("  NaN scores, skipping")
                continue

            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])

            # Chi-squared null deviation for J*
            je_dev = compute_null_deviation(je[mask])
            # Also for R
            jr_dev = compute_null_deviation(jr[mask])

            actual = GT[ds]
            results[ds] = {
                "auc_je": auc_je, "auc_jr": auc_jr,
                "actual": actual, "rho": rho,
                "je_ks": je_dev["ks"], "je_tail_ratio": je_dev["tail_ratio"],
                "je_p99_ratio": je_dev["p99_ratio"], "je_k_est": je_dev["k_est"],
                "jr_ks": jr_dev["ks"], "jr_tail_ratio": jr_dev["tail_ratio"],
                "jr_p99_ratio": jr_dev["p99_ratio"],
            }

            print("  J*=%.1f%% R=%.1f%% actual=%s rho=%.3f" %
                  (auc_je*100, auc_jr*100, actual, rho))
            print("  J* null dev: KS=%.3f tail_ratio=%.1f p99_ratio=%.2f k=%.1f" %
                  (je_dev["ks"], je_dev["tail_ratio"], je_dev["p99_ratio"], je_dev["k_est"]))
            print("  R  null dev: KS=%.3f tail_ratio=%.1f p99_ratio=%.2f" %
                  (jr_dev["ks"], jr_dev["tail_ratio"], jr_dev["p99_ratio"]))

        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    # Test rules
    print("\n=== RULE ANALYSIS ===")
    print("%-15s %5s %5s %8s %8s %8s %8s %3s" %
          ("Dataset", "J*", "R", "J*_KS", "J*_tR", "J*_p99R", "R_p99R", "gt"))
    for ds in datasets:
        if ds not in results:
            continue
        r = results[ds]
        print("%-15s %4.1f%% %4.1f%% %8.3f %8.1f %8.2f %8.2f  %s" %
              (ds, r["auc_je"]*100, r["auc_jr"]*100,
               r["je_ks"], r["je_tail_ratio"], r["je_p99_ratio"],
               r["jr_p99_ratio"], r["actual"]))

    # Test: if J* tail_ratio > R tail_ratio -> J*, else R
    print("\n=== J* tail_ratio vs R tail_ratio ===")
    for ds, r in results.items():
        pred = "J*" if r["je_tail_ratio"] > r["jr_tail_ratio"] else "R"
        ok = "OK" if pred == r["actual"] else "MISS"
        print("  %-15s J*_tR=%.1f R_tR=%.1f pred=%s actual=%s %s" %
              (ds, r["je_tail_ratio"], r["jr_tail_ratio"], pred, r["actual"], ok))
    n_correct = sum(1 for r in results.values()
                    if ("J*" if r["je_tail_ratio"] > r["jr_tail_ratio"] else "R") == r["actual"])
    print("  Correct: %d/%d" % (n_correct, len(results)))

    # Test: if J* p99_ratio > R p99_ratio -> J*, else R
    print("\n=== J* p99_ratio vs R p99_ratio ===")
    for ds, r in results.items():
        pred = "J*" if r["je_p99_ratio"] > r["jr_p99_ratio"] else "R"
        ok = "OK" if pred == r["actual"] else "MISS"
        print("  %-15s J*_p99=%.2f R_p99=%.2f pred=%s actual=%s %s" %
              (ds, r["je_p99_ratio"], r["jr_p99_ratio"], pred, r["actual"], ok))
    n_correct = sum(1 for r in results.values()
                    if ("J*" if r["je_p99_ratio"] > r["jr_p99_ratio"] else "R") == r["actual"])
    print("  Correct: %d/%d" % (n_correct, len(results)))

    # Test: if J* KS > R KS -> J* deviates more from null -> J*, else R
    print("\n=== J* KS vs R KS ===")
    for ds, r in results.items():
        pred = "J*" if r["je_ks"] > r["jr_ks"] else "R"
        ok = "OK" if pred == r["actual"] else "MISS"
        print("  %-15s J*_ks=%.3f R_ks=%.3f pred=%s actual=%s %s" %
              (ds, r["je_ks"], r["jr_ks"], pred, r["actual"], ok))
    n_correct = sum(1 for r in results.values()
                    if ("J*" if r["je_ks"] > r["jr_ks"] else "R") == r["actual"])
    print("  Correct: %d/%d" % (n_correct, len(results)))

    os.makedirs("results", exist_ok=True)
    with open("results/chisq_rule.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
