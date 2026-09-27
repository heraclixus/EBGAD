"""Test: ML selects gamma, corr(J*,R) selects score.

If corr(J*, R) < threshold -> R (they disagree, magnitude misleading)
If corr(J*, R) >= threshold -> J* (they agree, magnitude helps)
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
GT = {"weibo":"J*","reddit":"J*","amazon":"J*","yelpchi":"J*","blogcatalog":"J*",
      "facebook":"R","acm":"J*","elliptic":"R","elliptic_plus_plus":"R","t_finance":"R"}
BASELINES = {"weibo":95.0,"reddit":60.3,"amazon":78.1,"yelpchi":57.2,"blogcatalog":82.5,
             "facebook":91.4,"acm":88.8,"elliptic":65.3,"elliptic_plus_plus":62.9,"t_finance":83.5}


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

    print("%-15s %5s %6s %6s %6s %5s %5s" %
          ("Dataset", "MLg", "corr", "J*", "R", "gt", "pred"))

    for ds in datasets:
        try:
            load_name = ("YelpChi" if ds == "yelpchi" else
                         "Facebook" if ds == "facebook" else ds)
            data = load_data(load_name)
            y = data.y.numpy()
            mask = (y >= 0) & (y <= 1)
            d = data.x.shape[1]
            pca_dim = 64 if d > 100 else None

            # ML selects gamma
            best_ml_val = -np.inf
            best_gamma = 0.5
            for gamma in [0.5, 1.0]:
                je, jr, ml, rho = run_gamma(data, gamma, pca_dim, args.device)
                if ml > best_ml_val:
                    best_ml_val = ml
                    best_gamma = gamma
                    best_je, best_jr, best_rho = je, jr, rho

            je, jr = best_je, best_jr
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue

            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            corr = float(spearmanr(je[mask], jr[mask])[0])

            actual = GT[ds]
            results[ds] = {
                "gamma": best_gamma, "corr": corr,
                "auc_je": auc_je, "auc_jr": auc_jr,
                "rho": best_rho, "actual": actual,
            }

            print("%-15s %5s %+5.2f %5.1f%% %5.1f%%  %3s" %
                  (ds, best_gamma, corr, auc_je*100, auc_jr*100, actual), flush=True)
        except Exception as e:
            print("%-15s ERROR: %s" % (ds, str(e)[:60]))

    # Test thresholds
    print("\n=== THRESHOLD ANALYSIS ===")
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        correct = 0
        total = 0
        for ds, r in results.items():
            pred = "R" if r["corr"] < thresh else "J*"
            if pred == r["actual"]:
                correct += 1
            total += 1
        print("  corr < %.1f -> R:  %d/%d correct" % (thresh, correct, total))

    # Show per-dataset for best threshold
    print("\n=== BEST RULE DETAIL ===")
    for ds, r in results.items():
        for thresh in [0.3]:
            pred = "R" if r["corr"] < thresh else "J*"
            ok = "OK" if pred == r["actual"] else "MISS"
            sel_auc = r["auc_jr"] if pred == "R" else r["auc_je"]
            oracle_auc = max(r["auc_je"], r["auc_jr"])
            base = BASELINES.get(ds, 0)
            print("%-15s corr=%+.2f pred=%s actual=%s %s  sel=%.1f%% oracle=%.1f%% vsBase=%+.1f" %
                  (ds, r["corr"], pred, r["actual"], ok, sel_auc*100, oracle_auc*100, sel_auc*100-base))

    os.makedirs("results", exist_ok=True)
    with open("results/corr_rule.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
