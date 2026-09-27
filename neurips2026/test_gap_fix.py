"""Targeted fix for BlogCat, Facebook, ACM gaps.

At the density-adjusted gamma, try:
- Both templates (low, affinity)
- Multiple PCA dims (16, 32, 64, 128 based on d)
- KS selects across the small grid
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
BASELINES = {"blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8}
REPORTED = {"blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5}


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


def run_config(data, tmpl, gamma, pca_dim, device):
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
    elif pca_dim and d <= pca_dim:
        return None
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
    return je, jr, best_ml, rho


def run_dataset(ds, gammas, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]

    # PCA candidates
    pca_dims = [None]
    for p in [16, 32, 64, 128]:
        if d > p:
            pca_dims.append(p)

    candidates = []
    for gamma in gammas:
        for tmpl in ["low", "affinity"]:
            for pca in pca_dims:
                tag = "%s p%s g=%s" % (tmpl[:3], pca or "no", gamma)
                try:
                    result = run_config(data, tmpl, gamma, pca, device)
                    if result is None:
                        continue
                    je, jr, ml, rho = result
                    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                        continue
                    auc_je = roc_auc_score(y[mask], je[mask])
                    auc_jr = roc_auc_score(y[mask], jr[mask])
                    ks_je = ks_null(je[mask])
                    ks_jr = ks_null(jr[mask])

                    for sname, auc, ks in [("J*", auc_je, ks_je), ("R", auc_jr, ks_jr)]:
                        candidates.append({
                            "tag": "%s %s" % (sname, tag), "auc": auc, "ks": ks,
                            "ml": ml, "rho": rho, "tmpl": tmpl, "pca": pca,
                            "gamma": gamma, "score": sname,
                        })

                    print("    %s: J*=%.1f%% R=%.1f%% rho=%.3f ml=%.0f" %
                          (tag, auc_je*100, auc_jr*100, rho, ml), flush=True)
                except Exception as e:
                    print("    %s: ERROR %s" % (tag, str(e)[:60]))

    if not candidates:
        return None

    oracle = max(candidates, key=lambda c: c["auc"])

    # KS selection
    ks_best = max(candidates, key=lambda c: c["ks"])

    # ML selection (pick by ML, then KS between J*/R at that config)
    ml_best_cfg = max(candidates, key=lambda c: c["ml"])
    # Find J* and R at same config
    ml_cfg_key = (ml_best_cfg["tmpl"], ml_best_cfg["pca"], ml_best_cfg["gamma"])
    ml_pair = [c for c in candidates
               if (c["tmpl"], c["pca"], c["gamma"]) == ml_cfg_key]
    if ml_pair:
        ml_sel = max(ml_pair, key=lambda c: c["ks"])
    else:
        ml_sel = ml_best_cfg

    # Per-PCA KS: for each PCA, find best by KS, then pick the PCA with best result
    pca_bests = {}
    for pca in pca_dims:
        pca_cands = [c for c in candidates if c["pca"] == pca]
        if pca_cands:
            pca_bests[pca] = max(pca_cands, key=lambda c: c["ks"])

    return {
        "oracle": oracle,
        "ks_best": ks_best,
        "ml_sel": ml_sel,
        "pca_bests": pca_bests,
        "n_candidates": len(candidates),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    configs = {
        "blogcatalog": {"gammas": [0.5]},       # density-adjusted picks 0.5
        "facebook": {"gammas": [1.0]},           # density-adjusted picks 1.0
        "acm": {"gammas": [0.5, 1.0]},          # try both moderate gammas
    }

    for ds, cfg in configs.items():
        print("\n=== %s ===" % ds, flush=True)
        r = run_dataset(ds, cfg["gammas"], args.device)
        if r:
            base = BASELINES[ds]
            rep = REPORTED[ds]
            print("\n  Oracle: %.1f%% (%s)" % (r["oracle"]["auc"]*100, r["oracle"]["tag"]))
            print("  KS-sel: %.1f%% (%s)" % (r["ks_best"]["auc"]*100, r["ks_best"]["tag"]))
            print("  ML-sel: %.1f%% (%s)" % (r["ml_sel"]["auc"]*100, r["ml_sel"]["tag"]))
            print("  vsBase: KS=%+.1f ML=%+.1f Oracle=%+.1f" %
                  (r["ks_best"]["auc"]*100 - base,
                   r["ml_sel"]["auc"]*100 - base,
                   r["oracle"]["auc"]*100 - base))
            print("  Per-PCA best:")
            for pca, best in sorted(r["pca_bests"].items(), key=lambda x: str(x[0])):
                print("    PCA=%s: %.1f%% (%s)" % (pca, best["auc"]*100, best["tag"]))

    os.makedirs("results", exist_ok=True)
    with open("results/gap_fix.json", "w") as f:
        json.dump({"note": "see logs"}, f)


if __name__ == "__main__":
    main()
