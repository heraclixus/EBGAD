"""ACM targeted sweep: PCA dimensions, truncated eigendecomposition, all scores.

Test whether J*/R with:
1. Different PCA dims (16, 64, 128, 256)
2. Truncated eigendecomposition (k=500, 1000, 2000 vs full 16K)
can match or beat the C score.
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
from soc.soc_anomaly import (precision_energy_anomaly, precision_ratio_anomaly,
                              control_energy_anomaly, ce_ratio_anomaly)
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def run_config(data, gamma, pca_dim, truncated_k=None, device="cpu"):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="low", graph_type="original", gamma=gamma)
    if pca_dim is not None and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    if truncated_k is not None:
        cfg["truncated_k"] = truncated_k

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

    results = {}
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()
        results["J*"] = je
        results["R"] = jr
        try:
            ce = control_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                        lam_penalty=50.0).cpu().numpy()
            cr = ce_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                  lam_penalty=50.0).cpu().numpy()
            results["C"] = ce
            results["C_R"] = cr
        except:
            pass

    return results, best_ml, rho, kappa, V.shape[1]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_data("acm")
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    n, d = data.x.shape
    print("ACM: n=%d d=%d" % (n, d), flush=True)

    # Compute effective rank (90% variance)
    x_np = data.x.float().numpy()
    x_c = x_np - x_np.mean(axis=0)
    from sklearn.utils.extmath import randomized_svd
    _, sv, _ = randomized_svd(x_c, n_components=min(500, d, n))
    sv_sq = sv ** 2
    cumvar = np.cumsum(sv_sq) / sv_sq.sum()
    eff_rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    eff_rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    eff_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print("Effective rank: 90%%=%d  95%%=%d  99%%=%d" %
          (eff_rank_90, eff_rank_95, eff_rank_99), flush=True)

    # Sweep: PCA dims × gammas × truncated_k
    configs = []

    for pca in [16, 64, 128, 256, eff_rank_90]:
        for gamma in [0.5, 1.0, 2.0]:
            for trunc_k in [None, 500, 1000, 2000]:
                tag = "pca%d g=%s k=%s" % (pca, gamma, trunc_k or "full")
                try:
                    scores, ml, rho, kappa, actual_k = run_config(
                        data, gamma, pca, trunc_k, args.device)
                    aucs = {}
                    for sname, sarr in scores.items():
                        if not np.any(np.isnan(sarr)):
                            aucs[sname] = roc_auc_score(y[mask], sarr[mask])
                    best_score = max(aucs, key=aucs.get)
                    best_auc = aucs[best_score]
                    print("  %s: best=%s(%.1f%%) rho=%.3f k=%d  [%s]" %
                          (tag, best_score, best_auc*100, rho, actual_k,
                           " ".join("%s=%.1f" % (s, a*100) for s, a in sorted(aucs.items()))),
                          flush=True)
                    configs.append({
                        "tag": tag, "pca": pca, "gamma": gamma, "trunc_k": trunc_k,
                        "best_score": best_score, "best_auc": best_auc,
                        "rho": rho, "actual_k": actual_k, "aucs": aucs,
                    })
                except Exception as e:
                    print("  %s: ERROR %s" % (tag, str(e)[:60]))

    # Summary
    if configs:
        print("\n=== TOP 10 CONFIGS ===")
        configs.sort(key=lambda c: c["best_auc"], reverse=True)
        for c in configs[:10]:
            print("  %.1f%% %s (%s)" % (c["best_auc"]*100, c["best_score"], c["tag"]))

        print("\n=== BEST J* (no C) ===")
        je_configs = [c for c in configs if "J*" in c["aucs"]]
        je_configs.sort(key=lambda c: c["aucs"]["J*"], reverse=True)
        for c in je_configs[:5]:
            print("  J*=%.1f%% R=%.1f%% (%s)" %
                  (c["aucs"]["J*"]*100, c["aucs"].get("R", 0)*100, c["tag"]))

        print("\n=== BEST R (no C) ===")
        r_configs = [c for c in configs if "R" in c["aucs"]]
        r_configs.sort(key=lambda c: c["aucs"]["R"], reverse=True)
        for c in r_configs[:5]:
            print("  R=%.1f%% J*=%.1f%% (%s)" %
                  (c["aucs"]["R"]*100, c["aucs"].get("J*", 0)*100, c["tag"]))

    os.makedirs("results", exist_ok=True)
    with open("results/acm_sweep.json", "w") as f:
        json.dump(configs, f, indent=2, default=str)


if __name__ == "__main__":
    main()
