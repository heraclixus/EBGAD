"""Test constrained rho optimization: rho >= rho_min.

At gamma=1.0, ML drives rho->0 on many datasets. Forcing rho >= rho_min
ensures the graph is used, potentially improving anomaly separation.
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


def solve_rho_constrained(S, lam_L, kappa, d, rho_min=0.0):
    """Newton solver for rho with lower bound constraint."""
    rho = solve_rho_newton(S, lam_L, kappa, d)
    return max(rho, rho_min)


def compute_edge_homophily(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean())


def run_dataset(ds, rho_min, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None
    homophily = compute_edge_homophily(data)

    # Gamma selection (same as our pipeline)
    if homophily < 0.10:
        gammas = [0.1, 0.2, 0.5]
    else:
        gammas = [1.0]

    cfg_base = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
                    alpha=2.0, T=1.0, normalize_mode="zscore",
                    prior_mean_mode="stationary", laplacian_variant="sym",
                    d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
                    template_type="low", graph_type="original")
    if pca_dim and d > pca_dim:
        cfg_base.update(dict(use_encoder=True, encoder_type="pca",
                            encoder_hid_dim=pca_dim, encoder_num_layers=1,
                            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                            encoder_alpha=1.0, encoder_weight_decay=0.0))

    best_ml = -np.inf
    best_result = None

    for gamma in gammas:
        cfg = dict(cfg_base, gamma=gamma)
        trainer = SOCTrainer(SOCTrainerConfig(**cfg))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl = trainer.template.cpu()
        x = trainer.x_T.cpu()
        n, d_eff = x.shape
        delta = x.numpy() - tmpl.numpy()
        S = compute_spectral_energy(delta, V)

        # Constrained rho optimization
        best_ml_g, rho, kappa = -np.inf, 0.5, 0.0
        for k in KAPPAS:
            r = solve_rho_constrained(S, lam_L, k, d_eff, rho_min=rho_min)
            q = Q_rho_eigenvalues(lam_L, r, k)
            ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
            if ml > best_ml_g:
                best_ml_g, rho, kappa = ml, r, k

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
        V_t = torch.from_numpy(V).float()
        with torch.no_grad():
            je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()

        if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
            continue

        if best_ml_g > best_ml:
            best_ml = best_ml_g
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            auc = roc_auc_score(y[mask], sel[mask])
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            best_result = {
                "auc": auc, "auc_je": auc_je, "auc_jr": auc_jr,
                "score": "J*" if use_je else "R",
                "gamma": gamma, "rho": rho, "kappa": kappa,
            }

    return best_result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = list(REPORTED.keys())
    rho_mins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    # Collect all results
    all_results = {}
    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        all_results[ds] = {}
        for rm in rho_mins:
            try:
                r = run_dataset(ds, rm, args.device)
                if r:
                    all_results[ds][rm] = r
                    print("  rho_min=%.2f: %s=%.1f%% (g=%s rho=%.3f)" %
                          (rm, r["score"], r["auc"]*100, r["gamma"], r["rho"]),
                          flush=True)
            except Exception as e:
                print("  rho_min=%.2f: ERROR %s" % (rm, str(e)[:60]))

    # Summary
    print("\n=== SUMMARY ===")
    print("%-15s" % "Dataset", end="")
    for rm in rho_mins:
        print(" %8s" % ("rm=%.2f" % rm), end="")
    print("  Reported  Baseline")

    for ds in datasets:
        print("%-15s" % ds, end="")
        for rm in rho_mins:
            if rm in all_results.get(ds, {}):
                auc = all_results[ds][rm]["auc"] * 100
                print(" %7.1f%%" % auc, end="")
            else:
                print(" %8s" % "---", end="")
        print("  %6.1f%%  %6.1f%%" % (REPORTED[ds], BASELINES[ds]))

    print("\nBeats baseline:")
    for rm in rho_mins:
        cnt = sum(1 for ds in datasets
                 if rm in all_results.get(ds, {}) and
                 all_results[ds][rm]["auc"]*100 >= BASELINES[ds])
        print("  rho_min=%.2f: %d/%d" % (rm, cnt, len(datasets)))

    os.makedirs("results", exist_ok=True)
    with open("results/rho_constraint.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
