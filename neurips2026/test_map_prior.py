"""MAP estimation with Beta prior on rho.

Instead of ML: max ML(rho)
Use MAP: max ML(rho) + (alpha-1)*log(rho) + (beta-1)*log(1-rho)

Beta(alpha, beta) prior on rho:
- alpha > 1 pushes rho away from 0 (graph should be used)
- beta = 1 (uniform on upper end, no constraint against rho=1)
- Test alpha = 1 (=ML), 2, 3, 5
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
    compute_spectral_energy,
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


def solve_rho_map(S, lam_L, kappa, d, alpha_prior=1.0, beta_prior=1.0,
                  max_iter=50, tol=1e-12):
    """Newton solver for rho with Beta(alpha, beta) prior.

    Objective: ML(rho) + (alpha-1)*log(rho) + (beta-1)*log(1-rho)
    """
    nu = 1
    a = (kappa**2 + lam_L)**nu - 1.0  # a_j = q_j(rho=1) - q_j(rho=0)
    k = len(a)

    rho = 0.5  # initial
    for _ in range(max_iter):
        q = rho * a + 1.0  # q_j = rho*a_j + 1
        q_safe = np.maximum(q, 1e-12)

        # ML gradient and Hessian
        grad_ml = np.sum(a * (-S / 2.0 + d / (2.0 * q_safe)))
        hess_ml = -np.sum(d * a**2 / (2.0 * q_safe**2))

        # Prior gradient and Hessian
        rho_safe = np.clip(rho, 1e-8, 1 - 1e-8)
        grad_prior = (alpha_prior - 1) / rho_safe - (beta_prior - 1) / (1 - rho_safe)
        hess_prior = -(alpha_prior - 1) / rho_safe**2 - (beta_prior - 1) / (1 - rho_safe)**2

        grad = grad_ml + grad_prior
        hess = hess_ml + hess_prior

        if abs(hess) < 1e-20:
            break

        step = -grad / hess
        rho_new = rho + step
        rho = np.clip(rho_new, 1e-6, 1 - 1e-6)

        if abs(step) < tol:
            break

    return float(rho)


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


def compute_edge_homophily(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean())


def run_dataset(ds, alpha_prior, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d_feat = data.x.shape[1]
    pca_dim = 64 if d_feat > 100 else None
    homophily = compute_edge_homophily(data)

    # Gamma from homophily rule
    if homophily < 0.10:
        gammas = [0.1, 0.2, 0.5]
    else:
        gammas = [1.0]

    cfg_base = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
                    alpha=2.0, T=1.0, normalize_mode="zscore",
                    prior_mean_mode="stationary", laplacian_variant="sym",
                    d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
                    template_type="low", graph_type="original")
    if pca_dim and d_feat > pca_dim:
        cfg_base.update(dict(use_encoder=True, encoder_type="pca",
                            encoder_hid_dim=pca_dim, encoder_num_layers=1,
                            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                            encoder_alpha=1.0, encoder_weight_decay=0.0))

    best_map = -np.inf
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

        # MAP optimize rho with Beta prior
        best_map_g, rho, kappa = -np.inf, 0.5, 0.0
        for k in KAPPAS:
            r = solve_rho_map(S, lam_L, k, d_eff,
                             alpha_prior=alpha_prior, beta_prior=1.0)
            q = Q_rho_eigenvalues(lam_L, r, k)
            ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
            # MAP = ML + log prior
            r_safe = np.clip(r, 1e-8, 1-1e-8)
            log_prior = (alpha_prior - 1) * np.log(r_safe)
            map_val = ml + log_prior
            if map_val > best_map_g:
                best_map_g, rho, kappa = map_val, r, k

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
        V_t = torch.from_numpy(V).float()
        with torch.no_grad():
            je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()

        if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
            continue

        ml_val = marginal_log_likelihood_stationary(
            S, Q_rho_eigenvalues(lam_L, rho, kappa), n, d_eff)

        if best_map_g > best_map:
            best_map = best_map_g
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            auc = roc_auc_score(y[mask], sel[mask])
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            best_result = {
                "auc": auc, "score": "J*" if use_je else "R",
                "gamma": gamma, "rho": rho, "kappa": kappa,
                "auc_je": auc_je, "auc_jr": auc_jr,
            }

    return best_result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = list(REPORTED.keys())
    alphas = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]  # alpha=1 is standard ML

    all_results = {}
    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        all_results[ds] = {}
        for alpha in alphas:
            try:
                r = run_dataset(ds, alpha, args.device)
                if r:
                    all_results[ds][alpha] = r
                    print("  alpha=%.1f: %s=%.1f%% (g=%s rho=%.3f kappa=%.2f)" %
                          (alpha, r["score"], r["auc"]*100,
                           r["gamma"], r["rho"], r["kappa"]), flush=True)
            except Exception as e:
                print("  alpha=%.1f: ERROR %s" % (alpha, str(e)[:60]))

    # Summary
    print("\n=== SUMMARY ===")
    print("%-15s" % "Dataset", end="")
    for a in alphas:
        print("  a=%-4s" % ("%.1f" % a), end="")
    print("  Report  Base")

    for ds in datasets:
        print("%-15s" % ds, end="")
        for a in alphas:
            if a in all_results.get(ds, {}):
                r = all_results[ds][a]
                print(" %5.1f%%" % (r["auc"]*100), end="")
            else:
                print("  ---  ", end="")
        print("  %5.1f%% %5.1f%%" % (REPORTED[ds], BASELINES[ds]))

    print("\nBeats baseline:")
    for a in alphas:
        cnt = sum(1 for ds in datasets
                 if a in all_results.get(ds, {}) and
                 all_results[ds][a]["auc"]*100 >= BASELINES[ds])
        print("  alpha=%.1f: %d/%d" % (a, cnt, len(datasets)))

    print("\nRho values:")
    for ds in datasets:
        print("%-15s" % ds, end="")
        for a in alphas:
            if a in all_results.get(ds, {}):
                print(" %5.3f" % all_results[ds][a]["rho"], end="")
            else:
                print("  --- ", end="")
        print()

    os.makedirs("results", exist_ok=True)
    with open("results/map_prior.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
