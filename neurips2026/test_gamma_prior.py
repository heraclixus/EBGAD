"""MAP with log-normal prior on gamma, centered by homophily.

gamma*(h) = 0.5 + h  (low homophily -> small gamma, high -> large)
p(gamma|h) = LogNormal(log gamma*(h), sigma^2)

MAP = ML(rho, kappa; gamma) - (log gamma - log gamma*(h))^2 / (2*sigma^2)
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


def gamma_center(h):
    """Predicted gamma center from homophily."""
    return 0.5 + h  # low h -> ~0.5, high h -> ~1.5


def log_gamma_prior(gamma, gamma_star, sigma):
    """Log of log-normal prior on gamma centered at gamma_star."""
    return -(np.log(gamma) - np.log(gamma_star))**2 / (2 * sigma**2)


def run_gamma_config(data, gamma, pca_dim, device):
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


def run_dataset(ds, sigma, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None
    h = compute_edge_homophily(data)
    g_star = gamma_center(h)

    best_map = -np.inf
    best_result = None

    for gamma in GAMMAS:
        try:
            je, jr, ml, rho = run_gamma_config(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            log_prior = log_gamma_prior(gamma, g_star, sigma)
            map_val = ml + log_prior

            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            auc = roc_auc_score(y[mask], sel[mask])

            if map_val > best_map:
                best_map = map_val
                best_result = {
                    "auc": auc, "score": "J*" if use_je else "R",
                    "gamma": gamma, "rho": rho, "ml": ml,
                    "map": map_val, "log_prior": log_prior,
                }
        except:
            pass

    return best_result, h, g_star


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = list(REPORTED.keys())
    # Test different prior widths
    sigmas = [0.3, 0.5, 0.7, 1.0, 2.0, 999.0]  # 999 = effectively no prior (= ML)

    for sigma in sigmas:
        print("\n======= sigma=%.1f =======" % sigma, flush=True)
        results = {}
        for ds in datasets:
            print("=== %s ===" % ds, flush=True)
            try:
                r, h, g_star = run_dataset(ds, sigma, args.device)
                if r:
                    rep = REPORTED[ds]
                    base = BASELINES[ds]
                    print("  h=%.3f g*=%.2f -> g=%s %s=%.1f%% rho=%.3f prior=%.1f" %
                          (h, g_star, r["gamma"], r["score"], r["auc"]*100,
                           r["rho"], r["log_prior"]), flush=True)
                    results[ds] = r
            except Exception as e:
                print("  ERROR: %s" % str(e)[:60])

        if len(results) > 1:
            print("\n--- sigma=%.1f summary ---" % sigma)
            print("%-15s %7s %5s %7s %7s" %
                  ("Dataset", "AUROC", "g", "vsRep", "vsBase"))
            for ds in datasets:
                if ds in results:
                    r = results[ds]
                    rep = REPORTED[ds]
                    base = BASELINES[ds]
                    print("%-15s %6.1f%% %5s %+6.1f %+6.1f" %
                          (ds, r["auc"]*100, r["gamma"],
                           r["auc"]*100-rep, r["auc"]*100-base))
            beats = sum(1 for ds, r in results.items()
                       if r["auc"]*100 >= BASELINES[ds])
            mean_auc = np.mean([r["auc"] for r in results.values()]) * 100
            print("  Mean: %.1f%%  Beats baseline: %d/%d" %
                  (mean_auc, beats, len(results)))

    os.makedirs("results", exist_ok=True)
    with open("results/gamma_prior.json", "w") as f:
        json.dump({"note": "see logs"}, f)


if __name__ == "__main__":
    main()
