"""Run full density-adjusted pipeline with minmax normalization.

Same pipeline as test_gamma_density.py but with normalize_mode="minmax".
Compare against zscore results to see which normalization works better per dataset.
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
GAMMA_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 70.6, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 63.1,
}
# Our zscore pipeline results for comparison
ZSCORE_RESULTS = {
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


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = data.edge_index.shape[1] // 2 / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean()), density


def get_gamma_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in GAMMA_GRID if g <= max_gamma]
    if not nearest: nearest = [GAMMA_GRID[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


def run_gamma(data, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
               normalize_mode="minmax", prior_mean_mode="stationary",
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
        if ml > best_ml: best_ml, rho, kappa = ml, r, k
    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
    V_t = torch.from_numpy(V).float()
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2, 1,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2, 1,
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
    h, density = compute_graph_stats(data)
    gamma_range = get_gamma_range(h, density)

    best_ml = -np.inf
    best_gamma = gamma_range[0]
    best_je, best_jr = None, None

    for gamma in gamma_range:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            print("    g=%s: J*=%.1f%% R=%.1f%% ml=%.0f rho=%.3f" %
                  (gamma, roc_auc_score(y[mask], je[mask])*100,
                   roc_auc_score(y[mask], jr[mask])*100, ml, rho), flush=True)
            if ml > best_ml:
                best_ml = ml
                best_gamma = gamma
                best_je, best_jr = je, jr
                best_rho = rho
        except:
            pass

    if best_je is None:
        return None

    ks_je = ks_null(best_je[mask])
    ks_jr = ks_null(best_jr[mask])
    use_je = ks_je > ks_jr
    scores = best_je if use_je else best_jr
    score_name = "J*" if use_je else "R"
    auc = roc_auc_score(y[mask], scores[mask])

    return {
        "auc": auc, "score": score_name, "gamma": best_gamma,
        "rho": best_rho, "ks": max(ks_je, ks_jr),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(BASELINES.keys())

    for ds in datasets:
        print("\n=== %s (minmax) ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                base = BASELINES.get(ds, 0)
                zs = ZSCORE_RESULTS.get(ds, 0)
                print("  Minmax: %.1f%% (%s g=%s rho=%.3f KS=%.3f)" %
                      (r["auc"]*100, r["score"], r["gamma"], r["rho"], r["ks"]))
                print("  Zscore: %.1f%%" % zs)
                print("  Better: %s (%+.1fpp)" %
                      ("minmax" if r["auc"]*100 > zs else "zscore",
                       r["auc"]*100 - zs))
                print("  vsBase: %+.1fpp" % (max(r["auc"]*100, zs) - base))
        except Exception as e:
            print("  ERROR: %s" % str(e)[:80])


if __name__ == "__main__":
    main()
