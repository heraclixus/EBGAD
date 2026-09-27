"""Multi-scale score aggregation: average scores across gamma values.

Option 1: Uniform average over density-adjusted range
Option 2: BMA with tempered ML weights
Option 3: Broad range {0.1, 0.2, 0.5, 1.0, 2.0, 5.0} with ML weights

For each gamma, KS selects J* vs R, z-score normalize, then aggregate.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest
from scipy.special import softmax

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
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 70.6, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 63.1,
}
CURRENT_PIPELINE = {
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


def zscore(s):
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-12: return np.zeros_like(s)
    return (s - mu) / sigma


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = data.edge_index.shape[1] // 2 / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean()), density


def get_density_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in ALL_GAMMAS if g <= max_gamma]
    if not nearest: nearest = [ALL_GAMMAS[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


def run_gamma(data, gamma, pca_dim, device):
    d = data.x.shape[1]
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
    density_range = get_density_range(h, density)

    # Compute scores at ALL gammas (for broad range option)
    gamma_data = {}
    for gamma in ALL_GAMMAS:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            # KS selects score at this gamma
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            sel = je if ks_je > ks_jr else jr
            sname = "J*" if ks_je > ks_jr else "R"
            auc = roc_auc_score(y[mask], sel[mask])
            gamma_data[gamma] = {
                "scores": sel, "ml": ml, "rho": rho, "sname": sname, "auc": auc,
            }
            print("    g=%s: %s=%.1f%% ml=%.0f rho=%.3f" %
                  (gamma, sname, auc*100, ml, rho), flush=True)
        except:
            pass

    if not gamma_data:
        return None

    results = {}

    # === Option 1: Uniform average over density-adjusted range ===
    range_gammas = [g for g in density_range if g in gamma_data]
    if range_gammas:
        avg_scores = np.mean([zscore(gamma_data[g]["scores"][mask]) for g in range_gammas], axis=0)
        # Map back to full array for AUROC
        full_avg = np.zeros(len(y))
        full_avg[mask] = avg_scores
        auc1 = roc_auc_score(y[mask], full_avg[mask])
        results["uniform_range"] = {"auc": auc1, "gammas": range_gammas}

    # === Option 2: BMA with tempered ML weights over density range ===
    if range_gammas:
        mls = np.array([gamma_data[g]["ml"] for g in range_gammas])
        for T_scale in [1, 5, 10, 50]:
            # Temperature = range of MLs / T_scale
            ml_range = mls.max() - mls.min()
            T = max(ml_range / T_scale, 1.0)
            weights = softmax(mls / T)
            weighted_scores = np.zeros(mask.sum())
            for i, g in enumerate(range_gammas):
                weighted_scores += weights[i] * zscore(gamma_data[g]["scores"][mask])
            full_w = np.zeros(len(y))
            full_w[mask] = weighted_scores
            auc2 = roc_auc_score(y[mask], full_w[mask])
            results["bma_range_T%d" % T_scale] = {"auc": auc2, "weights": {g: float(w) for g, w in zip(range_gammas, weights)}}

    # === Option 3: Uniform average over ALL gammas ===
    all_gammas = sorted(gamma_data.keys())
    if all_gammas:
        avg_all = np.mean([zscore(gamma_data[g]["scores"][mask]) for g in all_gammas], axis=0)
        full_all = np.zeros(len(y))
        full_all[mask] = avg_all
        auc3 = roc_auc_score(y[mask], full_all[mask])
        results["uniform_all"] = {"auc": auc3, "gammas": all_gammas}

    # === Option 4: BMA with tempered ML over ALL gammas ===
    if len(all_gammas) > 1:
        mls_all = np.array([gamma_data[g]["ml"] for g in all_gammas])
        for T_scale in [5, 10, 50]:
            ml_range = mls_all.max() - mls_all.min()
            T = max(ml_range / T_scale, 1.0)
            weights = softmax(mls_all / T)
            weighted = np.zeros(mask.sum())
            for i, g in enumerate(all_gammas):
                weighted += weights[i] * zscore(gamma_data[g]["scores"][mask])
            full_w = np.zeros(len(y))
            full_w[mask] = weighted
            auc4 = roc_auc_score(y[mask], full_w[mask])
            results["bma_all_T%d" % T_scale] = {"auc": auc4}

    # === Current pipeline (ML-select single gamma from range) ===
    if range_gammas:
        ml_best_g = max(range_gammas, key=lambda g: gamma_data[g]["ml"])
        results["ml_select"] = {"auc": gamma_data[ml_best_g]["auc"], "gamma": ml_best_g}

    return results, density_range


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    ds = args.dataset
    print("=== %s ===" % ds, flush=True)
    r, density_range = run_dataset(ds, args.device)
    if r:
        base = BASELINES.get(ds, 0)
        curr = CURRENT_PIPELINE.get(ds, 0)
        print("\n  Results (range=%s):" % density_range)
        for name, info in sorted(r.items()):
            auc = info["auc"] * 100
            print("  %-20s %.1f%%  vsBase=%+.1f  vsCurr=%+.1f" %
                  (name, auc, auc - base, auc - curr))

    os.makedirs("results", exist_ok=True)
    with open("results/multiscale_%s.json" % ds, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                   for k, v in r.items()} if r else {}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
