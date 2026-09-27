"""Gamma selection: restrict ML search range by edge homophily + density.

Low homophily -> small gamma (don't over-smooth heterophilic edges)
High homophily -> large gamma (smoothing is safe)
High density -> allow larger gamma (many neighbors = robust consensus)

Then ML optimizes within the restricted range.
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
ALL_GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]
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
    """Mean cosine similarity across edges (sampled for speed)."""
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    src, dst = edge_index[0, idx], edge_index[1, idx]
    cos = torch.nn.functional.cosine_similarity(x[src], x[dst], dim=1)
    return float(cos.mean()), float(cos.std())


def get_gamma_range(homophily, density):
    """Determine gamma search range from graph properties.

    Low homophily -> small gamma (features differ across edges, don't over-smooth)
    High homophily -> large gamma OK (features agree, smoothing helps)
    High density -> shift range up (many neighbors give robust average)
    """
    # Base range from homophily
    if homophily < 0.10:
        base = [0.1, 0.2, 0.5]
    elif homophily < 0.5:
        base = [0.2, 0.5, 1.0]
    elif homophily < 0.8:
        base = [0.5, 1.0, 2.0]
    else:
        base = [0.5, 1.0, 2.0, 5.0]

    # Density bonus: dense graphs can handle more smoothing
    if density > 50:
        extra = [g * 2 for g in base if g * 2 <= 10]
        base = sorted(set(base + extra))
    if density > 200:
        # Very dense: add both extremes (strong smoothing works via averaging many neighbors)
        for g in [0.1, 0.2, 5.0, 10.0]:
            if g not in base:
                base.append(g)

    return sorted(set(base))


def run_gamma(data, gamma, pca_dim, device):
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


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    n = data.x.shape[0]
    pca_dim = 64 if d > 100 else None

    # Compute graph properties
    homophily, hom_std = compute_edge_homophily(data)
    n_edges = data.edge_index.shape[1] // 2
    density = n_edges / n

    # Get restricted gamma range
    gamma_range = get_gamma_range(homophily, density)
    print("  homophily=%.3f density=%.1f -> gamma_range=%s" %
          (homophily, density, gamma_range), flush=True)

    # Run all gammas (full range for oracle comparison)
    all_configs = {}
    for gamma in ALL_GAMMAS:
        try:
            je, jr, ml, rho = run_gamma(data, gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            use_je = ks_je > ks_jr
            sel = je if use_je else jr
            auc = roc_auc_score(y[mask], sel[mask])
            all_configs[gamma] = {
                "auc": auc, "ml": ml, "rho": rho,
                "sel_name": "J*" if use_je else "R",
            }
            in_range = "*" if gamma in gamma_range else " "
            print("    %sg=%s: %s=%.1f%% ml=%.0f rho=%.3f" %
                  (in_range, gamma, all_configs[gamma]["sel_name"],
                   auc*100, ml, rho), flush=True)
        except Exception as e:
            print("    g=%s: ERROR %s" % (gamma, str(e)[:60]))

    if not all_configs:
        return None

    # Oracle (full range)
    oracle_gamma = max(all_configs, key=lambda g: all_configs[g]["auc"])
    oracle_auc = all_configs[oracle_gamma]["auc"]

    # ML on full range
    ml_gamma_full = max(all_configs, key=lambda g: all_configs[g]["ml"])
    ml_auc_full = all_configs[ml_gamma_full]["auc"]

    # ML on restricted range
    restricted = {g: c for g, c in all_configs.items() if g in gamma_range}
    if restricted:
        ml_gamma_rest = max(restricted, key=lambda g: restricted[g]["ml"])
        ml_auc_rest = restricted[ml_gamma_rest]["auc"]
    else:
        ml_gamma_rest = ml_gamma_full
        ml_auc_rest = ml_auc_full

    # ML on restricted range, excluding rho < 0.05 (graph must be used)
    rest_rho = {g: c for g, c in restricted.items() if c["rho"] >= 0.05}
    if rest_rho:
        ml_gamma_rest_rho = max(rest_rho, key=lambda g: rest_rho[g]["ml"])
        ml_auc_rest_rho = rest_rho[ml_gamma_rest_rho]["auc"]
    else:
        ml_gamma_rest_rho = ml_gamma_rest
        ml_auc_rest_rho = ml_auc_rest

    # Fixed 1.0
    fixed_auc = all_configs.get(1.0, {"auc": 0})["auc"]

    return {
        "homophily": homophily, "density": density,
        "gamma_range": gamma_range,
        "oracle_gamma": oracle_gamma, "oracle_auc": oracle_auc,
        "ml_full_gamma": ml_gamma_full, "ml_full_auc": ml_auc_full,
        "ml_rest_gamma": ml_gamma_rest, "ml_rest_auc": ml_auc_rest,
        "ml_rest_rho_gamma": ml_gamma_rest_rho, "ml_rest_rho_auc": ml_auc_rest_rho,
        "fixed_auc": fixed_auc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    all_results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                rep = REPORTED.get(ds, 0)
                base = BASELINES.get(ds, 0)
                print("  Oracle: %.1f%% (g=%s)" % (r["oracle_auc"]*100, r["oracle_gamma"]))
                print("  ML(full):  %.1f%% (g=%s) gap=%+.1f" %
                      (r["ml_full_auc"]*100, r["ml_full_gamma"],
                       (r["ml_full_auc"]-r["oracle_auc"])*100))
                print("  ML(rest):  %.1f%% (g=%s) gap=%+.1f  range=%s" %
                      (r["ml_rest_auc"]*100, r["ml_rest_gamma"],
                       (r["ml_rest_auc"]-r["oracle_auc"])*100, r["gamma_range"]))
                print("  ML(r+rho): %.1f%% (g=%s) gap=%+.1f" %
                      (r["ml_rest_rho_auc"]*100, r["ml_rest_rho_gamma"],
                       (r["ml_rest_rho_auc"]-r["oracle_auc"])*100))
                print("  Fixed1.0:  %.1f%% gap=%+.1f" %
                      (r["fixed_auc"]*100, (r["fixed_auc"]-r["oracle_auc"])*100))
                print("  vsBase: ML(r+rho)=%+.1f  ML(rest)=%+.1f  fixed=%+.1f" %
                      (r["ml_rest_rho_auc"]*100 - base,
                       r["ml_rest_auc"]*100 - base, r["fixed_auc"]*100 - base))
                all_results[ds] = r
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(all_results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %5s %5s %6s %8s %8s %8s %8s" %
              ("Dataset", "hom", "dens", "Oracle", "ML_full", "ML_rest", "ML_r+rho", "Fix1.0"))
        for ds in datasets:
            if ds not in all_results:
                continue
            r = all_results[ds]
            print("%-15s %5.3f %5.1f %5.1f%% %6.1f%%(%s) %6.1f%%(%s) %6.1f%%(%s) %5.1f%%" %
                  (ds, r["homophily"], r["density"], r["oracle_auc"]*100,
                   r["ml_full_auc"]*100, r["ml_full_gamma"],
                   r["ml_rest_auc"]*100, r["ml_rest_gamma"],
                   r["ml_rest_rho_auc"]*100, r["ml_rest_rho_gamma"],
                   r["fixed_auc"]*100))

        for method in ["ml_full", "ml_rest", "ml_rest_rho", "fixed"]:
            key = method + "_auc"
            cnt = sum(1 for r in all_results.values()
                     if abs(r[key] - r["oracle_auc"]) < 0.05)
            print("  %s within 5pp: %d/%d" % (method, cnt, len(all_results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/gamma_restricted_%s.json" % args.dataset
               if args.dataset else "results/gamma_restricted.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
