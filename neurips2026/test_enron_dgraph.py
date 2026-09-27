"""Quick test of the density-adjusted pipeline on Enron and DGraph.

Enron: n=13K, d=18, h~0.65, needs affinity graph + high template
DGraph: n=3.7M, uses cached eigenpairs
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


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    n_edges = edge_index.shape[1] // 2
    density = n_edges / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    homophily = float(cos.mean())
    return homophily, density


def get_gamma_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in GAMMA_GRID if g <= max_gamma]
    if not nearest:
        nearest = [GAMMA_GRID[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return gamma_center, nearest


def run_enron(device="cpu"):
    print("=== ENRON ===", flush=True)
    data = load_data("enron")
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    h, density = compute_graph_stats(data)
    gamma_center, gamma_range = get_gamma_range(h, density)
    print("  h=%.3f dens=%.1f center=%.2f range=%s" % (h, density, gamma_center, gamma_range))

    # Try both templates, both graph types
    best_result = None
    best_ml = -np.inf

    for graph_type in ["original", "affinity"]:
        for tmpl in ["low", "affinity", "high"]:
            for gamma in GAMMA_GRID:
                try:
                    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
                               alpha=2.0, T=1.0, normalize_mode="zscore",
                               prior_mean_mode="stationary", laplacian_variant="sym",
                               d_hidden=64, n_hidden_layers=2, epochs=0,
                               normalize_features=True,
                               template_type=tmpl, graph_type=graph_type, gamma=gamma)
                    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
                    trainer.train(data, device=device)
                    V = trainer.V.cpu().numpy()
                    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
                    tmpl_t = trainer.template.cpu()
                    x = trainer.x_T.cpu()
                    n, d_eff = x.shape
                    delta = x.numpy() - tmpl_t.numpy()
                    S = compute_spectral_energy(delta, V)
                    ml, rho, kappa = -np.inf, 0.5, 0.0
                    for k in KAPPAS:
                        r = solve_rho_newton(S, lam_L, k, d_eff)
                        q = Q_rho_eigenvalues(lam_L, r, k)
                        m = marginal_log_likelihood_stationary(S, q, n, d_eff)
                        if m > ml:
                            ml, rho, kappa = m, r, k
                    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
                    V_t = torch.from_numpy(V).float()
                    with torch.no_grad():
                        je = precision_energy_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                                      prior_mean_mode="stationary").cpu().numpy()
                        jr = precision_ratio_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                                     prior_mean_mode="stationary").cpu().numpy()
                    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                        continue
                    ks_je = ks_null(je[mask])
                    ks_jr = ks_null(jr[mask])
                    use_je = ks_je > ks_jr
                    sel = je if use_je else jr
                    auc = roc_auc_score(y[mask], sel[mask])
                    auc_je = roc_auc_score(y[mask], je[mask])
                    auc_jr = roc_auc_score(y[mask], jr[mask])
                    sname = "J*" if use_je else "R"
                    print("    %s+%s g=%s: %s=%.1f%% (J*=%.1f%% R=%.1f%%) rho=%.3f" %
                          (graph_type[:3], tmpl, gamma, sname, auc*100,
                           auc_je*100, auc_jr*100, rho), flush=True)
                except Exception as e:
                    pass

    print("  Reported: 81.3%  Best baseline: 71.6% (DiffGAD)")


def run_dgraph(device="cpu"):
    print("\n=== DGRAPH ===", flush=True)
    # Use cached eigenpairs
    cache_dir = "cache/dgraph_eigen"
    if not os.path.isdir(cache_dir):
        print("  No cached eigenpairs found, skipping")
        return

    data = load_data("dgraph")
    y = data.y.numpy()
    labeled_mask = (y >= 0) & (y <= 1)
    n = data.x.shape[0]
    d = data.x.shape[1]
    print("  n=%d d=%d labeled=%d" % (n, d, labeled_mask.sum()))

    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(data.x.float().numpy()).astype(np.float32)

    for subset in ["labeled"]:
        for k in [128, 256]:
            fname = "%s_original_k%d_lowrank.pt" % (subset, k)
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                continue
            print("  Loading %s..." % fname, flush=True)
            cache = torch.load(fpath, weights_only=False)
            V = cache["V"].numpy()
            lam_L = cache["lam_L"].numpy()

            idx = np.where(labeled_mask)[0]
            x_sub = x[idx]
            y_sub = y[idx]
            mask_sub = (y_sub >= 0) & (y_sub <= 1)

            for gamma in [0.5, 1.0, 2.0]:
                smoother = 1.0 / (gamma**2 + lam_L)
                tmpl = V @ (smoother[:, None] * (V.T @ x_sub))
                delta = x_sub - tmpl
                S = compute_spectral_energy(delta, V)
                n_sub = x_sub.shape[0]

                ml, rho, kappa = -np.inf, 0.5, 0.0
                for kk in KAPPAS:
                    r = solve_rho_newton(S, lam_L, kk, d)
                    q = Q_rho_eigenvalues(lam_L, r, kk)
                    m = marginal_log_likelihood_stationary(S, q, n_sub, d)
                    if m > ml:
                        ml, rho, kappa = m, r, kk

                q = Q_rho_eigenvalues(lam_L, rho, kappa)
                q_safe = np.maximum(q, 1e-12)
                delta_hat = V.T @ delta
                w = np.sqrt(q_safe)[:, None] * delta_hat
                je = np.sum((V @ w)**2, axis=1)
                tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
                jr = je / tot

                ks_je = ks_null(je[mask_sub])
                ks_jr = ks_null(jr[mask_sub])
                use_je = ks_je > ks_jr
                sel = je if use_je else jr
                sname = "J*" if use_je else "R"
                auc = roc_auc_score(y_sub[mask_sub], sel[mask_sub])
                auc_je = roc_auc_score(y_sub[mask_sub], je[mask_sub])
                auc_jr = roc_auc_score(y_sub[mask_sub], jr[mask_sub])

                print("    k=%d g=%s: %s=%.1f%% (J*=%.1f%% R=%.1f%%) rho=%.3f" %
                      (k, gamma, sname, auc*100, auc_je*100, auc_jr*100, rho),
                      flush=True)

    print("  Reported: 65.7%  Best baseline: 64.2% (DIF)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_enron(args.device)
    run_dgraph(args.device)
