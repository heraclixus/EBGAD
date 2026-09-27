"""Compute data for ablation tables:
  Table 3: Gamma-invariance (4 datasets x 5 Gamma values)
  Table 4: KS score selection (all datasets, J* vs R vs KS-selected)
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
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    marginal_log_likelihood_dynamic, time_dependent_precision,
    compute_spectral_energy, solve_rho_newton, solve_rho_dynamic,
    gamma_jacobian_correction,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


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


def score_nodes(delta, V, q, Gamma_t):
    """Spectral-projection score: only the k computed eigenmodes are scored."""
    if Gamma_t is None or Gamma_t > 1e5:
        qt = np.maximum(q, 1e-12)
    else:
        qt = time_dependent_precision(q, Gamma_t)
    delta_hat = V.T @ delta                           # (k, d)
    w = np.sqrt(qt)[:, None] * delta_hat              # (k, d)
    je = np.sum((V @ w)**2, axis=1)                   # (n,)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot
    return je, jr


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d_raw = data.x.shape[1]
    pca_dim = 64 if d_raw > 100 else None

    h, density = compute_graph_stats(data)
    density_range = get_density_range(h, density)

    # Precompute eigenpairs/template for each gamma
    gamma_cache = {}
    for gamma in density_range:
        cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
                   normalize_mode="zscore", prior_mean_mode="stationary",
                   laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
                   epochs=0, normalize_features=True,
                   template_type="low", graph_type="original", gamma=gamma)
        if pca_dim and d_raw > pca_dim:
            cfg.update(dict(use_encoder=True, encoder_type="pca",
                           encoder_hid_dim=pca_dim, encoder_num_layers=1,
                           encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                           encoder_alpha=1.0, encoder_weight_decay=0.0))
        trainer = SOCTrainer(SOCTrainerConfig(**cfg))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        x = trainer.x_T.cpu().numpy()
        tmpl = trainer.template.cpu().numpy()
        n, d = x.shape
        delta = x - tmpl
        S = compute_spectral_energy(delta, V)
        gamma_cache[gamma] = {"V": V, "lam_L": lam_L, "delta": delta, "S": S, "n": n, "d": d}

    # === Table 3: Gamma-invariance ===
    gamma_inv = {}
    for Gt in [0.5, 1.0, 5.0, 50.0, None]:  # None = stationary
        is_stat = (Gt is None)
        best_ml, best_rho, best_kappa, best_gamma = -np.inf, 0.5, 0.0, None

        for gamma in density_range:
            gc = gamma_cache[gamma]
            V, lam_L, delta, S, n, d = gc["V"], gc["lam_L"], gc["delta"], gc["S"], gc["n"], gc["d"]
            for kk in KAPPAS:
                if is_stat:
                    rho = solve_rho_newton(S, lam_L, kk, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = marginal_log_likelihood_stationary(S, q, n, d)
                else:
                    rho = solve_rho_dynamic(S, lam_L, kk, Gt, d)
                    q = Q_rho_eigenvalues(lam_L, rho, kk)
                    ml = marginal_log_likelihood_dynamic(S, q, Gt, n, d)
                if ml > best_ml:
                    best_ml, best_rho, best_kappa, best_gamma = ml, rho, kk, gamma

        gc = gamma_cache[best_gamma]
        q = Q_rho_eigenvalues(gc["lam_L"], best_rho, best_kappa)
        je, jr = score_nodes(gc["delta"], gc["V"], q, Gt)
        auc_je = roc_auc_score(y[mask], je[mask])
        auc_jr = roc_auc_score(y[mask], jr[mask])
        ks_je = ks_null(je[mask])
        ks_jr = ks_null(jr[mask])
        auc_ks = auc_je if ks_je > ks_jr else auc_jr

        Gt_str = "inf" if is_stat else str(Gt)
        gamma_inv[Gt_str] = {
            "rho": round(best_rho, 4), "kappa": round(best_kappa, 1),
            "gamma": best_gamma, "auc": round(auc_ks * 100, 1),
        }

    # === Table 4: KS score selection ===
    # Use equilibrium (Gamma=inf) ML-selected params
    stat = gamma_inv["inf"]
    gc = gamma_cache[stat["gamma"]]
    q = Q_rho_eigenvalues(gc["lam_L"], stat["rho"], stat["kappa"])
    je, jr = score_nodes(gc["delta"], gc["V"], q, None)
    auc_je = round(roc_auc_score(y[mask], je[mask]) * 100, 1)
    auc_jr = round(roc_auc_score(y[mask], jr[mask]) * 100, 1)
    ks_je = ks_null(je[mask])
    ks_jr = ks_null(jr[mask])
    ks_pick = "J*" if ks_je > ks_jr else "R"
    auc_ks = auc_je if ks_pick == "J*" else auc_jr

    ks_result = {
        "J*": auc_je, "R": auc_jr,
        "KS_pick": ks_pick, "KS_AUC": auc_ks,
        "better": "J*" if auc_je >= auc_jr else "R",
        "correct": (ks_pick == "J*" and auc_je >= auc_jr) or (ks_pick == "R" and auc_jr >= auc_je),
    }

    return gamma_inv, ks_result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog",
                     "facebook", "acm", "elliptic", "elliptic_plus_plus", "t_finance"]
    else:
        datasets = [args.dataset]

    all_gamma_inv = {}
    all_ks = {}

    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        try:
            gamma_inv, ks_result = run_dataset(ds, args.device)
            all_gamma_inv[ds] = gamma_inv
            all_ks[ds] = ks_result

            print("  Gamma-invariance:")
            for Gt, v in sorted(gamma_inv.items(), key=lambda x: float(x[0]) if x[0] != "inf" else 1e6):
                print(f"    G={Gt:>5s}: rho={v['rho']:.4f} kap={v['kappa']:.1f} g={v['gamma']} AUC={v['auc']:.1f}%")
            print(f"  KS selection: J*={ks_result['J*']:.1f}% R={ks_result['R']:.1f}% "
                  f"KS={ks_result['KS_pick']}={ks_result['KS_AUC']:.1f}% "
                  f"correct={ks_result['correct']}")
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_tables.json", "w") as f:
        json.dump({"gamma_invariance": all_gamma_inv, "ks_selection": all_ks}, f, indent=2)

    # Print LaTeX-ready tables
    if all_ks:
        print("\n=== KS Score Selection (Table 4) ===")
        print(f"{'Dataset':<18s} {'J*':>6s} {'R':>6s} {'KS':>4s} {'AUC':>6s} {'OK?':>4s}")
        for ds in datasets:
            if ds in all_ks:
                k = all_ks[ds]
                print(f"{ds:<18s} {k['J*']:5.1f}% {k['R']:5.1f}% {k['KS_pick']:>4s} {k['KS_AUC']:5.1f}% "
                      f"{'yes' if k['correct'] else 'NO'}")


if __name__ == "__main__":
    main()
