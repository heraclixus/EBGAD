"""Label efficiency ablation: unsupervised → semi-supervised → oracle.

For each dataset, at each label fraction:
1. Sample k% of labeled nodes as "validation" set
2. Run all configs (density-adjusted γ range × {J*, R})
3. Select config by AUROC on the validation set
4. Evaluate on remaining nodes (test set)
5. Repeat 10 times with different random splits

Label fractions: 0% (unsupervised pipeline), 1%, 5%, 10%, 50%, 100% (oracle)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
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
LABEL_FRACS = [0.0, 0.01, 0.05, 0.10, 0.50, 1.0]
N_SPLITS = 10

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


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    n_edges = edge_index.shape[1] // 2
    density = n_edges / n
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
    if not nearest:
        nearest = [GAMMA_GRID[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


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
    return je, jr, best_ml


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    labeled_idx = np.where(mask)[0]
    y_labeled = y[labeled_idx]
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None
    n_anom = int(y_labeled.sum())

    h, density = compute_graph_stats(data)
    gamma_range = get_gamma_range(h, density)
    print("  h=%.3f dens=%.1f gamma_range=%s n_labeled=%d n_anom=%d" %
          (h, density, gamma_range, len(labeled_idx), n_anom), flush=True)

    # Precompute all scores (shared across splits)
    all_scores = {}  # key: (gamma, score_name) -> score array
    for gamma in gamma_range:
        try:
            je, jr, ml = run_gamma(data, gamma, pca_dim, device)
            if not np.any(np.isnan(je)):
                all_scores[(gamma, "J*")] = je
            if not np.any(np.isnan(jr)):
                all_scores[(gamma, "R")] = jr
        except:
            pass
    print("  %d score candidates computed" % len(all_scores), flush=True)

    if not all_scores:
        return None

    # Precompute KS for unsupervised selection
    ks_vals = {k: ks_null(v[mask]) for k, v in all_scores.items()}

    results = {}
    for frac in LABEL_FRACS:
        if frac == 0.0:
            # Unsupervised: KS selects
            best_key = max(ks_vals, key=ks_vals.get)
            scores = all_scores[best_key]
            auc = roc_auc_score(y[mask], scores[mask])
            results[frac] = {
                "mean": auc, "std": 0.0, "tag": "%s g=%s" % (best_key[1], best_key[0]),
            }
            print("  frac=0.00: %.1f%% (%s) [unsupervised KS]" %
                  (auc*100, results[frac]["tag"]), flush=True)

        elif frac == 1.0:
            # Oracle: best on all labels
            best_auc = 0
            best_key = None
            for key, scores in all_scores.items():
                auc = roc_auc_score(y[mask], scores[mask])
                if auc > best_auc:
                    best_auc = auc
                    best_key = key
            results[frac] = {
                "mean": best_auc, "std": 0.0, "tag": "%s g=%s" % (best_key[1], best_key[0]),
            }
            print("  frac=1.00: %.1f%% (%s) [oracle]" %
                  (best_auc*100, results[frac]["tag"]), flush=True)

        else:
            # Semi-supervised: split labels into val/test
            if n_anom < 4:
                results[frac] = {"mean": 0, "std": 0, "tag": "too few anomalies"}
                continue

            val_size = max(0.01, frac)
            try:
                splitter = StratifiedShuffleSplit(n_splits=N_SPLITS,
                                                  test_size=1-val_size,
                                                  random_state=42)
            except:
                results[frac] = {"mean": 0, "std": 0, "tag": "split error"}
                continue

            test_aucs = []
            for val_local, test_local in splitter.split(labeled_idx, y_labeled):
                val_nodes = labeled_idx[val_local]
                test_nodes = labeled_idx[test_local]

                # Select by val AUROC
                best_val_auc = -1
                best_key = None
                for key, scores in all_scores.items():
                    try:
                        va = roc_auc_score(y[val_nodes], scores[val_nodes])
                        if va > best_val_auc:
                            best_val_auc = va
                            best_key = key
                    except:
                        pass

                if best_key:
                    try:
                        ta = roc_auc_score(y[test_nodes], all_scores[best_key][test_nodes])
                        test_aucs.append(ta)
                    except:
                        pass

            if test_aucs:
                results[frac] = {
                    "mean": float(np.mean(test_aucs)),
                    "std": float(np.std(test_aucs)),
                    "tag": "val-split",
                }
                print("  frac=%.2f: %.1f%%+/-%.1f%% [%d splits]" %
                      (frac, np.mean(test_aucs)*100, np.std(test_aucs)*100, len(test_aucs)),
                      flush=True)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    ds = args.dataset
    print("=== %s ===" % ds, flush=True)
    r = run_dataset(ds, args.device)

    if r:
        rep = REPORTED.get(ds, 0)
        base = BASELINES.get(ds, 0)
        print("\n  Summary:")
        for frac in LABEL_FRACS:
            if frac in r:
                m = r[frac]["mean"] * 100
                s = r[frac]["std"] * 100
                tag = r[frac]["tag"]
                if s > 0:
                    print("  %5.0f%%: %5.1f +/- %4.1f  vsBase=%+.1f  (%s)" %
                          (frac*100, m, s, m-base, tag))
                else:
                    print("  %5.0f%%: %5.1f%%           vsBase=%+.1f  (%s)" %
                          (frac*100, m, m-base, tag))

    os.makedirs("results", exist_ok=True)
    with open("results/label_eff_%s.json" % ds, "w") as f:
        json.dump(r, f, indent=2, default=str)


if __name__ == "__main__":
    main()
