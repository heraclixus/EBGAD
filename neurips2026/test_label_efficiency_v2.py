"""Label efficiency ablation v2: full grid when labels available.

0% = our unsupervised pipeline (density-adjusted γ + ML + KS)
1-100% = search full grid {low,affinity} × {0.1,0.2,0.5,0.7,1.0,2.0,5.0} × PCA × {J*,R}
         select by AUROC on labeled validation subset
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
FULL_GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]
UNSUP_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # no 0.7
TEMPLATES = ["low", "affinity"]
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
    density = data.edge_index.shape[1] // 2 / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean()), density


def get_unsup_gamma_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in UNSUP_GAMMAS if g <= max_gamma]
    if not nearest:
        nearest = [UNSUP_GAMMAS[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


def run_config(data, tmpl, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type=tmpl, graph_type="original", gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    elif pca_dim and d <= pca_dim:
        return None
    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl_t = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl_t.numpy()
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
        je = precision_energy_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl_t, V_t, lam_Q, 2.0, 1.0,
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
    n_anom = int(y_labeled.sum())
    h, density = compute_graph_stats(data)

    # PCA candidates
    pca_candidates = [None]
    if d > 100:
        pca_candidates.append(64)
    if d > 32:
        pca_candidates.append(16)
    if d > 200:
        pca_candidates.append(128)

    print("  n=%d d=%d h=%.3f dens=%.1f anom=%d pca=%s" %
          (len(y), d, h, density, n_anom, pca_candidates), flush=True)

    # =============================================
    # Precompute ALL scores for full grid (for labeled selection)
    # =============================================
    full_scores = []  # list of (tag, je, jr)
    for tmpl in TEMPLATES:
        for gamma in FULL_GAMMAS:
            for pca in pca_candidates:
                tag = "%s g=%s p%s" % (tmpl[:3], gamma, pca or "no")
                try:
                    result = run_config(data, tmpl, gamma, pca, device)
                    if result is None:
                        continue
                    je, jr, ml = result
                    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                        continue
                    full_scores.append({"tag": tag, "je": je, "jr": jr, "ml": ml,
                                        "tmpl": tmpl, "gamma": gamma, "pca": pca})
                except:
                    pass
    print("  %d full-grid configs computed" % len(full_scores), flush=True)

    # =============================================
    # Unsupervised pipeline (0% labels)
    # =============================================
    unsup_gamma_range = get_unsup_gamma_range(h, density)
    # ML selects gamma within range (low template only)
    best_ml = -np.inf
    unsup_je, unsup_jr = None, None
    for entry in full_scores:
        if entry["tmpl"] == "low" and entry["gamma"] in unsup_gamma_range and entry["pca"] == (64 if d > 100 else None):
            if entry["ml"] > best_ml:
                best_ml = entry["ml"]
                unsup_je, unsup_jr = entry["je"], entry["jr"]
                unsup_tag_base = entry["tag"]

    if unsup_je is None:
        # Fallback
        for entry in full_scores:
            if entry["tmpl"] == "low" and entry["gamma"] == 1.0:
                unsup_je, unsup_jr = entry["je"], entry["jr"]
                unsup_tag_base = entry["tag"]
                break

    # KS selects J* vs R
    if unsup_je is not None:
        ks_je = ks_null(unsup_je[mask])
        ks_jr = ks_null(unsup_jr[mask])
        if ks_je > ks_jr:
            unsup_scores = unsup_je
            unsup_tag = "J* " + unsup_tag_base
        else:
            unsup_scores = unsup_jr
            unsup_tag = "R " + unsup_tag_base
        unsup_auc = roc_auc_score(y[mask], unsup_scores[mask])
    else:
        unsup_auc = 0
        unsup_tag = "none"

    results = {0.0: {"mean": unsup_auc, "std": 0.0, "tag": unsup_tag}}
    print("  0%%: %.1f%% (%s)" % (unsup_auc*100, unsup_tag), flush=True)

    # =============================================
    # Full-grid oracle (100% labels)
    # =============================================
    oracle_auc = 0
    oracle_tag = ""
    for entry in full_scores:
        for sname, sarr in [("J*", entry["je"]), ("R", entry["jr"])]:
            auc = roc_auc_score(y[mask], sarr[mask])
            if auc > oracle_auc:
                oracle_auc = auc
                oracle_tag = "%s %s" % (sname, entry["tag"])
    results[1.0] = {"mean": oracle_auc, "std": 0.0, "tag": oracle_tag}
    print("  100%%: %.1f%% (%s)" % (oracle_auc*100, oracle_tag), flush=True)

    # =============================================
    # Semi-supervised (1%, 5%, 10%, 50% labels)
    # =============================================
    # Build flat candidate list: (tag, scores_array)
    candidates = []
    for entry in full_scores:
        for sname, sarr in [("J*", entry["je"]), ("R", entry["jr"])]:
            candidates.append(("%s %s" % (sname, entry["tag"]), sarr))

    for frac in [0.01, 0.05, 0.10, 0.50]:
        if n_anom < 4:
            results[frac] = {"mean": 0, "std": 0, "tag": "too few"}
            continue
        try:
            splitter = StratifiedShuffleSplit(n_splits=N_SPLITS,
                                              test_size=1-frac,
                                              random_state=42)
        except:
            results[frac] = {"mean": 0, "std": 0, "tag": "split error"}
            continue

        test_aucs = []
        for val_local, test_local in splitter.split(labeled_idx, y_labeled):
            val_nodes = labeled_idx[val_local]
            test_nodes = labeled_idx[test_local]

            best_val = -1
            best_cand = None
            for tag, sarr in candidates:
                try:
                    va = roc_auc_score(y[val_nodes], sarr[val_nodes])
                    if va > best_val:
                        best_val = va
                        best_cand = (tag, sarr)
                except:
                    pass

            if best_cand:
                try:
                    ta = roc_auc_score(y[test_nodes], best_cand[1][test_nodes])
                    test_aucs.append(ta)
                except:
                    pass

        if test_aucs:
            results[frac] = {
                "mean": float(np.mean(test_aucs)),
                "std": float(np.std(test_aucs)),
                "tag": "val-split (%d cands)" % len(candidates),
            }
            print("  %.0f%%: %.1f%%+/-%.1f%%" %
                  (frac*100, np.mean(test_aucs)*100, np.std(test_aucs)*100), flush=True)

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
                print("  %5.0f%%: %5.1f%% +/- %4.1f%%  vsRep=%+.1f  vsBase=%+.1f" %
                      (frac*100, m, s, m-rep, m-base))

    os.makedirs("results", exist_ok=True)
    with open("results/label_eff_v2_%s.json" % ds, "w") as f:
        json.dump(r, f, indent=2, default=str)


if __name__ == "__main__":
    main()
