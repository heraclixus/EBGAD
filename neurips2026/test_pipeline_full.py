"""Full unsupervised pipeline with PCA search, C score, and rho fallback.

Improvements over previous pipeline:
1. Search PCA ∈ {None, 16, 64, 128} (based on d), KS selects best
2. Add C score alongside J* and R
3. If selected γ gives rho < 0.01, fall back to next-best γ
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
from soc.soc_anomaly import (precision_energy_anomaly, precision_ratio_anomaly,
                              control_energy_anomaly, ce_ratio_anomaly)
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
    "weibo": ("ANOM.", 95.0), "reddit": ("CoLA", 60.3), "amazon": ("DIF", 78.1),
    "yelpchi": ("LOF", 57.2), "blogcatalog": ("TAM", 82.5), "facebook": ("TAM", 91.4),
    "acm": ("TAM", 88.8), "elliptic": ("CoLA", 65.3),
    "elliptic_plus_plus": ("CoLA", 62.9), "t_finance": ("ECOD", 83.5),
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
    src, dst = edge_index[0, idx], edge_index[1, idx]
    cos = torch.nn.functional.cosine_similarity(x[src], x[dst], dim=1)
    return float(cos.mean())


def run_config(data, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
               alpha=2.0, T=1.0, normalize_mode="zscore",
               prior_mean_mode="stationary", laplacian_variant="sym",
               d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
               template_type="low", graph_type="original", gamma=gamma)
    if pca_dim is not None and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    elif pca_dim is not None:
        return None  # pca_dim >= d, skip

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

    scores = {}
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()
        try:
            ce = control_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                        lam_penalty=50.0).cpu().numpy()
            cr = ce_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                  lam_penalty=50.0).cpu().numpy()
            scores["C"] = ce
            scores["C_R"] = cr
        except:
            pass

    scores["J*"] = je
    scores["R"] = jr
    return scores, best_ml, rho, kappa


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]

    # Edge homophily for gamma selection
    homophily = compute_edge_homophily(data)

    # Gamma selection
    if homophily < 0.10:
        gamma_candidates = [0.1, 0.2, 0.5]
    else:
        gamma_candidates = [1.0]

    # PCA candidates
    pca_candidates = [None]
    if d > 32:
        pca_candidates.append(16)
    if d > 100:
        pca_candidates.append(64)
    if d > 200:
        pca_candidates.append(128)

    print("  h=%.3f d=%d -> gamma=%s pca=%s" %
          (homophily, d, gamma_candidates, pca_candidates), flush=True)

    # For gamma selection: ML within range (using default PCA=64 or None)
    default_pca = 64 if d > 100 else None
    best_ml = -np.inf
    best_gamma = gamma_candidates[0]

    for gamma in gamma_candidates:
        try:
            scores, ml, rho, kappa = run_config(data, gamma, default_pca, device)
            if ml > best_ml and rho >= 0.01:  # rho fallback: skip rho≈0
                best_ml = ml
                best_gamma = gamma
            print("    gamma=%s (default PCA): ml=%.0f rho=%.3f" %
                  (gamma, ml, rho), flush=True)
        except:
            pass

    # If all gammas had rho < 0.01, use the one with highest ML regardless
    if best_ml == -np.inf:
        for gamma in gamma_candidates:
            try:
                scores, ml, rho, kappa = run_config(data, gamma, default_pca, device)
                if ml > best_ml:
                    best_ml = ml
                    best_gamma = gamma
            except:
                pass

    print("  Selected gamma=%s" % best_gamma, flush=True)

    # Now search PCA at selected gamma, KS selects best (PCA, score)
    candidates = []
    for pca_dim in pca_candidates:
        try:
            scores, ml, rho, kappa = run_config(data, best_gamma, pca_dim, device)
            if scores is None:
                continue

            pca_tag = "p%d" % pca_dim if pca_dim else "noP"
            for sname, sarr in scores.items():
                if np.any(np.isnan(sarr)):
                    continue
                ks = ks_null(sarr[mask])
                auc = roc_auc_score(y[mask], sarr[mask])
                candidates.append({
                    "tag": "%s %s" % (sname, pca_tag),
                    "score_name": sname, "pca": pca_dim,
                    "scores": sarr, "ks": ks, "auc": auc,
                    "rho": rho,
                })

            score_strs = " ".join("%s=%.1f%%" % (s, roc_auc_score(y[mask], a[mask])*100)
                                  for s, a in scores.items() if not np.any(np.isnan(a)))
            print("    pca=%s: %s rho=%.3f" % (pca_tag, score_strs, rho), flush=True)
        except Exception as e:
            print("    pca=%s: ERROR %s" % (pca_dim, str(e)[:60]))

    if not candidates:
        return None

    # KS selects best candidate
    best = max(candidates, key=lambda c: c["ks"])
    oracle = max(candidates, key=lambda c: c["auc"])

    return {
        "auc_selected": best["auc"],
        "auc_oracle": oracle["auc"],
        "tag": best["tag"],
        "oracle_tag": oracle["tag"],
        "gamma": best_gamma, "homophily": homophily,
        "ks": best["ks"], "rho": best["rho"],
        "n_candidates": len(candidates),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                rep = REPORTED.get(ds, 0)
                bname, bauc = BASELINES.get(ds, ("?", 0))
                print("  KS-selected: %.1f%% (%s, KS=%.3f)" %
                      (r["auc_selected"]*100, r["tag"], r["ks"]))
                print("  Oracle:      %.1f%% (%s)" %
                      (r["auc_oracle"]*100, r["oracle_tag"]))
                print("  vs Reported: %+.1fpp  vs %s: %+.1fpp" %
                      (r["auc_selected"]*100 - rep, bname, r["auc_selected"]*100 - bauc))
                results[ds] = r
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %7s %7s %7s %7s %7s  %s" %
              ("Dataset", "KS-sel", "Oracle", "Report", "vsRep", "vsBase", "Config"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                rep = REPORTED.get(ds, 0)
                _, bauc = BASELINES.get(ds, ("?", 0))
                print("%-15s %6.1f%% %6.1f%% %6.1f%% %+6.1f %+6.1f  %s g=%s" %
                      (ds, r["auc_selected"]*100, r["auc_oracle"]*100, rep,
                       r["auc_selected"]*100 - rep, r["auc_selected"]*100 - bauc,
                       r["tag"], r["gamma"]))

        sel_m = np.mean([r["auc_selected"] for r in results.values()]) * 100
        beats = sum(1 for ds, r in results.items()
                   if r["auc_selected"]*100 >= BASELINES[ds][1])
        print("\n  Mean: %.1f%%  Beats baseline: %d/%d" % (sel_m, beats, len(results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/pipeline_full_%s.json" % args.dataset
               if args.dataset else "results/pipeline_full.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
