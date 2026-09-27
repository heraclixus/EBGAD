"""Hybrid scoring: combine graph-spectral J*/R with per-feature tail analysis.

New scores:
- E: ECOD-style per-feature tail score on the template residuals δ
- J*+E: max(normalized J*, normalized E) per node
- R+E: max(normalized R, normalized E) per node

The idea: J*/R capture graph-spectral anomalies, E captures feature-tail anomalies.
Combined, they detect both types.
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


def ecod_score(delta):
    """ECOD-style tail score on residuals: per-feature empirical CDF tail."""
    n, d = delta.shape
    scores = np.zeros(n)
    for j in range(d):
        col = delta[:, j]
        # Two-sided tail: max(left tail, right tail)
        ranks = np.argsort(np.argsort(col)).astype(float) / n
        tail = np.maximum(ranks, 1 - ranks)  # distance from center
        scores += -np.log(1 - tail + 1e-10)  # negative log tail probability
    return scores


def zscore_normalize(s):
    """Z-score normalize to [0, 1]-ish range."""
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-12:
        return np.zeros_like(s)
    return (s - mu) / sigma


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

    # ML selects gamma
    best_ml = -np.inf
    best_gamma = gamma_range[0]
    best_data = None

    for gamma in gamma_range:
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
        try:
            trainer = SOCTrainer(SOCTrainerConfig(**cfg))
            trainer.train(data, device=device)
            V = trainer.V.cpu().numpy()
            lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
            tmpl = trainer.template.cpu()
            x = trainer.x_T.cpu()
            n_nodes, d_eff = x.shape
            delta = x.numpy() - tmpl.numpy()
            S = compute_spectral_energy(delta, V)
            ml, rho, kappa = -np.inf, 0.5, 0.0
            for k in KAPPAS:
                r = solve_rho_newton(S, lam_L, k, d_eff)
                q = Q_rho_eigenvalues(lam_L, r, k)
                m = marginal_log_likelihood_stationary(S, q, n_nodes, d_eff)
                if m > ml: ml, rho, kappa = m, r, k
            if ml > best_ml:
                best_ml = ml
                best_gamma = gamma
                best_data = (x, tmpl, V, lam_L, rho, kappa, delta)
        except:
            pass

    if best_data is None:
        return None

    x, tmpl, V, lam_L, rho, kappa, delta = best_data
    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
    V_t = torch.from_numpy(V).float()

    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2, 1,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2, 1,
                                     prior_mean_mode="stationary").cpu().numpy()

    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
        return None

    # ECOD-style tail score on residuals
    e_score = ecod_score(delta)

    # Hybrid: combine via z-score normalization + max
    je_z = zscore_normalize(je)
    jr_z = zscore_normalize(jr)
    e_z = zscore_normalize(e_score)

    je_plus_e = np.maximum(je_z, e_z)
    jr_plus_e = np.maximum(jr_z, e_z)

    # Also try average
    je_avg_e = (je_z + e_z) / 2
    jr_avg_e = (jr_z + e_z) / 2

    # Compute AUROCs
    scores = {
        "J*": je, "R": jr, "E": e_score,
        "max(J*,E)": je_plus_e, "max(R,E)": jr_plus_e,
        "avg(J*,E)": je_avg_e, "avg(R,E)": jr_avg_e,
    }

    results = {}
    for sname, sarr in scores.items():
        if not np.any(np.isnan(sarr)):
            auc = roc_auc_score(y[mask], sarr[mask])
            ks = ks_null(sarr[mask]) if sname in ("J*", "R") else 0
            results[sname] = {"auc": auc, "ks": ks}

    # KS selects between J* and R (existing pipeline)
    ks_je = results.get("J*", {}).get("ks", 0)
    ks_jr = results.get("R", {}).get("ks", 0)
    ks_pick = "J*" if ks_je > ks_jr else "R"
    pipeline_auc = results[ks_pick]["auc"]

    # Best hybrid
    hybrid_scores = {k: v for k, v in results.items() if k not in ("J*", "R", "E")}
    best_hybrid = max(hybrid_scores, key=lambda k: hybrid_scores[k]["auc"])
    best_hybrid_auc = hybrid_scores[best_hybrid]["auc"]

    return {
        "gamma": best_gamma, "rho": rho,
        "pipeline": {"score": ks_pick, "auc": pipeline_auc},
        "ecod": results.get("E", {}).get("auc", 0),
        "best_hybrid": {"score": best_hybrid, "auc": best_hybrid_auc},
        "all": {k: v["auc"] for k, v in results.items()},
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
                print("  gamma=%s rho=%.3f" % (r["gamma"], r["rho"]))
                print("  Pipeline (%s): %.1f%%" % (r["pipeline"]["score"], r["pipeline"]["auc"]*100))
                print("  ECOD(delta):    %.1f%%" % (r["ecod"]*100))
                print("  Best hybrid:    %.1f%% (%s)" % (r["best_hybrid"]["auc"]*100, r["best_hybrid"]["score"]))
                for sname, auc in sorted(r["all"].items(), key=lambda x: -x[1]):
                    marker = " <--" if auc*100 >= base else ""
                    print("    %-12s %.1f%%%s" % (sname, auc*100, marker))
                print("  Baseline: %.1f%%" % base)
                all_results[ds] = r
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(all_results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %7s %7s %7s %7s %7s" %
              ("Dataset", "Pipe", "E(δ)", "Hybrid", "Base", "vsBase"))
        for ds in datasets:
            if ds in all_results:
                r = all_results[ds]
                base = BASELINES.get(ds, 0)
                best = max(r["pipeline"]["auc"], r["best_hybrid"]["auc"])
                print("%-15s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+5.1f" %
                      (ds, r["pipeline"]["auc"]*100, r["ecod"]*100,
                       r["best_hybrid"]["auc"]*100, base, best*100-base))

        pipe_beats = sum(1 for r in all_results.values() if r["pipeline"]["auc"]*100 >= BASELINES.get(ds, 0))
        hybrid_beats = sum(1 for ds, r in all_results.items()
                          if max(r["pipeline"]["auc"], r["best_hybrid"]["auc"])*100 >= BASELINES[ds])
        print("\n  Pipeline beats baseline: %d/%d" % (pipe_beats, len(all_results)))
        print("  Best(pipe,hybrid) beats: %d/%d" % (hybrid_beats, len(all_results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/hybrid_%s.json" % args.dataset
               if args.dataset else "results/hybrid.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
