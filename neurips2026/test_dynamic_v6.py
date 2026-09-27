"""Dynamic framework v6: Binary KS selection.

Two candidates:
  A: ML-best stationary (Gamma=inf) -- proven pipeline
  B: ML-best dynamic (best ML among all finite Gamma)
KS picks between A and B.

Each candidate has ML-optimized (rho, kappa) so neither is degenerate.
KS just decides: does the dynamics help or not?

Also tests a variant: for candidate B, try each Gamma independently
and pick the Gamma whose ML-optimized config has highest KS.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os
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
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ALL_GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
GAMMA_T_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 70.6, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 63.1,
}
CURRENT = {
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
    if Gamma_t is None or Gamma_t > 1e5:
        qt = np.maximum(q, 1e-12)
    else:
        qt = time_dependent_precision(q, Gamma_t)
    delta_hat = V.T @ delta
    w = np.sqrt(qt)[:, None] * delta_hat
    je = np.sum((V @ w)**2, axis=1)
    tot = np.maximum(np.sum(delta**2, axis=1), 1e-12)
    jr = je / tot
    return je, jr


def eval_config(delta, V, q, Gamma_t, y, mask):
    """Score and evaluate a config. Returns dict with scores and metrics."""
    je, jr = score_nodes(delta, V, q, Gamma_t)
    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
        return None
    auc_je = roc_auc_score(y[mask], je[mask])
    auc_jr = roc_auc_score(y[mask], jr[mask])
    ks_je = ks_null(je[mask])
    ks_jr = ks_null(jr[mask])
    ks_pick = "J*" if ks_je > ks_jr else "R"
    auc_ks = auc_je if ks_pick == "J*" else auc_jr
    ks_best = max(ks_je, ks_jr)
    return {"auc_je": auc_je, "auc_jr": auc_jr, "auc_ks": auc_ks,
            "ks_je": ks_je, "ks_jr": ks_jr, "ks_best": ks_best,
            "ks_pick": ks_pick, "je": je, "jr": jr}


def run_dataset(ds, device="cpu"):
    print("\n=== %s ===" % ds, flush=True)
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d_raw = data.x.shape[1]
    pca_dim = 64 if d_raw > 100 else None

    h, density = compute_graph_stats(data)
    density_range = get_density_range(h, density)
    print("  h=%.3f density=%.1f gamma_range=%s" % (h, density, density_range), flush=True)

    # Precompute
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

    # ===== Candidate A: ML-best stationary =====
    candA = {"ml": -np.inf}
    for gamma in density_range:
        gc = gamma_cache[gamma]
        V, lam_L, delta, S, n, d = gc["V"], gc["lam_L"], gc["delta"], gc["S"], gc["n"], gc["d"]
        for kk in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kk, d)
            q = Q_rho_eigenvalues(lam_L, rho, kk)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > candA["ml"]:
                candA = {"ml": ml, "gamma": gamma, "rho": rho, "kappa": kk,
                         "V": V, "lam_L": lam_L, "delta": delta, "q": q, "n": n, "d": d}

    evA = eval_config(candA["delta"], candA["V"], candA["q"], None, y, mask)
    print("  Cand A (stationary): g=%s rho=%.3f kap=%.1f → %s=%.1f%% KS=%.4f" %
          (candA["gamma"], candA["rho"], candA["kappa"],
           evA["ks_pick"], evA["auc_ks"]*100, evA["ks_best"]), flush=True)

    # ===== Candidate B: for each (gamma, Gamma), ML-optimize (rho, kappa) =====
    # Strategy 1: Single best-ML dynamic config
    candB_ml = {"ml": -np.inf}
    # Strategy 2: Best-KS among all ML-optimized dynamic configs
    candB_ks = {"ks": -1}
    # Strategy 3: Per-Gamma profile: for each Gamma, best ML across (gamma, kappa)
    per_gamma_best = {}

    for gamma in density_range:
        gc = gamma_cache[gamma]
        V, lam_L, delta, S, n, d = gc["V"], gc["lam_L"], gc["delta"], gc["S"], gc["n"], gc["d"]

        for Gt in GAMMA_T_GRID:
            best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
            for kk in KAPPAS:
                rho = solve_rho_dynamic(S, lam_L, kk, Gt, d)
                q = Q_rho_eigenvalues(lam_L, rho, kk)
                ml = marginal_log_likelihood_dynamic(S, q, Gt, n, d)
                if ml > best_ml:
                    best_ml, best_rho, best_kappa = ml, rho, kk

            q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
            ev = eval_config(delta, V, q, Gt, y, mask)
            if ev is None:
                continue

            # Strategy 1: best ML among dynamic
            if best_ml > candB_ml["ml"]:
                candB_ml = {"ml": best_ml, "gamma": gamma, "Gt": Gt,
                            "rho": best_rho, "kappa": best_kappa, "ev": ev}

            # Strategy 2: best KS among dynamic
            if ev["ks_best"] > candB_ks.get("ks", -1):
                candB_ks = {"ks": ev["ks_best"], "gamma": gamma, "Gt": Gt,
                            "rho": best_rho, "kappa": best_kappa, "ev": ev, "ml": best_ml}

            # Strategy 3: per-Gamma profile
            if Gt not in per_gamma_best or best_ml > per_gamma_best[Gt]["ml"]:
                per_gamma_best[Gt] = {"ml": best_ml, "gamma": gamma,
                                       "rho": best_rho, "kappa": best_kappa, "ev": ev}

            print("    g=%s G=%s: %s=%.1f%% KS=%.4f rho=%.3f kap=%.1f ml=%.0f" %
                  (gamma, Gt, ev["ks_pick"], ev["auc_ks"]*100, ev["ks_best"],
                   best_rho, best_kappa, best_ml), flush=True)

    # ===== Selection strategies =====
    base = BASELINES.get(ds, 0)
    curr = CURRENT.get(ds, 0)

    print("\n  --- Selection strategies ---", flush=True)

    # S1: Binary KS between A (stat ML-best) and B (dyn ML-best)
    if candB_ml.get("ev"):
        evB1 = candB_ml["ev"]
        pick1 = "B" if evB1["ks_best"] > evA["ks_best"] else "A"
        auc1 = evB1["auc_ks"] if pick1 == "B" else evA["auc_ks"]
        print("  S1 (A vs B-ml):  %.1f%% pick=%s  [A:KS=%.4f B:KS=%.4f] (G=%s rho=%.3f)" %
              (auc1*100, pick1, evA["ks_best"], evB1["ks_best"],
               candB_ml["Gt"], candB_ml["rho"]), flush=True)
    else:
        auc1 = evA["auc_ks"]
        pick1 = "A"
        print("  S1: no dynamic candidate", flush=True)

    # S2: Binary KS between A and B (dyn KS-best, but only from ML-optimized configs)
    if candB_ks.get("ev"):
        evB2 = candB_ks["ev"]
        pick2 = "B" if evB2["ks_best"] > evA["ks_best"] else "A"
        auc2 = evB2["auc_ks"] if pick2 == "B" else evA["auc_ks"]
        print("  S2 (A vs B-ks):  %.1f%% pick=%s  [A:KS=%.4f B:KS=%.4f] (G=%s rho=%.3f)" %
              (auc2*100, pick2, evA["ks_best"], evB2["ks_best"],
               candB_ks["Gt"], candB_ks["rho"]), flush=True)
    else:
        auc2 = evA["auc_ks"]
        pick2 = "A"

    # S3: Profile KS -- for each Gamma, get ML-best config, then KS picks Gamma
    # Binary: compare stat vs best-KS-among-per-Gamma-profiles
    best_profile_ks = {"ks": -1}
    for Gt, info in per_gamma_best.items():
        ev = info["ev"]
        if ev["ks_best"] > best_profile_ks.get("ks", -1):
            best_profile_ks = {"ks": ev["ks_best"], "Gt": Gt, "ev": ev,
                               "gamma": info["gamma"], "rho": info["rho"], "kappa": info["kappa"]}

    if best_profile_ks.get("ev"):
        evB3 = best_profile_ks["ev"]
        pick3 = "B" if evB3["ks_best"] > evA["ks_best"] else "A"
        auc3 = evB3["auc_ks"] if pick3 == "B" else evA["auc_ks"]
        print("  S3 (A vs profile): %.1f%% pick=%s  [B: G=%s rho=%.3f kap=%.1f]" %
              (auc3*100, pick3, best_profile_ks["Gt"],
               best_profile_ks["rho"], best_profile_ks["kappa"]), flush=True)
    else:
        auc3 = evA["auc_ks"]
        pick3 = "A"

    # S4: Multi-candidate KS -- A plus one candidate per Gamma, KS picks
    candidates = [("inf", evA)]
    for Gt in sorted(per_gamma_best.keys()):
        candidates.append((str(Gt), per_gamma_best[Gt]["ev"]))
    ks_winner = max(candidates, key=lambda c: c[1]["ks_best"])
    auc4 = ks_winner[1]["auc_ks"]
    print("  S4 (multi-KS):   %.1f%% G=%s %s KS=%.4f" %
          (auc4*100, ks_winner[0], ks_winner[1]["ks_pick"], ks_winner[1]["ks_best"]), flush=True)

    # Oracle
    all_evs = [("inf", evA)]
    for Gt, info in per_gamma_best.items():
        all_evs.append((str(Gt), info["ev"]))
    oracle = max(all_evs, key=lambda c: max(c[1]["auc_je"], c[1]["auc_jr"]))
    oracle_auc = max(oracle[1]["auc_je"], oracle[1]["auc_jr"])

    print("  ---")
    print("  Stationary ML: %.1f%%" % (evA["auc_ks"]*100))
    print("  Oracle:        %.1f%%  G=%s" % (oracle_auc*100, oracle[0]))
    print("  Baseline:      %.1f%%  Current: %.1f%%" % (base, curr))

    return {
        "stat": evA["auc_ks"]*100,
        "s1": auc1*100, "s1_pick": pick1,
        "s2": auc2*100, "s2_pick": pick2,
        "s3": auc3*100, "s3_pick": pick3,
        "s4": auc4*100,
        "oracle": oracle_auc*100,
    }


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

    summary = {}
    for ds in datasets:
        try:
            summary[ds] = run_dataset(ds, args.device)
        except Exception as e:
            import traceback
            traceback.print_exc()

    if len(summary) > 1:
        print("\n" + "="*80)
        print("%-18s %6s %6s %6s %6s %6s %6s %5s" % (
            "Dataset", "Stat", "S1", "S2", "S3", "S4", "Oracle", "Base"))
        wins = {s: 0 for s in ["s1", "s2", "s3", "s4"]}
        losses = {s: 0 for s in ["s1", "s2", "s3", "s4"]}
        for ds in datasets:
            if ds in summary:
                s = summary[ds]
                print("%-18s %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %4.1f%%" % (
                    ds, s["stat"], s["s1"], s["s2"], s["s3"], s["s4"],
                    s["oracle"], BASELINES.get(ds, 0)))
                for strat in ["s1", "s2", "s3", "s4"]:
                    if s[strat] > s["stat"] + 0.5: wins[strat] += 1
                    if s[strat] < s["stat"] - 0.5: losses[strat] += 1
        print("---")
        for strat in ["s1", "s2", "s3", "s4"]:
            print("  %s: %d wins, %d losses vs stationary" % (strat, wins[strat], losses[strat]))


if __name__ == "__main__":
    main()
