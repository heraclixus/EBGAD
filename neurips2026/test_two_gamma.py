"""Minimal pipeline: 2 gammas (0.5 and 1.0), low template, original graph.

Only 4 candidates: {g=0.5, g=1.0} × {J*, R}.
Test multiple selection criteria on this tiny grid.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

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
BASELINES = {
    "weibo": 95.0, "reddit": 60.3, "amazon": 78.1, "yelpchi": 57.2,
    "blogcatalog": 82.5, "facebook": 91.4, "acm": 88.8,
    "elliptic": 65.3, "elliptic_plus_plus": 62.9, "t_finance": 83.5,
}
REPORTED = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1, "yelpchi": 72.0,
    "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}


def gini(s):
    s = np.sort(s)
    n = len(s)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) / (n * np.sum(s) + 1e-12)) - (n + 1) / n)


def trimmed_gini(s, frac=0.99):
    s = np.sort(s)
    k = max(1, int(len(s) * frac))
    return gini(s[:k])


def run_gamma(data, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type="low", graph_type="original", gamma=gamma,
    )
    if pca_dim is not None and d > pca_dim:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=pca_dim, encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))
    trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl.numpy()
    S = compute_spectral_energy(delta, V)

    best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
    for kappa in KAPPAS:
        rho = solve_rho_newton(S, lam_L, kappa, d_eff)
        q = Q_rho_eigenvalues(lam_L, rho, kappa)
        ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
        if ml > best_ml:
            best_ml, best_rho, best_kappa = ml, rho, kappa

    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
    V_t = torch.from_numpy(V).float()
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()
    return {"je": je, "jr": jr, "ml": best_ml, "rho": best_rho}


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    candidates = []
    for gamma in [0.5, 1.0]:
        try:
            r = run_gamma(data, gamma, pca_dim, device)
        except Exception as e:
            print("    g=%s: ERROR %s" % (gamma, str(e)[:60]), flush=True)
            continue

        if np.any(np.isnan(r["je"])) or np.any(np.isnan(r["jr"])):
            continue

        for sname, scores in [("J*", r["je"]), ("R", r["jr"])]:
            s = scores[mask]
            auc = roc_auc_score(y[mask], scores[mask])
            g = gini(s)
            tg = trimmed_gini(s)
            sk = float(skew(s))
            p95 = float(np.percentile(s, 95) / (np.percentile(s, 50) + 1e-12))
            candidates.append({
                "tag": "%s g=%s" % (sname, gamma),
                "auc": auc, "gini": g, "tgini": tg,
                "skew": sk, "p95_p50": p95,
                "ml": r["ml"], "rho": r["rho"],
                "scores": scores,
            })

        print("    g=%s: J*=%.1f%% R=%.1f%% ml=%.0f rho=%.3f" %
              (gamma, roc_auc_score(y[mask], r["je"][mask])*100,
               roc_auc_score(y[mask], r["jr"][mask])*100,
               r["ml"], r["rho"]), flush=True)

    if not candidates:
        return None

    oracle = max(candidates, key=lambda c: c["auc"])

    # Test criteria
    criteria = {
        "gini": max(candidates, key=lambda c: c["gini"]),
        "tgini": max(candidates, key=lambda c: c["tgini"]),
        "skew": max(candidates, key=lambda c: c["skew"]),
        "p95_p50": max(candidates, key=lambda c: c["p95_p50"]),
        "ml": max(candidates, key=lambda c: c["ml"]),
    }

    # Rank product (ML × tgini)
    from scipy.stats import rankdata
    ml_r = rankdata([-c["ml"] for c in candidates])
    tg_r = rankdata([-c["tgini"] for c in candidates])
    rp = ml_r * tg_r
    criteria["rank_prod"] = candidates[np.argmin(rp)]

    return {"oracle": oracle, "criteria": criteria, "n": len(candidates)}


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
        r = run_dataset(ds, args.device)
        if r:
            rep = REPORTED.get(ds, 0)
            base = BASELINES.get(ds, 0)
            print("  Oracle: %.1f%% (%s)  Reported: %.1f%%" %
                  (r["oracle"]["auc"]*100, r["oracle"]["tag"], rep))
            for cname, cval in r["criteria"].items():
                gap = (cval["auc"] - r["oracle"]["auc"]) * 100
                marker = " ***" if abs(gap) < 3 else ""
                print("  %-12s %.1f%% (%s) gap=%+.1f%s" %
                      (cname, cval["auc"]*100, cval["tag"], gap, marker))
            all_results[ds] = {
                "oracle_auc": r["oracle"]["auc"],
                "oracle_tag": r["oracle"]["tag"],
            }
            for cname, cval in r["criteria"].items():
                all_results[ds][cname + "_auc"] = cval["auc"]
                all_results[ds][cname + "_tag"] = cval["tag"]

    if len(all_results) > 1:
        print("\n=== SUMMARY ===")
        crit_names = ["gini", "tgini", "skew", "p95_p50", "ml", "rank_prod"]
        print("%-15s %6s" % ("Dataset", "Oracle"), end="")
        for c in crit_names:
            print(" %8s" % c, end="")
        print()
        for ds in datasets:
            if ds not in all_results:
                continue
            r = all_results[ds]
            print("%-15s %5.1f%%" % (ds, r["oracle_auc"]*100), end="")
            for c in crit_names:
                auc = r.get(c+"_auc", 0) * 100
                gap = auc - r["oracle_auc"]*100
                mark = "*" if abs(gap) < 3 else " "
                print(" %6.1f%%%s" % (auc, mark), end="")
            print()

        print("\nWithin 3pp of oracle:")
        for c in crit_names:
            count = sum(1 for ds in all_results
                       if abs(all_results[ds].get(c+"_auc",0) - all_results[ds]["oracle_auc"]) < 0.03)
            print("  %-12s %d/%d" % (c, count, len(all_results)))

    os.makedirs("results", exist_ok=True)
    outfile = "results/two_gamma_%s.json" % args.dataset if args.dataset else "results/two_gamma.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
