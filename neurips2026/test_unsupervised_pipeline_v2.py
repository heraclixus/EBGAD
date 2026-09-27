"""Fully unsupervised EB-GAD pipeline.

Pipeline:
1. Use original graph Laplacian (default)
2. ML-optimize rho, kappa, gamma (with and without PCA)
3. Select PCA by higher ML
4. Compute J* and R
5. Pick score with higher Gini coefficient
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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

ORACLE_AUROCS = {
    "enron": 81.3, "weibo": 95.8, "reddit": 62.6, "amazon": 78.1,
    "yelpchi": 72.0, "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}

PCA_DIMS = [16, 32, 64, 128]


def gini_coefficient(scores):
    """Compute Gini coefficient of a score distribution."""
    sorted_s = np.sort(scores)
    n = len(sorted_s)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_s) / (n * np.sum(sorted_s) + 1e-12))
                 - (n + 1) / n)


def run_config(data, template_type, gamma, pca_dim, graph_type="original", device="cpu"):
    """Run one config: ML-optimize rho/kappa, return J*, R, and ML."""
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type=template_type, graph_type=graph_type, gamma=gamma,
    )
    if pca_dim is not None and data.x.shape[1] > pca_dim:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=pca_dim, encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))
    elif pca_dim is not None:
        return None  # PCA dim >= feature dim, skip

    try:
        trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl = trainer.template.cpu()
        x = trainer.x_T.cpu()
        x_np = x.numpy()
        n, d = x_np.shape
        delta = x_np - tmpl.numpy()
        S = compute_spectral_energy(delta, V)

        best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa

        if best_rho < 0.01:
            return None

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t = torch.from_numpy(V).float()

        with torch.no_grad():
            je = precision_energy_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x, tmpl, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()

        return {
            "je": je, "jr": jr, "ml": best_ml,
            "rho": best_rho, "kappa": best_kappa,
        }
    except Exception as e:
        print(f"    Error: {str(e)[:80]}", flush=True)
        return None


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]

    # Step 1: PCA candidates
    pca_candidates = [None]
    if d > 32:
        pca_candidates.append(16)
    if d > 100:
        pca_candidates.append(64)
    if d > 200:
        pca_candidates.append(128)

    # Step 2: Run all configs, ML-optimize rho/kappa within each
    # Collect ALL score arrays, then pick by Gini
    candidates = []

    for pca_dim in pca_candidates:
        for graph in ["original", "affinity"]:
            for tmpl in ["low", "affinity", "high"]:
                for gamma in [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]:
                    result = run_config(data, tmpl, gamma, pca_dim, graph, device)
                    if result is not None:
                        je, jr = result["je"], result["jr"]
                        if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                            continue
                        gini_je = gini_coefficient(je[mask])
                        gini_jr = gini_coefficient(jr[mask])
                        auc_je = roc_auc_score(y[mask], je[mask])
                        auc_jr = roc_auc_score(y[mask], jr[mask])

                        pca_tag = f"p{pca_dim}" if pca_dim else "noP"
                        tag = f"{graph[:3]}+{tmpl}"
                        # J* candidate
                        candidates.append({
                            "scores": je, "score_name": "J*",
                            "gini": gini_je, "auc": auc_je,
                            "tmpl": tmpl, "graph": graph, "gamma": gamma,
                            "pca": pca_dim,
                            "rho": result["rho"], "kappa": result["kappa"],
                            "ml": result["ml"],
                        })
                        # R candidate
                        candidates.append({
                            "scores": jr, "score_name": "R",
                            "gini": gini_jr, "auc": auc_jr,
                            "tmpl": tmpl, "graph": graph, "gamma": gamma,
                            "pca": pca_dim,
                            "rho": result["rho"], "kappa": result["kappa"],
                            "ml": result["ml"],
                        })

                        print(f"    {tag:12s} g={gamma:<3} {pca_tag:>5s}: "
                              f"ML={result['ml']:.0f} rho={result['rho']:.3f} "
                              f"G(J*)={gini_je:.3f} G(R)={gini_jr:.3f} "
                              f"J*={auc_je*100:.1f}% R={auc_jr*100:.1f}%",
                              flush=True)

    if not candidates:
        return None

    # Step 3: Pick candidate with highest Gini (unsupervised)
    best = max(candidates, key=lambda c: c["gini"])
    selected_scores = best["scores"]

    # Oracle: best AUROC across all candidates
    oracle_auc = max(c["auc"] for c in candidates)
    # Also track ML-selected (for comparison)
    ml_best = max(candidates, key=lambda c: c["ml"])

    return {
        "auc_selected": best["auc"],
        "auc_oracle": oracle_auc,
        "auc_ml_selected": ml_best["auc"],
        "score_name": best["score_name"],
        "gini": best["gini"],
        "tmpl": best["tmpl"],
        "graph": best["graph"],
        "pca": best.get("pca"),
        "gamma": best["gamma"],
        "rho": best["rho"],
        "kappa": best["kappa"],
        "ml": best["ml"],
        "ml_tmpl": ml_best["tmpl"],
        "ml_graph": ml_best["graph"],
        "ml_gamma": ml_best["gamma"],
        "ml_score": ml_best["score_name"],
        "n_candidates": len(candidates),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else [
        "enron", "weibo", "reddit", "amazon", "yelpchi",
        "blogcatalog", "facebook", "acm",
        "elliptic", "elliptic_plus_plus", "t_finance",
    ]

    results = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        r = run_dataset(ds, args.device)
        if r:
            rep = ORACLE_AUROCS.get(ds, 0)
            print(f"  Gini-selected: {r['auc_selected']*100:.1f}% "
                  f"({r['score_name']}, {r['graph'][:3]}+{r['tmpl']}, "
                  f"g={r['gamma']}, Gini={r['gini']:.3f})")
            print(f"  ML-selected:   {r['auc_ml_selected']*100:.1f}% "
                  f"({r['ml_score']}, {r['ml_graph'][:3]}+{r['ml_tmpl']}, "
                  f"g={r['ml_gamma']})")
            print(f"  Oracle:        {r['auc_oracle']*100:.1f}%  "
                  f"Reported: {rep}%")
            print(f"  rho={r['rho']:.3f}  kappa={r['kappa']:.2f}  "
                  f"PCA={r['pca']}  ({r['n_candidates']} candidates)")
            print(f"  vs Reported: {r['auc_selected']*100 - rep:+.1f}pp")
            results[ds] = r

    os.makedirs("results", exist_ok=True)
    outfile = ("results/unsup_pipeline_v2_%s.json" % args.dataset
               if args.dataset else "results/unsup_pipeline_v2.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        print(f"{'Dataset':15s}  {'Gini':>6s} {'ML':>6s} {'Oracle':>6s} "
              f"{'Reported':>7s} {'vsRep':>6s} {'Config':>20s}")
        for ds in datasets:
            if ds in results:
                r = results[ds]
                rep = ORACLE_AUROCS.get(ds, 0)
                pcatag = f"p{r['pca']}" if r['pca'] else "noP"
                cfg = f"{r['score_name']} {r['graph'][:3]}+{r['tmpl']} g={r['gamma']} {pcatag}"
                print(f"{ds:15s}  {r['auc_selected']*100:5.1f}% "
                      f"{r['auc_ml_selected']*100:5.1f}% "
                      f"{r['auc_oracle']*100:5.1f}% "
                      f"{rep:6.1f}% {r['auc_selected']*100 - rep:+5.1f} "
                      f"{cfg}")

        sel = np.mean([r["auc_selected"] for r in results.values()]) * 100
        ml = np.mean([r["auc_ml_selected"] for r in results.values()]) * 100
        orc = np.mean([r["auc_oracle"] for r in results.values()]) * 100
        rep = np.mean([ORACLE_AUROCS.get(ds, 0) for ds in results.keys()])
        print(f"\n  Mean Gini-selected: {sel:.1f}%")
        print(f"  Mean ML-selected:   {ml:.1f}%")
        print(f"  Mean oracle:        {orc:.1f}%")
        print(f"  Mean reported:      {rep:.1f}%")
        print(f"  Gini vs oracle:     {sel - orc:+.1f}pp")
        print(f"  Gini vs reported:   {sel - rep:+.1f}pp")
        print(f"  ML vs reported:     {ml - rep:+.1f}pp")


if __name__ == "__main__":
    main()
