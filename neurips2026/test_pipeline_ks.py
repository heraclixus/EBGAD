"""Final unsupervised pipeline with KS null-deviation score selection.

1. Original graph, low template, gamma=1.0, PCA=64 if d>100
2. ML-optimize rho, kappa
3. Compute J* and R
4. KS null-deviation: pick score with higher KS against chi-squared null
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest, chi2

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


def ks_null_deviation(scores):
    """KS statistic of scores against fitted chi-squared null."""
    s = scores[scores > 0]
    if len(s) < 10:
        return 0.0
    mu, var = np.mean(s), np.var(s)
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k_est = max(0.01, 2 * mu**2 / var)
    a_est = var / (2 * mu)
    s_norm = s / a_est
    try:
        ks_stat, _ = kstest(s_norm, 'chi2', args=(k_est,))
        return float(ks_stat)
    except:
        return 0.0


def run_dataset(ds, device="cpu", modeled_subspace="legacy"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type="low", graph_type="original", gamma=1.0,
        modeled_subspace=modeled_subspace,
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
    lam_L = trainer.lam_L_model.cpu().numpy()
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

    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
        return None

    # KS null-deviation rule
    ks_je = ks_null_deviation(je[mask])
    ks_jr = ks_null_deviation(jr[mask])
    use_je = ks_je > ks_jr
    scores = je if use_je else jr
    score_name = "J*" if use_je else "R"

    auc_sel = roc_auc_score(y[mask], scores[mask])
    auc_je = roc_auc_score(y[mask], je[mask])
    auc_jr = roc_auc_score(y[mask], jr[mask])

    return {
        "auc_selected": auc_sel, "auc_je": auc_je, "auc_jr": auc_jr,
        "auc_oracle": max(auc_je, auc_jr),
        "score_name": score_name, "ks_je": ks_je, "ks_jr": ks_jr,
        "rho": best_rho, "kappa": best_kappa, "pca": pca_dim,
        "modeled_subspace": modeled_subspace,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--modeled-subspace", default="legacy",
                        choices=["legacy", "drop_nullspace"])
    parser.add_argument("--output-tag", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(REPORTED.keys())
    results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device, modeled_subspace=args.modeled_subspace)
            if r:
                rep = REPORTED.get(ds, 0)
                bname, bauc = BASELINES.get(ds, ("?", 0))
                print("  KS-selected: %.1f%% (%s, KS_J*=%.3f KS_R=%.3f)" %
                      (r["auc_selected"]*100, r["score_name"], r["ks_je"], r["ks_jr"]))
                print("  J*=%.1f%%  R=%.1f%%  Oracle=%.1f%%" %
                      (r["auc_je"]*100, r["auc_jr"]*100, r["auc_oracle"]*100))
                print("  rho=%.3f  kappa=%.2f  PCA=%s  subspace=%s" %
                      (r["rho"], r["kappa"], r["pca"], r["modeled_subspace"]))
                print("  vs Reported: %+.1fpp  vs %s: %+.1fpp" %
                      (r["auc_selected"]*100 - rep, bname, r["auc_selected"]*100 - bauc))
                results[ds] = r
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %7s %6s %6s %7s %7s %7s %5s" %
              ("Dataset", "KS-sel", "J*", "R", "Report", "vsRep", "vsBase", "Score"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                rep = REPORTED.get(ds, 0)
                _, bauc = BASELINES.get(ds, ("?", 0))
                print("%-15s %6.1f%% %5.1f%% %5.1f%% %6.1f%% %+6.1f %+6.1f  %s" %
                      (ds, r["auc_selected"]*100, r["auc_je"]*100, r["auc_jr"]*100,
                       rep, r["auc_selected"]*100 - rep,
                       r["auc_selected"]*100 - bauc, r["score_name"]))

        sel = np.mean([r["auc_selected"] for r in results.values()]) * 100
        orc = np.mean([r["auc_oracle"] for r in results.values()]) * 100
        rep_m = np.mean([REPORTED[ds] for ds in results])
        base_m = np.mean([BASELINES[ds][1] for ds in results])
        beats = sum(1 for ds, r in results.items()
                    if r["auc_selected"]*100 >= BASELINES[ds][1])
        print("\n  Mean KS-selected: %.1f%%" % sel)
        print("  Mean oracle:      %.1f%%" % orc)
        print("  Mean reported:    %.1f%%" % rep_m)
        print("  Mean baseline:    %.1f%%" % base_m)
        print("  Beats baseline:   %d/%d" % (beats, len(results)))

    os.makedirs("results", exist_ok=True)
    suffix_parts = []
    if args.dataset:
        suffix_parts.append(args.dataset)
    if args.modeled_subspace != "legacy":
        suffix_parts.append(args.modeled_subspace)
    if args.output_tag:
        suffix_parts.append(args.output_tag)
    suffix = "_".join(suffix_parts)
    outfile = ("results/pipeline_ks_%s.json" % suffix
               if suffix else "results/pipeline_ks.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
