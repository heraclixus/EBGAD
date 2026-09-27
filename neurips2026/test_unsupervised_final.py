"""Final unsupervised EB-GAD pipeline.

1. Original graph Laplacian
2. PCA = 64 if d > 100
3. ML-optimize rho, kappa, gamma
4. hf_frac < 0.05 -> R, else J*
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
GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]

ORACLE_AUROCS = {
    "weibo": 95.8, "reddit": 62.6, "amazon": 78.1,
    "yelpchi": 72.0, "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}

BEST_BASELINES = {
    "weibo": ("ANOM.", 95.0), "reddit": ("CoLA", 60.3), "amazon": ("DIF", 78.1),
    "yelpchi": ("LOF", 57.2), "blogcatalog": ("TAM", 82.5), "facebook": ("TAM", 91.4),
    "acm": ("TAM", 88.8), "elliptic": ("CoLA", 65.3),
    "elliptic_plus_plus": ("CoLA", 62.9), "t_finance": ("ECOD", 83.5),
}


def run_reference_config(data, pca_dim, device):
    """Run reference config (gamma=0.5) to determine hf_frac for score selection."""
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type="low", graph_type="original", gamma=0.5,
    )
    if pca_dim is not None and data.x.shape[1] > pca_dim:
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
    delta = x.numpy() - tmpl.numpy()
    S = compute_spectral_energy(delta, V)
    n_modes = len(S)
    k10 = max(1, n_modes // 10)
    hf_frac = float(S[-k10:].sum() / (S.sum() + 1e-12))
    return hf_frac


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    # Stage 1: determine score type from reference config (gamma=0.5)
    hf_frac = run_reference_config(data, pca_dim, device)
    use_R = hf_frac < 0.05
    score_name = "R" if use_R else "J*"
    print("  hf_frac=%.4f -> use %s" % (hf_frac, score_name), flush=True)

    # Stage 2: ML-optimize gamma, rho, kappa for the selected score
    best_ml = -np.inf
    best_result = None
    best_gamma = None

    for gamma in GAMMAS:
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

        try:
            trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
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
                if m > ml:
                    ml, rho, kappa = m, r, k

            if ml > best_ml:
                best_ml = ml
                best_gamma = gamma
                best_result = {
                    "V": V, "lam_L": lam_L, "tmpl": tmpl, "x": x,
                    "S": S, "rho": rho, "kappa": kappa,
                    "n": n_nodes, "d": d_eff,
                }
                print("    gamma=%s: ML=%.0f rho=%.3f kappa=%.2f (new best)" %
                      (gamma, ml, rho, kappa), flush=True)
            else:
                print("    gamma=%s: ML=%.0f rho=%.3f kappa=%.2f" %
                      (gamma, ml, rho, kappa), flush=True)
        except Exception as e:
            print("    gamma=%s: ERROR %s" % (gamma, str(e)[:60]), flush=True)

    if best_result is None:
        return None

    r = best_result
    lam_Q = torch.from_numpy(Q_rho_eigenvalues(r["lam_L"], r["rho"], r["kappa"])).float()
    V_t = torch.from_numpy(r["V"]).float()

    with torch.no_grad():
        je = precision_energy_anomaly(r["x"], r["tmpl"], V_t, lam_Q, 2.0, 1.0,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(r["x"], r["tmpl"], V_t, lam_Q, 2.0, 1.0,
                                     prior_mean_mode="stationary").cpu().numpy()

    if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
        return None

    scores = jr if use_R else je
    auc_selected = roc_auc_score(y[mask], scores[mask])
    auc_je = roc_auc_score(y[mask], je[mask])
    auc_jr = roc_auc_score(y[mask], jr[mask])

    return {
        "auc_selected": auc_selected,
        "auc_je": auc_je, "auc_jr": auc_jr,
        "auc_oracle": max(auc_je, auc_jr),
        "score_name": score_name, "hf_frac": hf_frac,
        "gamma": best_gamma, "rho": r["rho"], "kappa": r["kappa"],
        "pca": pca_dim, "ml": best_ml,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(ORACLE_AUROCS.keys())
    results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        r = run_dataset(ds, args.device)
        if r:
            rep = ORACLE_AUROCS.get(ds, 0)
            base_name, base_auc = BEST_BASELINES.get(ds, ("?", 0))
            print("  Selected: %.1f%% (%s, hf=%.4f)" %
                  (r["auc_selected"]*100, r["score_name"], r["hf_frac"]))
            print("  J*=%.1f%%  R=%.1f%%  Oracle=%.1f%%" %
                  (r["auc_je"]*100, r["auc_jr"]*100, r["auc_oracle"]*100))
            print("  gamma=%s  rho=%.3f  kappa=%.2f  PCA=%s" %
                  (r["gamma"], r["rho"], r["kappa"], r["pca"]))
            print("  vs Reported: %+.1fpp  vs %s: %+.1fpp" %
                  (r["auc_selected"]*100 - rep, base_name, r["auc_selected"]*100 - base_auc))
            results[ds] = r

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %7s %6s %6s %7s %7s %7s %5s" %
              ("Dataset", "Unsup", "J*", "R", "Report", "vsRep", "vsBase", "Score"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                rep = ORACLE_AUROCS.get(ds, 0)
                _, base_auc = BEST_BASELINES.get(ds, ("?", 0))
                print("%-15s %6.1f%% %5.1f%% %5.1f%% %6.1f%% %+6.1f %+6.1f  %s" %
                      (ds, r["auc_selected"]*100, r["auc_je"]*100, r["auc_jr"]*100,
                       rep, r["auc_selected"]*100 - rep,
                       r["auc_selected"]*100 - base_auc, r["score_name"]))

        sel_mean = np.mean([r["auc_selected"] for r in results.values()]) * 100
        rep_mean = np.mean([ORACLE_AUROCS[ds] for ds in results])
        base_mean = np.mean([BEST_BASELINES[ds][1] for ds in results])
        beats = sum(1 for ds, r in results.items()
                    if r["auc_selected"]*100 >= BEST_BASELINES[ds][1])
        print("\n  Mean unsup: %.1f%%  Mean reported: %.1f%%  Mean baseline: %.1f%%" %
              (sel_mean, rep_mean, base_mean))
        print("  Beats best baseline: %d/%d" % (beats, len(results)))

    os.makedirs("results", exist_ok=True)
    outfile = ("results/unsup_final_%s.json" % args.dataset
               if args.dataset else "results/unsup_final.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
