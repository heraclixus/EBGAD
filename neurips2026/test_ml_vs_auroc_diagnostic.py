"""Diagnostic: ML vs AUROC correlation per config.

For a single dataset, compute ML and AUROC for every config,
then analyze whether ML selects configs with good AUROC.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from eval_utils import mask_labeled
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

TEMPLATES = ["zero", "low", "affinity"]
GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0]
GRAPHS = ["original", "affinity"]
KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def run_diagnostic(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    n, d = data.x.shape

    pca_configs = [(False, 0)]
    if d > 50:
        for pd in [32, 64, 128]:
            if pd < d:
                pca_configs.append((True, pd))

    results = []

    for template in TEMPLATES:
        for graph in GRAPHS:
            for gamma in GAMMAS:
                for use_pca, pca_dim in pca_configs:
                    cfg_kwargs = dict(
                        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
                        alpha=2.0, T=1.0, template_type=template,
                        graph_type=graph, gamma=gamma,
                        normalize_mode="zscore",
                        prior_mean_mode="stationary",
                        laplacian_variant="sym",
                        d_hidden=64, n_hidden_layers=2, epochs=0,
                        normalize_features=True,
                    )
                    if use_pca:
                        cfg_kwargs.update(dict(
                            use_encoder=True, encoder_type="pca",
                            encoder_hid_dim=pca_dim, encoder_num_layers=1,
                            encoder_dropout=0.0, encoder_lr=0.0,
                            encoder_epochs=0, encoder_alpha=1.0,
                            encoder_weight_decay=0.0,
                        ))

                    try:
                        trainer_cfg = SOCTrainerConfig(**cfg_kwargs)
                        trainer = SOCTrainer(trainer_cfg)
                        trainer.train(data, device=device)

                        V = trainer.V.cpu().numpy()
                        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
                        tmpl = trainer.template.cpu().numpy()
                        x = data.x.float()
                        if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
                            x = trainer.normalizer(x)
                        if hasattr(trainer, 'encoder') and trainer.encoder is not None:
                            with torch.no_grad():
                                x = trainer.encoder(x.to(device),
                                                    data.edge_index.to(device)).cpu()
                        x_np = x.numpy()
                        nd = x_np.shape[1]
                        delta = x_np - tmpl
                        S = compute_spectral_energy(delta, V)

                        # Optimize rho for each kappa
                        best_ml = -np.inf
                        best_rho = 0.5
                        best_kappa = 0.0
                        for kappa in KAPPAS:
                            rho_star = solve_rho_newton(S, lam_L, kappa, nd)
                            q = Q_rho_eigenvalues(lam_L, rho_star, kappa)
                            ml = marginal_log_likelihood_stationary(S, q, n, nd)
                            if ml > best_ml:
                                best_ml = ml
                                best_rho = rho_star
                                best_kappa = kappa

                        # Compute scores
                        lam_Q = torch.from_numpy(
                            Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
                        V_t = torch.from_numpy(V).float()
                        tmpl_t = torch.from_numpy(tmpl).float()
                        x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x_np).float()

                        with torch.no_grad():
                            je = precision_energy_anomaly(
                                x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                prior_mean_mode="stationary").cpu().numpy()
                            jr = precision_ratio_anomaly(
                                x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                prior_mean_mode="stationary").cpu().numpy()

                        y = data.y.numpy()
                        mask = (y >= 0) & (y <= 1)
                        auc_je = roc_auc_score(y[mask], je[mask])
                        auc_jr = roc_auc_score(y[mask], jr[mask])

                        tag = f"t={template} g={graph} gam={gamma} pca={pca_dim if use_pca else 'no'}"
                        results.append({
                            "tag": tag, "ml": best_ml,
                            "auc_je": auc_je, "auc_jr": auc_jr,
                            "best_auc": max(auc_je, auc_jr),
                            "rho": best_rho, "kappa": best_kappa,
                        })
                    except Exception as e:
                        pass

    # Analysis
    mls = np.array([r["ml"] for r in results])
    aucs_je = np.array([r["auc_je"] for r in results])
    aucs_jr = np.array([r["auc_jr"] for r in results])
    aucs_best = np.array([r["best_auc"] for r in results])

    corr_je, _ = spearmanr(mls, aucs_je)
    corr_jr, _ = spearmanr(mls, aucs_jr)
    corr_best, _ = spearmanr(mls, aucs_best)

    ml_idx = np.argmax(mls)
    oracle_je_idx = np.argmax(aucs_je)
    oracle_jr_idx = np.argmax(aucs_jr)
    oracle_best_idx = np.argmax(aucs_best)

    print(f"\n=== {ds} ({len(results)} configs) ===")
    print(f"Spearman correlation: ML vs J*={corr_je:.3f}  ML vs R={corr_jr:.3f}  ML vs best={corr_best:.3f}")
    print(f"ML-selected:  J*={aucs_je[ml_idx]*100:.1f}%  R={aucs_jr[ml_idx]*100:.1f}%  best={aucs_best[ml_idx]*100:.1f}%")
    print(f"  config: {results[ml_idx]['tag']}  rho={results[ml_idx]['rho']:.2f}  kappa={results[ml_idx]['kappa']}")
    print(f"Oracle J*:    {aucs_je[oracle_je_idx]*100:.1f}%  config: {results[oracle_je_idx]['tag']}")
    print(f"Oracle R:     {aucs_jr[oracle_jr_idx]*100:.1f}%  config: {results[oracle_jr_idx]['tag']}")
    print(f"Oracle best:  {aucs_best[oracle_best_idx]*100:.1f}%  config: {results[oracle_best_idx]['tag']}")

    # Top 5 by ML vs top 5 by AUROC
    ml_order = np.argsort(-mls)
    auc_order = np.argsort(-aucs_best)
    print(f"\nTop 5 by ML:")
    for i in ml_order[:5]:
        print(f"  ML={mls[i]:.0f}  J*={aucs_je[i]*100:.1f}%  R={aucs_jr[i]*100:.1f}%  {results[i]['tag']}")
    print(f"Top 5 by AUROC:")
    for i in auc_order[:5]:
        print(f"  ML={mls[i]:.0f}  J*={aucs_je[i]*100:.1f}%  R={aucs_jr[i]*100:.1f}%  {results[i]['tag']}")

    return {
        "corr_je": corr_je, "corr_jr": corr_jr, "corr_best": corr_best,
        "ml_selected_je": aucs_je[ml_idx], "ml_selected_jr": aucs_jr[ml_idx],
        "oracle_je": aucs_je[oracle_je_idx], "oracle_jr": aucs_jr[oracle_jr_idx],
        "oracle_best": aucs_best[oracle_best_idx],
        "ml_config": results[ml_idx]["tag"],
        "oracle_config": results[oracle_best_idx]["tag"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_diagnostic(args.dataset, args.device)
    with open(f"results/ml_diagnostic_{args.dataset}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
