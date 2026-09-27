"""Test the fully unsupervised pipeline.

For each dataset:
1. Run Algorithm 1 under 4 configs (stationary/bridge × PCA/noPCA)
2. Select the config with highest marginal likelihood (no labels)
3. Compute max(J*, R) or C scores
4. Evaluate AUROC
5. Compare against oracle-selected (best AUROC) result

This verifies whether ML-based selection matches oracle selection.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from eval_utils import mask_labeled
from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

DATASETS = [
    "enron", "weibo", "reddit", "amazon", "yelpchi",
    "blogcatalog", "facebook", "acm",
    "elliptic", "elliptic_plus_plus", "t_finance",
]

# Best configs from sweeps (oracle-selected by AUROC)
ORACLE_AUROCS = {
    "enron": 81.3, "weibo": 95.8, "reddit": 62.6, "amazon": 78.1,
    "yelpchi": 72.0, "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
}

# Templates and gammas to try (from sweep best configs)
TEMPLATES = ["zero", "low", "affinity"]
GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0]
GRAPHS = ["original", "affinity"]
NORMS = ["zscore"]
PRIOR_MODES = ["stationary"]
KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def evaluate_config(data, ds, template, graph, gamma, use_pca, pca_dim,
                    device="cpu"):
    """Evaluate one config: optimize rho/kappa via ML, compute scores."""
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, template_type=template, graph_type=graph,
        gamma=gamma, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0,
        normalize_features=True,
    )
    if use_pca:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=pca_dim, encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))

    try:
        trainer_cfg = SOCTrainerConfig(**cfg_kwargs)
        trainer = SOCTrainer(trainer_cfg)
        trainer.train(data, device=device)

        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        template_vec = trainer.template.cpu().numpy()

        x = data.x.float()
        if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
            x = trainer.normalizer(x)
        if hasattr(trainer, 'encoder') and trainer.encoder is not None:
            with torch.no_grad():
                x = trainer.encoder(x.to(device),
                                    data.edge_index.to(device)).cpu()

        x_np = x.numpy()
        n, d = x_np.shape
        delta = x_np - template_vec
        S = compute_spectral_energy(delta, V)

        # Optimize rho for each kappa via Newton, find best ML
        best_ml = -np.inf
        best_rho = 0.5
        best_kappa = 0.0
        for kappa in KAPPAS:
            rho_star = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho_star, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml = ml
                best_rho = rho_star
                best_kappa = kappa

        # Compute J* and R scores at best (rho, kappa)
        lam_Q = torch.from_numpy(
            Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t = torch.from_numpy(V).float()
        tmpl_t = torch.from_numpy(template_vec).float()
        x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x_np).float()

        with torch.no_grad():
            je = precision_energy_anomaly(
                x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(
                x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                prior_mean_mode="stationary").cpu().numpy()

        # max(J*, R) per node
        max_score = np.maximum(
            (je - je.mean()) / (je.std() + 1e-8),
            (jr - jr.mean()) / (jr.std() + 1e-8),
        )

        y = data.y.numpy()
        mask = (y >= 0) & (y <= 1)

        auc_je = roc_auc_score(y[mask], je[mask])
        auc_jr = roc_auc_score(y[mask], jr[mask])
        auc_max = roc_auc_score(y[mask], max_score[mask])

        return {
            "ml": float(best_ml),
            "rho": float(best_rho),
            "kappa": float(best_kappa),
            "auc_je": float(auc_je),
            "auc_jr": float(auc_jr),
            "auc_max": float(auc_max),
            "template": template,
            "graph": graph,
            "gamma": gamma,
            "pca": pca_dim if use_pca else None,
        }
    except Exception as e:
        return {"error": str(e)[:80]}


def run_dataset(ds, device="cpu"):
    """Run full unsupervised pipeline for one dataset."""
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    n, d = data.x.shape

    # Determine PCA dims to try
    pca_configs = [(False, 0)]
    if d > 50:
        for pd in [32, 64, 128]:
            if pd < d:
                pca_configs.append((True, pd))

    all_results = []
    for template in TEMPLATES:
        for graph in GRAPHS:
            for gamma in GAMMAS:
                for use_pca, pca_dim in pca_configs:
                    result = evaluate_config(
                        data, ds, template, graph, gamma,
                        use_pca, pca_dim, device)
                    if "error" not in result:
                        all_results.append(result)

    if not all_results:
        return None

    # ML-selected config (fully unsupervised)
    ml_best = max(all_results, key=lambda r: r["ml"])

    # Oracle: best AUROC across all configs and scoring modes
    oracle_best_auc = max(
        max(r["auc_je"], r["auc_jr"]) for r in all_results)

    return {
        "ml_selected": {
            "auc_je": ml_best["auc_je"],
            "auc_jr": ml_best["auc_jr"],
            "auc_max": ml_best["auc_max"],
            "ml": ml_best["ml"],
            "config": f"t={ml_best['template']} g={ml_best['graph']} "
                      f"gamma={ml_best['gamma']} pca={ml_best['pca']} "
                      f"rho={ml_best['rho']:.2f} kappa={ml_best['kappa']}",
        },
        "oracle_best_auc": oracle_best_auc,
        "n_configs_evaluated": len(all_results),
    }


def main(device="cpu"):
    out_file = "results/unsupervised_pipeline_test.json"
    results = {}

    for ds in DATASETS:
        print(f"=== {ds} ===", flush=True)
        t0 = time.time()
        result = run_dataset(ds, device)
        elapsed = time.time() - t0

        if result:
            ml = result["ml_selected"]
            oracle = result["oracle_best_auc"]
            reported = ORACLE_AUROCS.get(ds, 0)

            print(f"  ML-selected:  J*={ml['auc_je']*100:.1f}%  "
                  f"R={ml['auc_jr']*100:.1f}%  "
                  f"max={ml['auc_max']*100:.1f}%")
            print(f"  Oracle:       {oracle*100:.1f}%")
            print(f"  Reported:     {reported}%")
            print(f"  Config: {ml['config']}")
            print(f"  ({result['n_configs_evaluated']} configs, "
                  f"{elapsed:.0f}s)", flush=True)

            results[ds] = result

            with open(out_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Dataset':15s}  {'ML-max':>7s}  {'Oracle':>7s}  "
          f"{'Reported':>8s}  {'Gap':>5s}")
    for ds in DATASETS:
        if ds in results:
            ml_max = results[ds]["ml_selected"]["auc_max"] * 100
            oracle = results[ds]["oracle_best_auc"] * 100
            reported = ORACLE_AUROCS.get(ds, 0)
            gap = ml_max - reported
            print(f"{ds:15s}  {ml_max:6.1f}%  {oracle:6.1f}%  "
                  f"{reported:7.1f}%  {gap:+5.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.device)
