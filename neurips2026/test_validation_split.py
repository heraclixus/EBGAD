"""Test: Validation split for discrete config selection.

1. ML optimizes rho/kappa (unsupervised, no labels)
2. Val AUROC selects discrete config (scoring mode, PCA)
3. Report AUROC on held-out test set

Multiple random splits to measure variance.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

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

ORACLE_AUROCS = {
    "enron": 81.3, "weibo": 95.8, "reddit": 62.6, "amazon": 78.1,
    "yelpchi": 72.0, "blogcatalog": 78.8, "facebook": 96.4, "acm": 93.5,
    "elliptic": 83.6, "elliptic_plus_plus": 83.7, "t_finance": 85.8,
    "dgraph": 65.7,
}

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Discrete configs to evaluate
CONFIGS = [
    {"name": "low_nopca", "template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"name": "low_pca64", "template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": 64},
    {"name": "aff_nopca", "template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"name": "aff_pca64", "template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": 64},
    {"name": "low_aff_g", "template_type": "low", "graph_type": "affinity", "gamma": 0.5, "pca": None},
    {"name": "low_g01", "template_type": "low", "graph_type": "original", "gamma": 0.1, "pca": None},
]


def build_and_score(data, cfg_spec, device="cpu"):
    """Build trainer, optimize rho/kappa via ML, return per-node scores."""
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, template_type=cfg_spec["template_type"],
        graph_type=cfg_spec["graph_type"], gamma=cfg_spec["gamma"],
        normalize_mode="zscore", prior_mean_mode="stationary",
        laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
        epochs=0, normalize_features=True,
    )
    if cfg_spec["pca"] is not None and data.x.shape[1] > cfg_spec["pca"]:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=cfg_spec["pca"], encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))
    elif cfg_spec["pca"] is not None:
        return None  # skip if PCA dim >= feature dim

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
        n, d = x_np.shape
        delta = x_np - tmpl
        S = compute_spectral_energy(delta, V)

        # ML-optimize rho/kappa
        best_ml = -np.inf
        best_rho, best_kappa = 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa

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

        return {"je": je, "jr": jr, "ml": best_ml,
                "rho": best_rho, "kappa": best_kappa}
    except Exception as e:
        return None


def run_dataset(ds, n_splits=10, val_ratio=0.5, device="cpu"):
    """Run validation-split evaluation for one dataset."""
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    labeled_idx = np.where(mask)[0]
    y_labeled = y[labeled_idx]

    n_anom = y_labeled.sum()
    n_total = len(y_labeled)
    print(f"  {n_total} labeled nodes, {n_anom} anomalies "
          f"({n_anom/n_total*100:.1f}%)", flush=True)

    if n_anom < 4:
        print(f"  Too few anomalies for split, skipping", flush=True)
        return None

    # Compute all config scores once (scores are the same regardless of split)
    all_scores = {}
    for cfg in CONFIGS:
        result = build_and_score(data, cfg, device)
        if result is not None:
            all_scores[cfg["name"]] = result

    if not all_scores:
        return None

    # Multiple random val/test splits
    splitter = StratifiedShuffleSplit(n_splits=n_splits,
                                      test_size=val_ratio,
                                      random_state=42)

    split_results = []
    for split_idx, (test_idx_local, val_idx_local) in enumerate(
            splitter.split(labeled_idx, y_labeled)):
        val_nodes = labeled_idx[val_idx_local]
        test_nodes = labeled_idx[test_idx_local]

        # For each config, compute val AUROC (J* and R separately)
        best_val_auc = -1
        best_cfg_name = None
        best_score_type = None

        for cfg_name, scores in all_scores.items():
            for score_type, score_arr in [("je", scores["je"]),
                                           ("jr", scores["jr"])]:
                try:
                    val_auc = roc_auc_score(y[val_nodes], score_arr[val_nodes])
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_cfg_name = cfg_name
                        best_score_type = score_type
                except ValueError:
                    pass

        # Evaluate selected config on test set
        if best_cfg_name:
            test_scores = all_scores[best_cfg_name][best_score_type]
            try:
                test_auc = roc_auc_score(y[test_nodes], test_scores[test_nodes])
            except ValueError:
                test_auc = 0.5

            # Also compute oracle test AUROC (best config selected by test set)
            oracle_test_auc = -1
            for cfg_name, scores in all_scores.items():
                for st in ["je", "jr"]:
                    try:
                        ta = roc_auc_score(y[test_nodes], scores[st][test_nodes])
                        oracle_test_auc = max(oracle_test_auc, ta)
                    except ValueError:
                        pass

            split_results.append({
                "val_auc": best_val_auc,
                "test_auc": test_auc,
                "oracle_test_auc": oracle_test_auc,
                "selected_cfg": best_cfg_name,
                "selected_score": best_score_type,
            })

    if not split_results:
        return None

    test_aucs = [r["test_auc"] for r in split_results]
    oracle_aucs = [r["oracle_test_auc"] for r in split_results]

    return {
        "test_mean": float(np.mean(test_aucs)),
        "test_std": float(np.std(test_aucs)),
        "oracle_mean": float(np.mean(oracle_aucs)),
        "oracle_std": float(np.std(oracle_aucs)),
        "gap": float(np.mean(test_aucs) - np.mean(oracle_aucs)),
        "n_splits": len(split_results),
        "selected_configs": [r["selected_cfg"] for r in split_results],
    }


def main(device="cpu"):
    datasets = ["enron", "weibo", "reddit", "amazon", "yelpchi",
                "blogcatalog", "facebook", "acm",
                "elliptic", "elliptic_plus_plus", "t_finance"]

    results = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        result = run_dataset(ds, n_splits=10, device=device)
        if result:
            reported = ORACLE_AUROCS.get(ds, 0)
            print(f"  Val-split: {result['test_mean']*100:.1f} +/- "
                  f"{result['test_std']*100:.1f}%  "
                  f"Oracle: {result['oracle_mean']*100:.1f}%  "
                  f"Reported: {reported}%  "
                  f"Gap: {result['gap']*100:+.1f}pp")
            configs_selected = set(result["selected_configs"])
            print(f"  Configs selected: {configs_selected}")
            results[ds] = result
            with open("results/validation_split_test.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    print("\n=== SUMMARY ===")
    print(f"{'Dataset':15s}  {'Val-split':>10s}  {'Oracle':>8s}  "
          f"{'Reported':>8s}  {'Gap':>6s}")
    for ds, r in results.items():
        print(f"{ds:15s}  "
              f"{r['test_mean']*100:5.1f}+/-{r['test_std']*100:4.1f}  "
              f"{r['oracle_mean']*100:6.1f}%  "
              f"{ORACLE_AUROCS.get(ds,0):7.1f}%  "
              f"{r['gap']*100:+5.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.device)
