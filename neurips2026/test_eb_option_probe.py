"""Label-free probe for new EB design choices on small benchmark datasets.

Compares four fully unsupervised variants under the same template pipeline:

1. legacy single-scale prior
2. single-scale + nullspace drop
3. two-scale prior
4. two-scale + nullspace drop

All continuous parameters are selected by the residual-space ML objective.
Labels are used only for reporting AUROC after selection.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import chi2, kstest
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(line_buffering=True)

from data_utils import load_data
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    compute_spectral_energy,
    coordinate_descent_stationary,
)
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig


def ks_null_deviation(scores: np.ndarray) -> float:
    """KS deviation against a fitted scaled chi-squared null."""
    s = scores[np.isfinite(scores) & (scores > 0)]
    if len(s) < 10:
        return 0.0
    mu = float(np.mean(s))
    var = float(np.var(s))
    if mu < 1e-12 or var < 1e-12:
        return 0.0
    k_est = max(0.01, 2.0 * mu * mu / var)
    a_est = var / (2.0 * mu)
    try:
        ks_stat, _ = kstest(s / a_est, chi2(df=k_est).cdf)
        return float(ks_stat)
    except Exception:
        return 0.0


def build_variants() -> List[Dict]:
    return [
        {
            "name": "legacy",
            "precision_family": "single_scale",
            "modeled_subspace": "legacy",
            "optimize_eta": False,
            "optimize_kappa2": False,
            "eta_init": 0.5,
            "kappa2_init": 5.0,
        },
        {
            "name": "drop_nullspace",
            "precision_family": "single_scale",
            "modeled_subspace": "drop_nullspace",
            "optimize_eta": False,
            "optimize_kappa2": False,
            "eta_init": 0.5,
            "kappa2_init": 5.0,
        },
        {
            "name": "two_scale",
            "precision_family": "two_scale",
            "modeled_subspace": "legacy",
            "optimize_eta": True,
            "optimize_kappa2": True,
            "eta_init": 0.5,
            "kappa2_init": 5.0,
        },
        {
            "name": "two_scale_drop_nullspace",
            "precision_family": "two_scale",
            "modeled_subspace": "drop_nullspace",
            "optimize_eta": True,
            "optimize_kappa2": True,
            "eta_init": 0.5,
            "kappa2_init": 5.0,
        },
    ]


def canonical_dataset_name(name: str) -> str:
    if name.lower() == "yelpchi":
        return "YelpChi"
    if name.lower() == "facebook":
        return "Facebook"
    return name


def setup_trainer(data, device: str, variant: Dict) -> SOCTrainer:
    d = data.x.shape[1]
    cfg = SOCTrainerConfig(
        kappa=1.0,
        kappa2=variant.get("kappa2_init", 5.0),
        nu=1.0,
        rho=0.5,
        lam_penalty=50.0,
        alpha=2.0,
        T=1.0,
        normalize_mode="zscore",
        prior_mean_mode="stationary",
        laplacian_variant="sym",
        normalize_features=True,
        template_type="low",
        graph_type="original",
        gamma=1.0,
        epochs=0,
        pca_features=d > 100,
        pca_n_components=64,
        precision_family=variant["precision_family"],
        eta=variant.get("eta_init", 0.5),
        modeled_subspace=variant["modeled_subspace"],
    )
    trainer = SOCTrainer(cfg)
    trainer._setup(data, device=device)
    return trainer


def optimize_variant(
    S: np.ndarray,
    lam_L: np.ndarray,
    n: int,
    d: int,
    variant: Dict,
) -> Dict:
    best = None
    best_ml = -np.inf
    rho_inits = [0.0, 0.5, 1.0]
    kappa_inits = [0.1, 1.0, 5.0]
    eta_inits = [0.25, 0.5, 0.75] if variant["precision_family"] == "two_scale" else [0.5]

    for rho_init in rho_inits:
        for kappa_init in kappa_inits:
            for eta_init in eta_inits:
                try:
                    opt = coordinate_descent_stationary(
                        S,
                        lam_L,
                        n,
                        d,
                        rho_init=rho_init,
                        kappa_init=kappa_init,
                        kappa2_init=max(5.0, kappa_init),
                        nu=1.0,
                        precision_family=variant["precision_family"],
                        eta_init=eta_init,
                        optimize_eta=variant["optimize_eta"],
                        optimize_kappa2=variant["optimize_kappa2"],
                        modeled_subspace=variant["modeled_subspace"],
                        use_data_driven_bounds=True,
                        bound_percentile=50.0,
                        n_iters=8,
                    )
                    if opt["marginal_likelihood"] > best_ml:
                        best = opt
                        best_ml = opt["marginal_likelihood"]
                except Exception:
                    continue

    if best is None:
        raise RuntimeError(f"Failed to optimize variant={variant['name']}")
    return best


def score_variant(
    trainer: SOCTrainer,
    opt: Dict,
    variant: Dict,
    mask: np.ndarray,
    y: np.ndarray,
) -> Dict:
    x = trainer.x_T.detach().cpu()
    tmpl = trainer.template.detach().cpu()
    V = trainer.V.detach().cpu().numpy()
    V_t = torch.from_numpy(V).float()
    lam_L = trainer.lam_L_model.detach().cpu().numpy()

    q = Q_rho_eigenvalues(
        lam_L,
        opt["rho"],
        opt["kappa"],
        nu=1.0,
        precision_family=variant["precision_family"],
        eta=opt.get("eta", 0.5),
        kappa2=opt.get("kappa2"),
    )
    lam_Q = torch.from_numpy(q).float()

    with torch.no_grad():
        j_scores = precision_energy_anomaly(
            x,
            tmpl,
            V_t,
            lam_Q,
            alpha=2.0,
            T=1.0,
            prior_mean_mode="stationary",
        ).cpu().numpy()
        r_scores = precision_ratio_anomaly(
            x,
            tmpl,
            V_t,
            lam_Q,
            alpha=2.0,
            T=1.0,
            prior_mean_mode="stationary",
        ).cpu().numpy()

    ks_j = ks_null_deviation(j_scores[mask])
    ks_r = ks_null_deviation(r_scores[mask])
    use_j = ks_j >= ks_r
    selected = j_scores if use_j else r_scores
    auc_j = float(roc_auc_score(y[mask], j_scores[mask]))
    auc_r = float(roc_auc_score(y[mask], r_scores[mask]))
    oracle_score = "J*" if auc_j >= auc_r else "R"
    selected_score = "J*" if use_j else "R"
    auc_selected = auc_j if use_j else auc_r
    auc_oracle = max(auc_j, auc_r)

    return {
        "score_name": selected_score,
        "oracle_score": oracle_score,
        "ks_matches_oracle": bool(selected_score == oracle_score),
        "auc_selected": float(auc_selected),
        "auc_oracle": float(auc_oracle),
        "selection_gap": float(auc_oracle - auc_selected),
        "auc_j": float(auc_j),
        "auc_r": float(auc_r),
        "ks_j": float(ks_j),
        "ks_r": float(ks_r),
        "rho": float(opt["rho"]),
        "kappa": float(opt["kappa"]),
        "kappa2": float(opt.get("kappa2", opt["kappa"])),
        "eta": float(opt.get("eta", 0.5)),
        "modeled_rank": int(opt.get("modeled_rank", len(lam_L))),
        "marginal_likelihood": float(opt["marginal_likelihood"]),
    }


def run_dataset(dataset: str, device: str) -> Dict:
    data = load_data(canonical_dataset_name(dataset))

    y = data.y.cpu().numpy()
    mask = (y >= 0) & (y <= 1)

    results = {}
    for variant in build_variants():
        print(f"=== {dataset} | {variant['name']} ===", flush=True)
        trainer = setup_trainer(data, device=device, variant=variant)

        x = trainer.x_T.detach().cpu().numpy()
        tmpl = trainer.template.detach().cpu().numpy()
        V = trainer.V.detach().cpu().numpy()
        lam_L = trainer.lam_L_model.detach().cpu().numpy()
        delta = x - tmpl
        S = compute_spectral_energy(delta, V)
        n, d = x.shape

        opt = optimize_variant(S, lam_L, n, d, variant)
        res = score_variant(trainer, opt, variant, mask, y)
        results[variant["name"]] = res
        print(
            "  sel={score_name} oracle={oracle_score} match={ks_matches_oracle} "
            "auc={auc_selected:.4f} oracle_auc={auc_oracle:.4f} gap={selection_gap:.4f} "
            "J={auc_j:.4f} R={auc_r:.4f} "
            "rho={rho:.3f} k1={kappa:.3f} k2={kappa2:.3f} eta={eta:.3f} rank={modeled_rank}".format(
                **res
            ),
            flush=True,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/eb_option_probe")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = run_dataset(args.dataset, args.device)
    out_path = os.path.join(args.output_dir, f"{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
