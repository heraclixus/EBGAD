"""Test unsupervised criteria for config selection.

For each dataset, compute multiple unsupervised criteria across configs
and check which best correlates with AUROC.

Criteria:
1. Marginal likelihood (ML) — baseline, already shown to fail
2. Score skewness — higher = heavier anomaly tail
3. Bimodality coefficient — (skew²+1)/kurtosis, higher = more bimodal
4. Excess mass — fraction of scores > mean + 3*std
5. Max/median ratio — extreme outlier measure
6. Posterior predictive deviation (PPD) — KL between real and synthetic scores
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis, spearmanr

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

CONFIGS = [
    {"template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"template_type": "low", "graph_type": "affinity", "gamma": 0.5, "pca": None},
    {"template_type": "high", "graph_type": "affinity", "gamma": 0.5, "pca": None},
    {"template_type": "low", "graph_type": "original", "gamma": 0.1, "pca": None},
    {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": None},
    {"template_type": "low", "graph_type": "original", "gamma": 2.0, "pca": None},
    {"template_type": "zero", "graph_type": "original", "gamma": 0.5, "pca": None},
    {"template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": 64},
    {"template_type": "affinity", "graph_type": "original", "gamma": 0.5, "pca": 64},
    {"template_type": "affinity", "graph_type": "original", "gamma": 0.1, "pca": 64},
    {"template_type": "low", "graph_type": "original", "gamma": 0.5, "pca": 128},
]


def compute_ppd(V, lam_L, tmpl, S, rho, kappa, n, d, scores_real, n_synthetic=5):
    """Posterior predictive deviation: generate synthetic, compare scores."""
    q = Q_rho_eigenvalues(lam_L, rho, kappa)
    q_safe = np.maximum(q, 1e-12)

    # Generate synthetic from N(m, Q_rho^{-1})
    # In eigenspace: each mode j has variance 1/q_j
    all_synth_scores = []
    for _ in range(n_synthetic):
        # Sample in eigenspace
        z = np.random.randn(len(q), d) / np.sqrt(q_safe[:, None])
        # Transform to node space
        synth_delta = V @ z  # (n, d)
        # Compute J* scores on synthetic
        weighted = np.sqrt(q_safe)[:, None] * (V.T @ synth_delta)
        synth_scores = np.sum((V @ weighted)**2, axis=1)
        all_synth_scores.append(synth_scores)

    synth_scores = np.mean(all_synth_scores, axis=0)

    # Compare real vs synthetic score distributions
    # Use ratio of means: if real >> synthetic, anomalies are present
    real_mean = np.mean(scores_real)
    synth_mean = np.mean(synth_scores)
    mean_ratio = real_mean / (synth_mean + 1e-12)

    # KL-like divergence: compare tails
    real_tail = np.mean(scores_real > np.percentile(scores_real, 95))
    synth_tail = np.mean(synth_scores > np.percentile(synth_scores, 95))
    tail_ratio = real_tail / (synth_tail + 1e-12)

    # Max deviation
    real_max = np.max(scores_real) / (np.median(scores_real) + 1e-12)
    synth_max = np.max(synth_scores) / (np.median(synth_scores) + 1e-12)
    max_deviation = real_max / (synth_max + 1e-12)

    return {"mean_ratio": mean_ratio, "tail_ratio": tail_ratio,
            "max_deviation": max_deviation}


def eval_config_full(data, cfg, device="cpu"):
    """Evaluate config: ML-optimize rho, compute scores + all criteria."""
    cfg_kwargs = dict(
        kappa=1.0, nu=1.0, rho=0.5, lam_penalty=50.0,
        alpha=2.0, T=1.0, normalize_mode="zscore",
        prior_mean_mode="stationary", laplacian_variant="sym",
        d_hidden=64, n_hidden_layers=2, epochs=0, normalize_features=True,
        template_type=cfg["template_type"], graph_type=cfg["graph_type"],
        gamma=cfg["gamma"],
    )
    if cfg.get("pca") and data.x.shape[1] > cfg["pca"]:
        cfg_kwargs.update(dict(
            use_encoder=True, encoder_type="pca",
            encoder_hid_dim=cfg["pca"], encoder_num_layers=1,
            encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
            encoder_alpha=1.0, encoder_weight_decay=0.0,
        ))
    elif cfg.get("pca"):
        return None

    try:
        trainer = SOCTrainer(SOCTrainerConfig(**cfg_kwargs))
        trainer.train(data, device=device)
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
        tmpl = trainer.template.cpu().numpy()
        x = data.x.float()
        if hasattr(trainer, 'normalizer') and trainer.normalizer is not None:
            x = trainer.normalizer(x)
        if hasattr(trainer, 'encoder') and trainer.encoder is not None:
            with torch.no_grad():
                x = trainer.encoder(x.to(device), data.edge_index.to(device)).cpu()
        x_np = x.numpy()
        n, d = x_np.shape
        delta = x_np - tmpl
        S = compute_spectral_energy(delta, V)

        best_ml, best_rho, best_kappa = -np.inf, 0.5, 0.0
        for kappa in KAPPAS:
            rho = solve_rho_newton(S, lam_L, kappa, d)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = marginal_log_likelihood_stationary(S, q, n, d)
            if ml > best_ml:
                best_ml, best_rho, best_kappa = ml, rho, kappa

        if best_rho < 0.01:
            return None  # degenerate

        lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, best_rho, best_kappa)).float()
        V_t, tmpl_t = torch.from_numpy(V).float(), torch.from_numpy(tmpl).float()
        x_t = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x_np).float()

        with torch.no_grad():
            je = precision_energy_anomaly(x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                          prior_mean_mode="stationary").cpu().numpy()
            jr = precision_ratio_anomaly(x_t, tmpl_t, V_t, lam_Q, 2.0, 1.0,
                                         prior_mean_mode="stationary").cpu().numpy()

        y = data.y.numpy()
        mask = (y >= 0) & (y <= 1)

        # For each score type, compute all criteria
        results = {}
        for sname, scores in [("je", je), ("jr", jr)]:
            s = scores[mask] if mask.sum() > 0 else scores
            auc = roc_auc_score(y[mask], scores[mask])
            sk = skew(s)
            ku = kurtosis(s)
            bimod = (sk**2 + 1) / (ku + 3 + 1e-8)
            excess = np.mean(s > np.mean(s) + 3*np.std(s))
            maxmed = np.max(s) / (np.median(s) + 1e-12)

            ppd = compute_ppd(V, lam_L, tmpl, S, best_rho, best_kappa,
                              n, d, scores)

            results[sname] = {
                "auc": float(auc),
                "ml": float(best_ml),
                "skewness": float(sk),
                "bimodality": float(bimod),
                "excess_mass": float(excess),
                "max_median": float(maxmed),
                "ppd_mean_ratio": float(ppd["mean_ratio"]),
                "ppd_max_dev": float(ppd["max_deviation"]),
            }

        return results
    except Exception as e:
        return None


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)

    all_results = []
    for i, cfg in enumerate(CONFIGS):
        r = eval_config_full(data, cfg, device)
        if r:
            for sname in ["je", "jr"]:
                entry = dict(r[sname])
                entry["config_idx"] = i
                entry["score_type"] = sname
                entry["cfg"] = f"t={cfg['template_type']} g={cfg['graph_type']} " \
                               f"gam={cfg['gamma']} pca={cfg.get('pca','no')}"
                all_results.append(entry)

    if len(all_results) < 4:
        return None

    aucs = [r["auc"] for r in all_results]
    criteria = ["ml", "skewness", "bimodality", "excess_mass",
                "max_median", "ppd_mean_ratio", "ppd_max_dev"]

    correlations = {}
    for c in criteria:
        vals = [r[c] for r in all_results]
        corr, pval = spearmanr(vals, aucs)
        correlations[c] = float(corr)

        # Also check: does the criterion select the best config?
        best_by_criterion = all_results[np.argmax(vals)]
        best_by_auc = all_results[np.argmax(aucs)]
        correlations[f"{c}_selected_auc"] = float(best_by_criterion["auc"])

    correlations["oracle_auc"] = float(max(aucs))
    correlations["n_configs"] = len(all_results)

    return correlations


def main(device="cpu"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ds = args.dataset
    print(f"=== {ds} ===", flush=True)
    result = run_dataset(ds, args.device)

    if result:
        print(f"  Oracle AUC: {result['oracle_auc']*100:.1f}%")
        print(f"  Correlations with AUROC:")
        for c in ["ml", "skewness", "bimodality", "excess_mass",
                   "max_median", "ppd_mean_ratio", "ppd_max_dev"]:
            sel_auc = result.get(f"{c}_selected_auc", 0) * 100
            print(f"    {c:20s}  corr={result[c]:+.3f}  selected={sel_auc:.1f}%")

        with open(f"results/unsup_criteria_{ds}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
