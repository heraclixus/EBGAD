"""Majority vote ensemble: select config by consensus across criteria.

For each config, compute rank under each criterion.
Pick the config with the best average rank (or most top-3 appearances).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis

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


def eval_config(data, cfg, device="cpu"):
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
            return None

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

        # Compute criteria for BOTH J* and R
        results = []
        for sname, scores in [("je", je), ("jr", jr)]:
            s = scores
            auc = roc_auc_score(y[mask], s[mask])
            sk = skew(s)
            ku = kurtosis(s)
            bimod = (sk**2 + 1) / (ku + 3 + 1e-8)
            excess = np.mean(s > np.mean(s) + 3*np.std(s))
            maxmed = np.max(s) / (np.median(s) + 1e-12)

            # PPD: generate synthetic, compare
            q_safe = np.maximum(Q_rho_eigenvalues(lam_L, best_rho, best_kappa), 1e-12)
            synth_z = np.random.RandomState(42).randn(len(q_safe), d) / np.sqrt(q_safe[:, None])
            synth_delta = V @ synth_z
            synth_weighted = np.sqrt(q_safe)[:, None] * (V.T @ synth_delta)
            synth_scores = np.sum((V @ synth_weighted)**2, axis=1)
            ppd_dev = np.max(s) / (np.max(synth_scores) + 1e-12)

            results.append({
                "score_type": sname, "auc": auc, "ml": best_ml,
                "skewness": sk, "bimodality": bimod,
                "excess_mass": excess, "max_median": maxmed,
                "ppd_dev": ppd_dev,
            })

        return results
    except:
        return None


def majority_vote_select(all_entries):
    """Select config+score by majority vote across criteria."""
    criteria = ["ml", "skewness", "bimodality", "excess_mass",
                "max_median", "ppd_dev"]

    n = len(all_entries)
    if n == 0:
        return None

    # For each criterion, rank all entries (higher = better)
    ranks = np.zeros((n, len(criteria)))
    for ci, crit in enumerate(criteria):
        vals = np.array([e[crit] for e in all_entries])
        ranks[:, ci] = np.argsort(np.argsort(-vals))  # rank 0 = best

    # Average rank (lower = better)
    avg_rank = ranks.mean(axis=1)
    best_idx = np.argmin(avg_rank)

    # Also try: count how many criteria rank this entry in top-3
    top3_counts = (ranks < 3).sum(axis=1)
    best_top3_idx = np.argmax(top3_counts)

    return {
        "avg_rank_idx": int(best_idx),
        "avg_rank_auc": all_entries[best_idx]["auc"],
        "top3_idx": int(best_top3_idx),
        "top3_auc": all_entries[best_top3_idx]["auc"],
    }


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)

    all_entries = []
    cfg_names = []
    for i, cfg in enumerate(CONFIGS):
        tag = f"t={cfg['template_type']} g={cfg['graph_type']} gam={cfg['gamma']} pca={cfg.get('pca','no')}"
        result = eval_config(data, cfg, device)
        if result:
            for r in result:
                r["config_idx"] = i
                r["tag"] = tag
                all_entries.append(r)

    if not all_entries:
        return None

    # Oracle
    oracle_auc = max(e["auc"] for e in all_entries)

    # Majority vote
    vote = majority_vote_select(all_entries)

    # ML-selected
    ml_idx = np.argmax([e["ml"] for e in all_entries])
    ml_auc = all_entries[ml_idx]["auc"]

    return {
        "oracle": oracle_auc,
        "ml_auc": ml_auc,
        "vote_avg_rank_auc": vote["avg_rank_auc"],
        "vote_top3_auc": vote["top3_auc"],
        "n_entries": len(all_entries),
        "vote_tag": all_entries[vote["avg_rank_idx"]]["tag"],
        "vote_score": all_entries[vote["avg_rank_idx"]]["score_type"],
    }


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
        rep = ORACLE_AUROCS.get(ds, 0)
        print(f"  Oracle:     {result['oracle']*100:.1f}%")
        print(f"  ML-select:  {result['ml_auc']*100:.1f}%")
        print(f"  Vote(rank): {result['vote_avg_rank_auc']*100:.1f}%  "
              f"cfg={result['vote_tag']} score={result['vote_score']}")
        print(f"  Vote(top3): {result['vote_top3_auc']*100:.1f}%")
        print(f"  Reported:   {rep}%")

        with open(f"results/majority_vote_{ds}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
