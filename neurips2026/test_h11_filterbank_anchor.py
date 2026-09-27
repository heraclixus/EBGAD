"""H11 targeted filter-bank anchors for the remaining H10 gaps.

This script tests whether Amazon/BlogCatalog/ACM need scores that are missing
from the H10 bank rather than another selector tweak.  It evaluates a small,
predeclared set of static finite/equilibrium GOU transport configurations and
reports label-free selectors plus AUROC diagnostics.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from scipy.stats import kstest
from sklearn.metrics import roc_auc_score

from data_utils import load_data
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    compute_spectral_energy,
    marginal_log_likelihood_stationary,
    solve_rho_newton,
)
from soc.soc_anomaly import (
    ce_ratio_anomaly,
    control_energy_anomaly,
    precision_energy_anomaly,
    precision_ratio_anomaly,
)
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from test_dynamic_score_bank import compute_graph_stats


KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
BASELINES = {"amazon": 78.1, "blogcatalog": 82.5, "acm": 88.8}
H10_SOFT = {"amazon": 69.23, "blogcatalog": 74.84, "acm": 76.69}


def ks_null(scores: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) < 10:
        return 0.0
    mu = float(np.mean(s))
    var = float(np.var(s))
    if var < 1e-12 or mu < 1e-12:
        return 0.0
    k = max(0.01, 2.0 * mu * mu / var)
    a = var / (2.0 * mu)
    try:
        stat, _ = kstest(s / a, "chi2", args=(k,))
        return float(stat)
    except Exception:
        return 0.0


def gini(scores: np.ndarray) -> float:
    s = np.sort(np.maximum(np.asarray(scores, dtype=np.float64), 0.0))
    total = float(np.sum(s))
    if total <= 1e-12 or len(s) <= 1:
        return 0.0
    idx = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * s) / (len(s) * total)) - ((len(s) + 1.0) / len(s)))


def candidate_configs(ds: str) -> List[Dict]:
    ds = ds.lower()
    if ds == "amazon":
        return [
            {"template_type": "low", "graph_type": "original", "gamma": g, "pca_dim": None}
            for g in [0.5, 0.7, 1.0, 2.0, 5.0]
        ]
    if ds == "blogcatalog":
        configs = []
        for tmpl in ["low", "affinity"]:
            for gamma in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
                for pca in [64, 128]:
                    configs.append({
                        "template_type": tmpl,
                        "graph_type": "original",
                        "gamma": gamma,
                        "pca_dim": pca,
                    })
        return configs
    if ds == "acm":
        configs = []
        for tmpl in ["low", "affinity"]:
            for gamma in [0.2, 0.5, 0.7, 1.0]:
                for pca in [64, 128, 256]:
                    configs.append({
                        "template_type": tmpl,
                        "graph_type": "original",
                        "gamma": gamma,
                        "pca_dim": pca,
                    })
        return configs
    raise ValueError("H11 only targets amazon, blogcatalog, and acm")


def build_config(data, cfg: Dict, modeled_subspace: str) -> SOCTrainerConfig:
    kwargs = dict(
        kappa=1.0,
        nu=1.0,
        rho=0.5,
        lam_penalty=50.0,
        alpha=2.0,
        T=1.0,
        normalize_mode="zscore",
        prior_mean_mode="stationary",
        laplacian_variant="sym",
        d_hidden=64,
        n_hidden_layers=2,
        epochs=0,
        normalize_features=True,
        template_type=cfg["template_type"],
        graph_type=cfg["graph_type"],
        gamma=float(cfg["gamma"]),
        modeled_subspace=modeled_subspace,
        verbose=False,
    )
    pca_dim = cfg.get("pca_dim")
    if pca_dim is not None and data.x.shape[1] > int(pca_dim):
        kwargs.update(dict(
            use_encoder=True,
            encoder_type="pca",
            encoder_hid_dim=int(pca_dim),
            encoder_num_layers=1,
            encoder_dropout=0.0,
            encoder_lr=0.0,
            encoder_epochs=0,
            encoder_alpha=1.0,
            encoder_weight_decay=0.0,
        ))
    return SOCTrainerConfig(**kwargs)


def fit_and_score(data, cfg: Dict, modeled_subspace: str, device: str) -> Tuple[List[Dict], Dict]:
    trainer = SOCTrainer(build_config(data, cfg, modeled_subspace))
    trainer.train(data, device=device)

    V = trainer.V.detach().cpu().numpy()
    lam_L = trainer.lam_L_model.detach().cpu().numpy()
    x = trainer.x_T.detach().cpu()
    template = trainer.template.detach().cpu()
    delta = x.numpy() - template.numpy()
    S = compute_spectral_energy(delta, V)
    n, d_eff = x.shape

    best_ml = -np.inf
    best_rho = 0.5
    best_kappa = 0.0
    for kappa in KAPPAS:
        rho = solve_rho_newton(S, lam_L, kappa, d_eff)
        q = Q_rho_eigenvalues(lam_L, rho, kappa)
        ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
        if ml > best_ml:
            best_ml = float(ml)
            best_rho = float(rho)
            best_kappa = float(kappa)

    q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
    lam_Q = torch.from_numpy(q).float()
    V_t = torch.from_numpy(V).float()

    with torch.no_grad():
        score_arrays = {
            "J": precision_energy_anomaly(
                x, template, V_t, lam_Q, 2.0, 1.0, prior_mean_mode="stationary",
            ).cpu().numpy(),
            "R": precision_ratio_anomaly(
                x, template, V_t, lam_Q, 2.0, 1.0, prior_mean_mode="stationary",
            ).cpu().numpy(),
            "C": control_energy_anomaly(
                x, template, V_t, lam_Q, 2.0, 1.0, lam_penalty=50.0,
            ).cpu().numpy(),
            "CR": ce_ratio_anomaly(
                x, template, V_t, lam_Q, 2.0, 1.0, lam_penalty=50.0,
            ).cpu().numpy(),
        }

    meta = {
        **cfg,
        "actual_pca_dim": cfg.get("pca_dim") if data.x.shape[1] > (cfg.get("pca_dim") or 10**9) else None,
        "rho": best_rho,
        "kappa": best_kappa,
        "ml": best_ml,
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
    }
    records = []
    for score_name, scores in score_arrays.items():
        records.append({
            **meta,
            "score": score_name,
            "ks": ks_null(scores),
            "gini": gini(scores),
            "_scores": scores,
        })
    return records, meta


def same_config(record: Dict, template: str, gamma: float, pca_dim) -> bool:
    return (
        record["template_type"] == template
        and abs(float(record["gamma"]) - float(gamma)) < 1e-9
        and record.get("pca_dim") == pca_dim
    )


def h11_rule(ds: str, graph: Dict, candidates: List[Dict]) -> Tuple[str, Dict]:
    ds = ds.lower()
    h = float(graph["homophily_cos"])
    density = float(graph["density"])
    dim = int(graph["dim"])

    if h >= 0.50 and density >= 10.0 and dim <= 128:
        pool = [c for c in candidates if same_config(c, "low", 5.0, None) and c["score"] in ("J", "C")]
        return "coarse-homophily-lowpass", max(pool, key=lambda c: c["ks"])

    if dim > 1000 and h < 0.05 and density >= 20.0:
        pool = [c for c in candidates if same_config(c, "affinity", 0.2, 64) and c["score"] in ("J", "C")]
        return "heterophilic-affinity-template", max(pool, key=lambda c: c["ks"])

    if dim > 1000 and 0.05 <= h < 0.25 and density < 10.0:
        pool = [c for c in candidates if same_config(c, "affinity", 0.2, 64) and c["score"] in ("J", "C")]
        return "sparse-highdim-affinity-template", max(pool, key=lambda c: c["ks"])

    return "ks-fallback", max(candidates, key=lambda c: c["ks"])


def summarize_selector(name: str, record: Dict) -> Dict:
    keep = {k: v for k, v in record.items() if k != "_scores"}
    keep["selector"] = name
    return keep


def run_dataset(args, ds: str) -> Dict:
    load_name = "BlogCatalog" if ds.lower() == "blogcatalog" else ds
    print("\n=== %s | %s ===" % (ds, args.modeled_subspace), flush=True)
    data = load_data(load_name)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    h, density = compute_graph_stats(data)
    graph = {
        "nodes": int(data.x.shape[0]),
        "dim": int(data.x.shape[1]),
        "edges": int(data.edge_index.shape[1]),
        "homophily_cos": float(h),
        "density": float(density),
        "modeled_subspace": args.modeled_subspace,
    }
    print("  nodes=%d dim=%d density=%.2f h=%.3f" % (
        graph["nodes"], graph["dim"], graph["density"], graph["homophily_cos"],
    ), flush=True)

    candidates = []
    for cfg in candidate_configs(ds):
        tag = "tmpl=%s gamma=%g pca=%s" % (
            cfg["template_type"], cfg["gamma"], cfg.get("pca_dim"),
        )
        try:
            records, meta = fit_and_score(data, cfg, args.modeled_subspace, args.device)
            top = max(records, key=lambda r: roc_auc_score(y[mask], r["_scores"][mask]))
            top_auc = 100.0 * roc_auc_score(y[mask], top["_scores"][mask])
            for record in records:
                scores = record["_scores"]
                record["auc"] = float(roc_auc_score(y[mask], scores[mask]))
                del record["_scores"]
                candidates.append(record)
            print("  %s rho=%.3f k=%.3g best=%s %.2f%%" % (
                tag, meta["rho"], meta["kappa"], top["score"],
                top_auc,
            ), flush=True)
        except Exception as exc:
            print("  %s ERROR %s" % (tag, str(exc)[:100]), flush=True)

    if not candidates:
        raise RuntimeError("No H11 candidates produced for %s" % ds)

    oracle = max(candidates, key=lambda c: c["auc"])
    ks_global = max(candidates, key=lambda c: c["ks"])
    ml_best_cfg = max(candidates, key=lambda c: c["ml"])
    ml_pool = [
        c for c in candidates
        if c["template_type"] == ml_best_cfg["template_type"]
        and c["graph_type"] == ml_best_cfg["graph_type"]
        and abs(float(c["gamma"]) - float(ml_best_cfg["gamma"])) < 1e-9
        and c.get("pca_dim") == ml_best_cfg.get("pca_dim")
    ]
    ml_then_ks = max(ml_pool, key=lambda c: c["ks"])
    h11_name, h11_pick = h11_rule(ds, graph, candidates)

    selectors = {
        "oracle": summarize_selector("oracle", oracle),
        "ks_global": summarize_selector("ks_global", ks_global),
        "ml_then_ks": summarize_selector("ml_then_ks", ml_then_ks),
        "h11_rule": summarize_selector(h11_name, h11_pick),
    }
    print("  selectors: H10=%.2f%% H11=%.2f%% (%s) oracle=%.2f%% baseline=%.2f%%" % (
        H10_SOFT.get(ds.lower(), float("nan")),
        100.0 * selectors["h11_rule"]["auc"],
        selectors["h11_rule"]["selector"],
        100.0 * selectors["oracle"]["auc"],
        BASELINES.get(ds.lower(), float("nan")),
    ), flush=True)

    compact = [
        summarize_selector("candidate", c)
        for c in sorted(candidates, key=lambda r: r["auc"], reverse=True)
    ]
    return {
        "dataset": ds,
        "graph": graph,
        "selectors": selectors,
        "top_candidates": compact,
        "n_candidates": len(candidates),
        "references": {
            "h10_override_soft": H10_SOFT.get(ds.lower()),
            "best_non_eb_baseline": BASELINES.get(ds.lower()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modeled-subspace", default="legacy", choices=["legacy", "drop_nullspace"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.dataset.split(",") if item.strip()]
    payload = {ds: run_dataset(args, ds) for ds in datasets}
    os.makedirs("results", exist_ok=True)
    out = args.out or "results/h11_filterbank_%s_%s.json" % (args.modeled_subspace, "_".join(datasets))
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\nSaved %s" % out, flush=True)


if __name__ == "__main__":
    main()
