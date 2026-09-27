"""E2 isolation ablation: does the EB-fitted GOU spectral family matter,
or would any spectral filter bank under the same label-free selector do?

For each dataset we build the exact Table-1 residual pipeline (same graph
operator, template, PCA, truncation), then score the SAME residuals with
five spectral precision families:

  gou_eb     EB-fitted GOU precision q_rho(lam) + finite-horizon transport
             interpolation (the paper's bank)
  gou_fixed  GOU precision with fixed rho=0.5, kappa=1 (no EB fit)
  heat       heat-kernel precision exp(t * lam), t grid (no fit)
  poly       polynomial precision (c + lam)^p, c/p grid (no fit)
  flat       precision = 1 (graph-free residual energy)

Every family goes through the identical label-free selector
(evaluate_score_bank: NullKS + stability diagnostics). We report the
selected AUROC and the oracle AUROC per family.
"""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data_utils import load_data
from test_dynamic_score_bank import (
    KAPPAS,
    compute_graph_stats,
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    format_horizon,
    horizon_score_bank,
    node_unsupervised_references,
    prepare_model_entries,
    q_for_fit,
    score_nodes,
)
from soc.prior_optimizer import Q_rho_eigenvalues
from soc.soc_anomaly import ce_ratio_anomaly, control_energy_anomaly

ERA_HORIZONS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, np.inf]
ERA_KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5,
              1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
ERA_TAUS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0]
ERA_LAMS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]

# Table-1-era per-dataset configs (subspace, template gammas, truncated_k,
# template type, pca override)
DS_CONFIG = {
    "weibo": dict(sub="drop_nullspace", gammas=[0.2, 0.5, 0.7], tk=None, tmpl="low"),
    "reddit": dict(sub="drop_nullspace", gammas=[0.7, 1.0, 2.0], tk=None, tmpl="low"),
    "amazon": dict(sub="drop_nullspace", gammas=[0.5, 0.7, 1.0, 2.0], tk=None, tmpl="low"),
    "yelpchi": dict(sub="drop_nullspace", gammas=[0.7, 1.0, 2.0], tk=500, tmpl="low"),
    "blogcatalog": dict(sub="drop_nullspace", gammas=[0.15, 0.18, 0.22], tk=None,
                        tmpl="affinity", pca=48),
    "facebook": dict(sub="drop_nullspace", gammas=[0.5, 0.7, 1.0], tk=None, tmpl="low"),
    "acm": dict(sub="drop_nullspace", gammas=[0.1, 0.15, 0.2], tk=500,
                tmpl="affinity", pca=256),
    "elliptic": dict(sub="legacy", gammas=[0.7, 1.0, 2.0], tk=300, tmpl="low"),
    "elliptic_plus_plus": dict(sub="legacy", gammas=[0.7, 1.0, 2.0], tk=300, tmpl="low"),
    "t_finance": dict(sub="legacy", gammas=[0.1], tk=500, tmpl="low"),
}

HEAT_TS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
POLY_GRID = [(c, p) for c in (0.01, 0.1, 1.0) for p in (1, 2)]


def generic_bank(entry: dict, precisions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name, prec in precisions.items():
        je, jr = score_nodes(entry["delta"], entry["V"], prec)
        scores["J|%s" % name] = je.astype(np.float32, copy=False)
        scores["R|%s" % name] = jr.astype(np.float32, copy=False)
    return scores


def transport_scores(entry: dict, q: np.ndarray) -> dict[str, np.ndarray]:
    """The paper's C/CR control-energy bank over the era (Gamma, lam_c) grid."""
    import torch
    delta = torch.from_numpy(np.asarray(entry["delta"], dtype=np.float32))
    zeros = torch.zeros_like(delta)
    V = torch.from_numpy(np.asarray(entry["V"], dtype=np.float32))
    lam_q = torch.from_numpy(np.asarray(q, dtype=np.float32))
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for tau in ERA_TAUS:
            for lp in ERA_LAMS:
                c = control_energy_anomaly(delta, zeros, V, lam_q, 1.0, float(tau),
                                           lam_penalty=float(lp))
                cr = ce_ratio_anomaly(delta, zeros, V, lam_q, 1.0, float(tau),
                                      lam_penalty=float(lp))
                out["C@%g,%g" % (tau, lp)] = c.cpu().numpy().astype(np.float32)
                out["CR@%g,%g" % (tau, lp)] = cr.cpu().numpy().astype(np.float32)
    return out


def family_banks(entry: dict, fit: dict) -> dict[str, dict[str, np.ndarray]]:
    lam = np.asarray(entry["lam_L"], dtype=np.float64)
    banks: dict[str, dict[str, np.ndarray]] = {}

    q_eb = q_for_fit(entry, fit)
    banks["gou_eb"] = horizon_score_bank(entry, q_eb, ERA_HORIZONS)
    banks["gou_eb"].update(transport_scores(entry, q_eb))

    q_fixed = Q_rho_eigenvalues(lam, rho=0.5, kappa=1.0, nu=1.0)
    banks["gou_fixed"] = horizon_score_bank(entry, q_fixed, ERA_HORIZONS)
    banks["gou_fixed"].update(transport_scores(entry, q_fixed))

    heat = {}
    lam_scale = float(np.max(lam)) or 1.0
    for t in HEAT_TS:
        heat["t=%g" % t] = np.exp(t * lam / lam_scale)
    banks["heat"] = generic_bank(entry, heat)

    poly = {}
    for c, p in POLY_GRID:
        poly["c=%g,p=%d" % (c, p)] = np.power(c + lam, p)
    banks["poly"] = generic_bank(entry, poly)

    banks["flat"] = generic_bank(entry, {"one": np.ones_like(lam)})
    return banks


def run_dataset(ds: str, device: str) -> dict:
    cfg = DS_CONFIG[ds]
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else
                 "BlogCatalog" if ds == "blogcatalog" else ds)
    print("\n=== %s | %s ===" % (ds, cfg["sub"]), flush=True)
    data = load_data(load_name)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    pca_dim = cfg.get("pca", 64 if data.x.shape[1] > 100 else None)

    entries = prepare_model_entries(
        data,
        graph_types=["original"],
        template_gammas=cfg["gammas"],
        pca_dim=pca_dim,
        truncated_k=cfg["tk"],
        modeled_subspace=cfg["sub"],
        template_type=cfg["tmpl"],
        device=device,
    )
    fit = fit_best_entry(entries, ERA_HORIZONS, ERA_KAPPAS, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    print("  fit: %s rho=%.3f kappa=%g ml=%.0f" % (
        fit["entry_key"], fit["rho"], fit["kappa"], fit["ml"]), flush=True)
    references = node_unsupervised_references(data, entry["delta"])

    out = {"dataset": ds, "fit": {k: fit[k] for k in ("entry_key", "rho", "kappa", "ml")},
           "families": {}}
    for family, scores in family_banks(entry, fit).items():
        summary = evaluate_score_bank(y, mask, scores, references=references)
        sel = summary.get("selectors", {})
        row = {
            "n_scores": summary.get("n_scores"),
            "auc_ks_selected": summary.get("auc_ks_selected"),
            "score_ks": summary.get("score_ks"),
            "auc_oracle": summary.get("auc_oracle"),
            "oracle_score": summary.get("oracle_score"),
            "auc_graph_tail": (sel.get("graph_tail") or {}).get("auc"),
            "auc_stability": (sel.get("stability") or {}).get("auc"),
        }
        out["families"][family] = row
        print("  %-10s sel(KS)=%5.1f%% (%s)  oracle=%5.1f%% (%s)  graph_tail=%5.1f%%" % (
            family,
            100.0 * (row["auc_ks_selected"] or 0),
            row["score_ks"],
            100.0 * (row["auc_oracle"] or 0),
            row["oracle_score"],
            100.0 * (row["auc_graph_tail"] or 0),
        ), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DS_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/e2/e2_isolation.json")
    args = parser.parse_args()

    results = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        try:
            results[ds] = run_dataset(ds, args.device)
        except Exception as exc:
            print("  ERROR %s: %s" % (ds, str(exc)[:200]), flush=True)
            results[ds] = {"error": str(exc)[:500]}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print("\nSaved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
