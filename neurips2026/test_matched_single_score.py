"""Matched single-score horizon comparison.

For each dataset, freeze the paper's published Table 6 configuration
(template family, template gamma, truncation k, PCA, modeled subspace),
re-run the stationary EB fit at that single configuration (paper kappa grid,
Newton rho), and score two matched single-score families under the one fit:

  energy: C_{Gamma, lambda_c=inf} at finite Gamma in {0.5,1,2,5,10,20}
          vs its limit J* (Gamma=inf).
  ratio:  CR_{Gamma, lambda_c=inf} at the same finite Gammas vs R.

No endpoint-tolerance grid, no fusion, no profile aggregation. The label-free
criterion is the paper's NullKS (moment-matched chi-squared KS), applied to
the finite members only, and also to the joint finite+limit pool.

Validation targets: fitted (rho, kappa) should reproduce Table 6; J*/R AUROC
should reproduce Table 8 (Weibo 95.1/47.3, Reddit 61.8/51.8, Amazon 77.9/48.2,
YelpChi 70.7/63.7, BlogCatalog 78.8/59.1, Facebook 62.1/85.9, ACM 83.5/52.0,
Elliptic 55.6/73.0, Elliptic++ 54.8/72.4, T-Finance 81.3/83.3).
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

np.random.seed(0)

from data_utils import load_data
from soc.soc_anomaly import ce_ratio_anomaly, control_energy_anomaly
from test_dynamic_score_bank import (
    entry_for_fit,
    fit_best_entry,
    horizon_score_bank,
    prepare_model_entries,
    q_for_fit,
    safe_auc,
)
from test_multiscale_gou_poe import format_horizon, ks_null_deviation

# Paper-text horizon grid (B-Appendix: Gamma in {0.5,1,2,5,10,20,inf}) and the
# pipeline's DEFAULT_HORIZONS grid actually used to build the bank
# (0.02..50); score the union, report selections over each subset.
TEXT_GAMMAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
PIPELINE_GAMMAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
ALL_GAMMAS = sorted(set(TEXT_GAMMAS) | set(PIPELINE_GAMMAS))
PAPER_KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

# (Gamma, lambda_c) single-score grids for the soft-endpoint C/CR family.
# TEXT_LAM_GRID follows B-Appendix (Gamma x lambda_c finite members);
# ERA_LAM_GRID is the pipeline-era grid from test_e2_isolation_banks.py.
TEXT_LAM_GRID = [(t, l) for t in TEXT_GAMMAS for l in (0.1, 1.0, 10.0, 100.0)]
ERA_LAM_GRID = [(t, l) for t in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0)
                for l in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0)]

# Table 6 (tab:ebgad_hyperparams) per-dataset configuration; modeled subspace
# from the Table-1-era DS_CONFIG in test_e2_isolation_banks.py.
# PCA deviations from Table 6 as printed: Facebook/Elliptic/Elliptic++ use
# pca=64 (Table 6 says "no", but only pca=64 reproduces Table 6's fitted
# rho/kappa and Table 8's J*/R).
TABLE6 = {
    "weibo": dict(tmpl="low", gamma=0.50, tk=None, pca=64,
                  rho=0.99, kappa=2.00, sub="drop_nullspace"),
    "reddit": dict(tmpl="low", gamma=1.00, tk=None, pca=None,
                   rho=0.85, kappa=1.00, sub="drop_nullspace"),
    "amazon": dict(tmpl="low", gamma=5.00, tk=None, pca=None,
                   rho=0.46, kappa=0.50, sub="drop_nullspace"),
    "yelpchi": dict(tmpl="low", gamma=1.00, tk=500, pca=None,
                    rho=1.00, kappa=20.0, sub="drop_nullspace"),
    "blogcatalog": dict(tmpl="affinity", gamma=0.18, tk=None, pca=48,
                        rho=1.00, kappa=1.00, sub="drop_nullspace"),
    "facebook": dict(tmpl="low", gamma=0.70, tk=None, pca=64,
                     rho=0.73, kappa=3.00, sub="drop_nullspace"),
    "acm": dict(tmpl="affinity", gamma=0.15, tk=500, pca=256,
                rho=1.00, kappa=0.01, sub="drop_nullspace"),
    "acm_aff": dict(tmpl="affinity", gamma=0.15, tk=500, pca=256,
                    rho=1.00, kappa=0.01, sub="drop_nullspace",
                    graph="affinity", load="acm"),
    "t_finance": dict(tmpl="low", gamma=0.20, tk=500, pca=None,
                      rho=0.999, kappa=0.00, sub="legacy"),
    "elliptic": dict(tmpl="low", gamma=1.00, tk=300, pca=64,
                     rho=1.00, kappa=20.0, sub="legacy"),
    "elliptic_plus_plus": dict(tmpl="low", gamma=1.00, tk=300, pca=64,
                               rho=1.00, kappa=20.0, sub="legacy"),
}

TABLE8 = {  # paper Table 8 (tab:ks_ablation) J*, R rows, AUROC x100
    "weibo": (95.1, 47.3), "reddit": (61.8, 51.8), "amazon": (77.9, 48.2),
    "yelpchi": (70.7, 63.7), "blogcatalog": (78.8, 59.1),
    "facebook": (62.1, 85.9), "acm": (83.5, 52.0), "elliptic": (55.6, 73.0),
    "elliptic_plus_plus": (54.8, 72.4), "t_finance": (81.3, 83.3),
}


def lam_grid_bank(entry, q, pairs) -> dict:
    """Single soft-endpoint scores C/CR at each (Gamma, lambda_c) pair."""
    import torch
    delta = torch.from_numpy(np.asarray(entry["delta"], dtype=np.float32))
    zeros = torch.zeros_like(delta)
    V = torch.from_numpy(np.asarray(entry["V"], dtype=np.float32))
    lam_q = torch.from_numpy(np.asarray(q, dtype=np.float32))
    out = {}
    with torch.no_grad():
        for tau, lp in pairs:
            c = control_energy_anomaly(delta, zeros, V, lam_q, 1.0, float(tau),
                                       lam_penalty=float(lp))
            cr = ce_ratio_anomaly(delta, zeros, V, lam_q, 1.0, float(tau),
                                  lam_penalty=float(lp))
            out["C@%g,%g" % (tau, lp)] = c.cpu().numpy().astype(np.float32)
            out["CR@%g,%g" % (tau, lp)] = cr.cpu().numpy().astype(np.float32)
    return out


def scored(per_bank, y_eval, mask) -> dict:
    per_score = {}
    for name, score in per_bank.items():
        s = np.nan_to_num(np.asarray(score, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmax(s) <= np.nanmin(s):
            continue
        per_score[name] = {
            "auc": float(safe_auc(y_eval, s[mask])),
            "nullks": float(ks_null_deviation(s[mask])),
        }
    return per_score


def lam_family_report(prefix, lam_per_score, limit_record, limit_name) -> dict:
    per_score = {n: r for n, r in lam_per_score.items()
                 if n.startswith(prefix + "@")}
    per_score[limit_name] = limit_record
    text_names = ["%s@%g,%g" % (prefix, t, l) for t, l in TEXT_LAM_GRID]
    era_names = ["%s@%g,%g" % (prefix, t, l) for t, l in ERA_LAM_GRID]
    return {
        "per_score": per_score,
        "text_grid": pool_selection(per_score, text_names, limit_name),
        "era_grid": pool_selection(per_score, era_names, limit_name),
    }


def pool_selection(per_score, finite_names, limit_name) -> dict:
    finite = {n: per_score[n] for n in finite_names if n in per_score}
    limit = per_score.get(limit_name)
    if not finite or limit is None:
        return {}
    sel_name = max(finite, key=lambda n: finite[n]["nullks"])
    ora_name = max(finite, key=lambda n: finite[n]["auc"])
    joint = dict(finite)
    joint[limit_name] = limit
    joint_name = max(joint, key=lambda n: joint[n]["nullks"])
    return {
        "selected_finite": {"name": sel_name, **finite[sel_name]},
        "oracle_finite": {"name": ora_name, **finite[ora_name]},
        "limit": {"name": limit_name, **limit},
        "joint_nullks_pick": {"name": joint_name, **joint[joint_name]},
    }


def family_report(prefix, bank, y_eval, mask) -> dict:
    per_score = {}
    for name, score in bank.items():
        if not name.startswith(prefix + "@"):
            continue
        s = np.nan_to_num(np.asarray(score, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmax(s) <= np.nanmin(s):
            continue
        per_score[name] = {
            "auc": float(safe_auc(y_eval, s[mask])),
            "nullks": float(ks_null_deviation(s[mask])),
        }
    limit_name = "%s@inf" % prefix
    text_names = ["%s@%s" % (prefix, format_horizon(h)) for h in TEXT_GAMMAS]
    pipe_names = ["%s@%s" % (prefix, format_horizon(h)) for h in PIPELINE_GAMMAS]
    return {
        "per_score": per_score,
        "text_grid": pool_selection(per_score, text_names, limit_name),
        "pipeline_grid": pool_selection(per_score, pipe_names, limit_name),
    }


def run_dataset(ds: str, device: str, impose_fit: bool = False) -> dict:
    cfg = TABLE6[ds]
    base = cfg.get("load", ds)
    load_name = ("YelpChi" if base == "yelpchi" else
                 "Facebook" if base == "facebook" else
                 "BlogCatalog" if base == "blogcatalog" else base)
    graph_type = cfg.get("graph", "original")
    print("\n=== %s | graph=%s tmpl=%s gamma=%g tk=%s pca=%s sub=%s ===" % (
        ds, graph_type, cfg["tmpl"], cfg["gamma"], cfg["tk"], cfg["pca"],
        cfg["sub"]), flush=True)
    data = load_data(load_name)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)

    entries = prepare_model_entries(
        data,
        graph_types=[graph_type],
        template_gammas=[cfg["gamma"]],
        pca_dim=cfg["pca"],
        truncated_k=cfg["tk"],
        modeled_subspace=cfg["sub"],
        template_type=cfg["tmpl"],
        device=device,
    )
    if impose_fit:
        entry = entries[0]
        fit = {"rho": float(cfg["rho"]), "kappa": float(cfg["kappa"]),
               "ml": float("nan"), "entry_key": entry["key"]}
        print("  fit imposed from Table 6: rho=%g kappa=%g" % (
            cfg["rho"], cfg["kappa"]), flush=True)
    else:
        fit = fit_best_entry(entries, [np.inf], PAPER_KAPPAS, use_gamma_jacobian=False)
        entry = entry_for_fit(entries, fit)
        print("  fit: rho=%.3f kappa=%g (Table 6: rho=%g kappa=%g)" % (
            fit["rho"], fit["kappa"], cfg["rho"], cfg["kappa"]), flush=True)

    q = q_for_fit(entry, fit)
    horizons = ALL_GAMMAS + [np.inf]
    bank = horizon_score_bank(entry, q, horizons)
    y_eval = y[mask]

    energy = family_report("J", bank, y_eval, mask)
    ratio = family_report("R", bank, y_eval, mask)

    lam_pairs = sorted(set(TEXT_LAM_GRID) | set(ERA_LAM_GRID))
    lam_per_score = scored(lam_grid_bank(entry, q, lam_pairs), y_eval, mask)
    energy_lam = lam_family_report(
        "C", lam_per_score, energy["per_score"]["J@inf"], "J@inf")
    ratio_lam = lam_family_report(
        "CR", lam_per_score, ratio["per_score"]["R@inf"], "R@inf")

    t8 = TABLE8.get(cfg.get("load", ds), (float("nan"), float("nan")))
    report_rows = (
        ("C/J*", energy, ("text_grid", "pipeline_grid"), t8[0]),
        ("CR/R", ratio, ("text_grid", "pipeline_grid"), t8[1]),
        ("Clam", energy_lam, ("text_grid", "era_grid"), t8[0]),
        ("CRlam", ratio_lam, ("text_grid", "era_grid"), t8[1]),
    )
    for label, fam, grids, t8val in report_rows:
        for grid in grids:
            pool = fam.get(grid)
            if not pool:
                continue
            print("  %-5s %-13s selected(%s)=%.2f  oracle(%s)=%.2f  "
                  "limit=%.2f  joint_pick=%s  [Table8 limit=%.1f]" % (
                      label, grid,
                      pool["selected_finite"]["name"],
                      100 * pool["selected_finite"]["auc"],
                      pool["oracle_finite"]["name"],
                      100 * pool["oracle_finite"]["auc"],
                      100 * pool["limit"]["auc"],
                      pool["joint_nullks_pick"]["name"],
                      t8val), flush=True)

    return {
        "dataset": ds,
        "config": {**{k: cfg[k] for k in ("tmpl", "gamma", "tk", "pca", "sub")},
                   "graph": graph_type},
        "fit": {"rho": float(fit["rho"]), "kappa": float(fit["kappa"]),
                "ml": float(fit["ml"]),
                "table6_rho": cfg["rho"], "table6_kappa": cfg["kappa"]},
        "table8_limit": t8,
        "energy": energy,
        "ratio": ratio,
        "energy_lam": energy_lam,
        "ratio_lam": ratio_lam,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(TABLE6))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--impose-fit", action="store_true",
                        help="Use Table 6 (rho, kappa) directly instead of refitting.")
    parser.add_argument("--out", default="results/horizon_study/matched_single_score.json")
    args = parser.parse_args()

    results = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        try:
            results[ds] = run_dataset(ds, args.device, impose_fit=args.impose_fit)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[ds] = {"error": str(exc)[:500]}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print("\nSaved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
