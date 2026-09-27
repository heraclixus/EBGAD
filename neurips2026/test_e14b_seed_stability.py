"""E14b: label-free selection stability under eigensolver randomness.

GraphEigenTruncated uses torch.lobpcg with an unseeded random initial block
(bridge/graph_ops.py), so truncated-eigenspace runs differ slightly between
executions. E14 showed the GOU rows stable across draws while the
uncalibrated heat selection on Elliptic++ flipped from 62.3 (E2 draw,
J|t=16) to 76.2 (E14 draw, R|t=4). This script quantifies that: for each
financial-trio dataset, rerun the isolation protocol under 5 seeds (seeding
torch + numpy before entry preparation so the lobpcg draw differs
controllably) and record every family's label-free selected AUROC and pick,
plus the calibrated variants of E14.

Protocol restriction: the template gamma is fixed to the value the E14/E2
fits chose (elliptic 1.0, elliptic++ 1.0, t_finance 0.1) so each seed costs
one eigendecomposition; (rho, kappa) are refit per seed.
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
import torch

from data_utils import load_data
from test_dynamic_score_bank import (
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    horizon_score_bank,
    node_unsupervised_references,
    prepare_model_entries,
)
from test_e2_isolation_banks import (
    DS_CONFIG,
    ERA_HORIZONS,
    ERA_KAPPAS,
    family_banks,
    generic_bank,
    transport_scores,
)
from test_e14_calibrated_generics import fit_generic_family

FIT_GAMMA = {"elliptic": 1.0, "elliptic_plus_plus": 1.0, "t_finance": 0.1}


def run_seed(ds: str, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = DS_CONFIG[ds]
    data = load_data(ds)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    pca_dim = cfg.get("pca", 64 if data.x.shape[1] > 100 else None)
    entries = prepare_model_entries(
        data, graph_types=["original"], template_gammas=[FIT_GAMMA[ds]],
        pca_dim=pca_dim, truncated_k=cfg["tk"], modeled_subspace=cfg["sub"],
        template_type=cfg["tmpl"], device=device,
    )
    fit = fit_best_entry(entries, ERA_HORIZONS, ERA_KAPPAS, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    references = node_unsupervised_references(data, entry["delta"])

    row = {"seed": seed, "rho": fit["rho"], "kappa": fit["kappa"], "families": {}}
    banks = family_banks(entry, fit)
    for family in ("heat", "poly"):
        gfit = fit_generic_family(entries, family)
        q_cal = gfit["q"]
        banks["%s_cal_strict" % family] = generic_bank(entry, {gfit["config"]: q_cal})
        dyn = horizon_score_bank(entry, q_cal, ERA_HORIZONS)
        dyn.update(transport_scores(entry, q_cal))
        banks["%s_cal_dynamics" % family] = dyn
        row["families"]["%s_cal_config" % family] = gfit["config"]
    for family, bank in banks.items():
        summary = evaluate_score_bank(y, mask, bank, references=references)
        row["families"][family] = {
            "auc_ks_selected": summary.get("auc_ks_selected"),
            "score_ks": summary.get("score_ks"),
            "auc_oracle": summary.get("auc_oracle"),
        }
        print("  seed=%d %-18s sel=%5.1f%% (%s) oracle=%5.1f%%" % (
            seed, family, 100 * (summary.get("auc_ks_selected") or 0),
            summary.get("score_ks"), 100 * (summary.get("auc_oracle") or 0)),
            flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="elliptic_plus_plus,elliptic,t_finance")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/e14/e14b_seeds.json")
    args = ap.parse_args()

    results: dict = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        print("\n=== %s ===" % ds, flush=True)
        results[ds] = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            try:
                results[ds].append(run_seed(ds, seed, args.device))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                results[ds].append({"seed": seed, "error": str(exc)[:300]})
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=str)
    print("\nSaved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
