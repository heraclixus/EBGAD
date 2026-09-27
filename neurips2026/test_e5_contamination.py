"""E5: robustness of the EB fit to anomaly contamination.

Inject additional contextual anomalies (feature swap with a random distant
node, the standard injection of the DOMINANT benchmark line) into real
datasets at increasing rates, refit EB on the contaminated graph, and report
the drift of the fitted prior (rho, kappa) and the label-free selected AUROC
against the union of original + injected anomalies.
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
    q_for_fit,
)

ERA_HORIZONS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, np.inf]
ERA_KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5,
              1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

DS_GAMMAS = {"weibo": [0.2, 0.5, 0.7], "reddit": [0.7, 1.0, 2.0]}


def inject(data, extra_frac: float, seed: int):
    """Feature-swap injection on a copy of data; returns (data, y_augmented)."""
    rng = np.random.default_rng(seed)
    y = data.y.detach().cpu().numpy().copy()
    normal = np.where(y == 0)[0]
    n_inject = int(round(extra_frac * len(y)))
    if n_inject == 0:
        return data, y
    chosen = rng.choice(normal, size=n_inject, replace=False)
    x = data.x.clone()
    n = x.shape[0]
    for i in chosen:
        # swap with a far-away random node's features (contextual anomaly)
        j = int(rng.integers(0, n))
        x[i] = data.x[j]
    y_aug = y.copy()
    y_aug[chosen] = 1
    out = data.clone()
    out.x = x
    return out, y_aug


def run(ds: str, extra_frac: float, seed: int, device: str) -> dict:
    base = load_data(ds)
    data, y_aug = inject(base, extra_frac, seed)
    mask = (y_aug >= 0) & (y_aug <= 1)
    entries = prepare_model_entries(
        data, graph_types=["original"], template_gammas=DS_GAMMAS[ds],
        pca_dim=64 if data.x.shape[1] > 100 else None,
        truncated_k=None, modeled_subspace="drop_nullspace",
        template_type="low", device=device,
    )
    fit = fit_best_entry(entries, ERA_HORIZONS, ERA_KAPPAS, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    q = q_for_fit(entry, fit)
    bank = horizon_score_bank(entry, q, ERA_HORIZONS)
    refs = node_unsupervised_references(data, entry["delta"])
    y_orig = base.y.detach().cpu().numpy()
    injected = (y_aug == 1) & (y_orig == 0)
    # decomposed evaluations: selection is label-free, so the selected score
    # is identical across the three calls; only the evaluation labels change
    summary = evaluate_score_bank(y_aug, mask, bank, references=refs)
    mask_orig = mask & ~injected  # original anomalies vs untouched normals
    sum_orig = evaluate_score_bank(y_orig, mask_orig, bank, references=refs)
    y_inj = injected.astype(np.int64)
    mask_inj = mask & (y_orig == 0)  # injected vs untouched normals
    sum_inj = (evaluate_score_bank(y_inj, mask_inj, bank, references=refs)
               if injected.any() else {})
    row = {
        "dataset": ds, "extra_frac": extra_frac,
        "total_contamination": float((y_aug[mask] > 0).mean()),
        "rho": fit["rho"], "kappa": fit["kappa"], "entry": fit["entry_key"],
        "auc_ks_selected": summary.get("auc_ks_selected"),
        "score_ks": summary.get("score_ks"),
        "auc_oracle": summary.get("auc_oracle"),
        "auc_original": sum_orig.get("auc_ks_selected"),
        "auc_injected": sum_inj.get("auc_ks_selected"),
    }
    print("  %-8s eps=%.2f cont=%.3f rho=%.3f kappa=%-6g sel=%s union=%.1f%% orig=%.1f%% inj=%s oracle=%.1f%%" % (
        ds, extra_frac, row["total_contamination"], row["rho"], row["kappa"],
        row["score_ks"], 100 * (row["auc_ks_selected"] or 0),
        100 * (row["auc_original"] or 0),
        ("%.1f%%" % (100 * row["auc_injected"]) if row["auc_injected"] else "n/a"),
        100 * (row["auc_oracle"] or 0)), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="weibo,reddit")
    ap.add_argument("--fracs", default="0,0.05,0.1,0.2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/e5/e5_contamination.json")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets.split(","):
        for f in [float(x) for x in args.fracs.split(",")]:
            try:
                rows.append(run(ds.strip(), f, args.seed, args.device))
            except Exception as exc:
                print("  ERROR %s eps=%s: %s" % (ds, f, str(exc)[:200]), flush=True)
                rows.append({"dataset": ds, "extra_frac": f, "error": str(exc)[:300]})
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(rows, fh, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
