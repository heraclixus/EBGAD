"""E16: contamination robustness at an interior EB fit (Amazon).

The E5 contamination study (Weibo, Reddit) showed the fitted prior identical
or one grid step away up to ~24% contamination, but both fits sit at the
rho clamp boundary (rho = 1.0), so a skeptic can attribute the stability to
the clamp. Table 1's Amazon fit is interior (rho = 0.46, kappa = 0.5,
gamma = 5.0; tab:ebgad_hyperparams), which the E5-era gamma grid could not
reach (it stopped at gamma = 2). This script reruns the E5 protocol on
Amazon with the template grid extended to include the paper's optimum, so
the baseline fit is interior and any contamination-induced drift of
(rho, kappa, gamma) is unconstrained in both directions.

Also records the cosine feature homophily h of the contaminated graph:
Amazon's family routing fires on h >= 0.50 (Rule 5), so h(eps) reports the
routing trigger's margin under contamination.
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
    compute_graph_stats,
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    horizon_score_bank,
    node_unsupervised_references,
    prepare_model_entries,
    q_for_fit,
)
from test_e5_contamination import inject

ERA_HORIZONS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, np.inf]
ERA_KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5,
              1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
# E2's Amazon grid plus the paper-grid upper range so the Table-1 optimum
# (gamma = 5) is reachable and the baseline fit is interior.
GAMMAS = [0.5, 0.7, 1.0, 2.0, 5.0]


def run(extra_frac: float, seed: int, device: str, subspace: str,
        gamma_jacobian: bool = False) -> dict:
    base = load_data("amazon")
    data, y_aug = inject(base, extra_frac, seed)
    mask = (y_aug >= 0) & (y_aug <= 1)
    h, density = compute_graph_stats(data)
    entries = prepare_model_entries(
        data, graph_types=["original"], template_gammas=GAMMAS,
        pca_dim=64 if data.x.shape[1] > 100 else None,
        truncated_k=None, modeled_subspace=subspace,
        template_type="low", device=device,
    )
    fit = fit_best_entry(entries, ERA_HORIZONS, ERA_KAPPAS,
                         use_gamma_jacobian=gamma_jacobian)
    entry = entry_for_fit(entries, fit)
    q = q_for_fit(entry, fit)
    bank = horizon_score_bank(entry, q, ERA_HORIZONS)
    refs = node_unsupervised_references(data, entry["delta"])
    y_orig = base.y.detach().cpu().numpy()
    injected = (y_aug == 1) & (y_orig == 0)
    summary = evaluate_score_bank(y_aug, mask, bank, references=refs)
    mask_orig = mask & ~injected
    sum_orig = evaluate_score_bank(y_orig, mask_orig, bank, references=refs)
    y_inj = injected.astype(np.int64)
    mask_inj = mask & (y_orig == 0)
    sum_inj = (evaluate_score_bank(y_inj, mask_inj, bank, references=refs)
               if injected.any() else {})
    row = {
        "dataset": "amazon", "extra_frac": extra_frac,
        "total_contamination": float((y_aug[mask] > 0).mean()),
        "rho": fit["rho"], "kappa": fit["kappa"], "entry": fit["entry_key"],
        "homophily_cos": float(h), "rule5_trigger_h_ge_0.50": bool(h >= 0.50),
        "auc_ks_selected": summary.get("auc_ks_selected"),
        "score_ks": summary.get("score_ks"),
        "auc_oracle": summary.get("auc_oracle"),
        "auc_original": sum_orig.get("auc_ks_selected"),
        "auc_injected": sum_inj.get("auc_ks_selected"),
    }
    print("  eps=%.2f cont=%.3f rho=%.3f kappa=%-6g entry=%s h=%.3f sel=%s union=%.1f%% orig=%.1f%% inj=%s oracle=%.1f%%" % (
        extra_frac, row["total_contamination"], row["rho"], row["kappa"],
        row["entry"], h, row["score_ks"], 100 * (row["auc_ks_selected"] or 0),
        100 * (row["auc_original"] or 0),
        ("%.1f%%" % (100 * row["auc_injected"]) if row["auc_injected"] else "n/a"),
        100 * (row["auc_oracle"] or 0)), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0,0.05,0.1,0.2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--subspace", default="drop_nullspace")
    ap.add_argument("--gamma-jacobian", action="store_true")
    ap.add_argument("--out", default="results/e16/e16_amazon.json")
    args = ap.parse_args()

    rows = []
    for f in [float(x) for x in args.fracs.split(",")]:
        try:
            rows.append(run(f, args.seed, args.device, args.subspace,
                            args.gamma_jacobian))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rows.append({"extra_frac": f, "error": str(exc)[:300]})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
