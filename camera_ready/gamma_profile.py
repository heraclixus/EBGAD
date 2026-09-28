"""Template bandwidth profile: for every bandwidth of a dataset's grid, the EB fit, its profile
likelihood and the AUROC of the two equilibrium scores. The pipeline selects the bandwidth with the
largest likelihood; the AUROC columns are diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
from data_utils import load_data  # noqa: E402
from test_dynamic_score_bank import (  # noqa: E402
    compute_graph_stats, density_adjusted_template_gammas, fit_best_entry, horizon_score_bank,
    prepare_model_entries, q_for_fit, safe_auc,
)
from test_multiscale_gou_poe import ks_null_deviation  # noqa: E402

KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
NAMES = {"yelpchi": "YelpChi", "facebook": "Facebook"}


def truncation(n):
    return None if n < 20_000 else (128 if n >= 1_000_000 else 200 if n >= 500_000 else 300 if n >= 100_000 else 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-gammas", default="1.0")
    args = ap.parse_args()
    ds = args.dataset
    data = load_data(NAMES.get(ds, ds))
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    h, de = compute_graph_stats(data)
    grid = density_adjusted_template_gammas(h, de)
    gammas = sorted(set(float(g) for g in grid) | set(float(g) for g in args.extra_gammas.split(",") if g))
    n = data.x.shape[0]
    k = truncation(n)
    entries = prepare_model_entries(
        data, graph_types=["original"], template_gammas=gammas,
        pca_dim=64 if data.x.shape[1] > 100 else None, truncated_k=k,
        modeled_subspace="legacy" if k is not None else "drop_nullspace", template_type="low", device="cpu")
    rows = []
    for e in entries:
        fit = fit_best_entry([e], [np.inf], KAPPAS, False)
        bank = horizon_score_bank(e, q_for_fit(e, fit), [np.inf])
        j, r = np.nan_to_num(np.asarray(bank["J@inf"], np.float64)), np.nan_to_num(np.asarray(bank["R@inf"], np.float64))
        rows.append({"gamma": float(e["template_gamma"]), "in_grid": float(e["template_gamma"]) in [float(g) for g in grid],
                     "ml": float(fit["ml"]), "rho": float(fit["rho"]), "kappa": float(fit["kappa"]),
                     "auc_J": 100 * safe_auc(y[mask], j[mask]), "auc_R": 100 * safe_auc(y[mask], r[mask]),
                     "ks_J": float(ks_null_deviation(j)), "ks_R": float(ks_null_deviation(r))})
        print(ds, rows[-1], flush=True)
    grid_rows = [x for x in rows if x["in_grid"]]
    sel = max(grid_rows, key=lambda x: x["ml"])
    json.dump({"dataset": ds, "grid": [float(g) for g in grid], "selected_gamma": sel["gamma"], "rows": rows},
              open(args.out, "w"), indent=1)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
