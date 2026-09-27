"""Run missing PyGOD baselines locally (sequential, CPU).

Fills remaining \tbd entries in runtime table.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data
from eval_utils import mask_labeled

# Remaining missing entries — try one at a time, skip on OOM
MISSING = [
    # ANOMALOUS on large datasets (doesn't use NeighborSampler, might work)
    ("ANOMALOUS", "elliptic"),
    ("ANOMALOUS", "elliptic_plus_plus"),
    # CONAD on larger datasets
    ("CONAD", "t_finance"),
    ("CONAD", "elliptic"),
    ("CONAD", "elliptic_plus_plus"),
    # CoLA on t_finance
    ("CoLA", "t_finance"),
    # AnomalyDAE on large datasets
    ("AnomalyDAE", "elliptic"),
    ("AnomalyDAE", "elliptic_plus_plus"),
]

OUT_FILE = "results/missing_baselines_local.json"


def run_one(method, ds):
    from pygod.detector import (
        DOMINANT as D, AnomalyDAE as A, CONAD as C,
        CoLA as Co, ANOMALOUS as AN,
    )
    detector_map = {
        "DOMINANT": D, "AnomalyDAE": A, "CONAD": C,
        "CoLA": Co, "ANOMALOUS": AN,
    }

    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)

    DetClass = detector_map[method]
    t0 = time.time()
    detector = DetClass()
    detector.fit(data)
    scores = detector.decision_score_
    elapsed = time.time() - t0

    y, scores_masked = mask_labeled(data.y, torch.tensor(scores))
    auc = roc_auc_score(y.numpy(), scores_masked.numpy()) if y is not None else 0.0

    return auc, elapsed


def main():
    results = {}
    if os.path.exists(OUT_FILE):
        results = json.load(open(OUT_FILE))

    for method, ds in MISSING:
        key = f"{method}_{ds}"
        if key in results and "elapsed_s" in results[key]:
            print(f"  SKIP {key} (already done: {results[key]['elapsed_s']:.0f}s)")
            continue

        print(f"  {method} on {ds}...", end="", flush=True)
        try:
            auc, elapsed = run_one(method, ds)
            results[key] = {"method": method, "dataset": ds,
                            "auc": float(auc), "elapsed_s": float(elapsed)}
            print(f" AUC={auc:.3f} time={elapsed:.0f}s", flush=True)
        except Exception as e:
            results[key] = {"method": method, "dataset": ds,
                            "error": str(e)[:150]}
            print(f" ERROR: {str(e)[:60]}", flush=True)

        # Save after each
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {OUT_FILE}")
    ok = {k: v for k, v in results.items() if "elapsed_s" in v}
    print(f"Successful: {len(ok)}/{len(results)}")
    for k, v in sorted(ok.items()):
        print(f"  {k:30s} {v['elapsed_s']:8.0f}s  auc={v['auc']:.3f}")


if __name__ == "__main__":
    main()
