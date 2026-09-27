"""Fill in missing timing entries for the runtime table (Appendix B).

Current \tbd entries:
  T-Finance: CONAD, CoLA, TAM
  Elliptic:  ANOM., AnomDAE, CONAD
  Elliptic++: ANOM., AnomDAE, CONAD
  DGraph:    CONAD

Also verifies OOM entries on DGraph: ANOM., DOMINANT, AnomDAE, DiffGAD
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data

MISSING = [
    # T-Finance (39K nodes)
    ("CONAD", "t_finance"),
    ("CoLA", "t_finance"),
    ("TAM", "t_finance"),
    # Elliptic (203K nodes)
    ("ANOMALOUS", "elliptic"),
    ("AnomalyDAE", "elliptic"),
    ("CONAD", "elliptic"),
    # Elliptic++ (203K nodes)
    ("ANOMALOUS", "elliptic_plus_plus"),
    ("AnomalyDAE", "elliptic_plus_plus"),
    ("CONAD", "elliptic_plus_plus"),
    # DGraph (3.7M nodes) -- likely OOM for most methods
    ("CONAD", "dgraph"),
]


def run_pygod_method(method, data, device="cpu"):
    from pygod.detector import (
        DOMINANT, AnomalyDAE, CONAD, CoLA, ANOMALOUS,
    )
    detector_map = {
        "DOMINANT": DOMINANT,
        "AnomalyDAE": AnomalyDAE,
        "CONAD": CONAD,
        "CoLA": CoLA,
        "ANOMALOUS": ANOMALOUS,
    }
    gpu = 0 if device.startswith("cuda") else -1
    DetectorClass = detector_map[method]

    t0 = time.time()
    detector = DetectorClass(gpu=gpu)
    detector.fit(data)
    scores = detector.decision_score_
    elapsed = time.time() - t0

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.nan_to_num(scores, nan=0.0)

    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    auc = roc_auc_score(y[mask], scores[mask])
    return auc, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index", type=int, default=-1,
                        help="Run only this index (-1 = all)")
    args = parser.parse_args()

    tasks = MISSING if args.index < 0 else [MISSING[args.index]]
    out_file = "results/timing_missing.json"
    os.makedirs("results", exist_ok=True)

    # Load existing results
    results = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)

    for method, ds in tasks:
        key = f"{method}_{ds}"
        if key in results and "elapsed_s" in results[key]:
            print(f"  {key}: already done ({results[key]['elapsed_s']:.1f}s), skipping")
            continue

        print(f"  {method} on {ds}...", end="", flush=True)
        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)
        try:
            data = load_data(load_name)
            if method == "TAM":
                from run_tam import run_tam_single
                t0 = time.time()
                auc = run_tam_single(data, device=args.device)
                elapsed = time.time() - t0
            else:
                auc, elapsed = run_pygod_method(method, data, args.device)
            results[key] = {"method": method, "dataset": ds,
                            "auc": float(auc), "elapsed_s": float(elapsed)}
            print(f" AUC={auc:.3f} time={elapsed:.1f}s", flush=True)
        except torch.cuda.OutOfMemoryError:
            results[key] = {"method": method, "dataset": ds, "error": "OOM"}
            print(f" OOM", flush=True)
        except MemoryError:
            results[key] = {"method": method, "dataset": ds, "error": "OOM"}
            print(f" OOM (CPU)", flush=True)
        except Exception as e:
            results[key] = {"method": method, "dataset": ds,
                            "error": str(e)[:200]}
            print(f" ERROR: {str(e)[:80]}", flush=True)

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_file}")
    for k, v in sorted(results.items()):
        if "elapsed_s" in v:
            print(f"  {k}: {v['elapsed_s']:.1f}s (AUC={v['auc']:.3f})")
        else:
            print(f"  {k}: {v.get('error', '?')}")


if __name__ == "__main__":
    main()
