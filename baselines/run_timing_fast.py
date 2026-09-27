"""Fast timing estimation: run 1 epoch, multiply by default epoch count.

For each (method, dataset), runs 1 epoch + inference, then estimates
full training time as: time_1epoch * default_epochs + inference_time.

Default epochs from PyGOD source:
  DOMINANT: 100, AnomalyDAE: 100, CONAD: 100, CoLA: 50, ANOMALOUS: 100
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data

DEFAULT_EPOCHS = {
    "DOMINANT": 100, "AnomalyDAE": 100, "CONAD": 100,
    "CoLA": 50, "ANOMALOUS": 100,
}

MISSING = [
    ("CONAD", "t_finance"),
    ("CoLA", "t_finance"),
    ("ANOMALOUS", "elliptic"),
    ("AnomalyDAE", "elliptic"),
    ("CONAD", "elliptic"),
    ("ANOMALOUS", "elliptic_plus_plus"),
    ("AnomalyDAE", "elliptic_plus_plus"),
    ("CONAD", "elliptic_plus_plus"),
    ("CONAD", "dgraph"),
]


def time_one_epoch(method, data, device="cpu"):
    """Run 1 epoch, return (time_per_epoch, inference_time, auc)."""
    from pygod.detector import DOMINANT, AnomalyDAE, CONAD, CoLA, ANOMALOUS
    detector_map = {
        "DOMINANT": DOMINANT, "AnomalyDAE": AnomalyDAE,
        "CONAD": CONAD, "CoLA": CoLA, "ANOMALOUS": ANOMALOUS,
    }
    gpu = 0 if device.startswith("cuda") else -1
    DetectorClass = detector_map[method]

    # 1 epoch training
    t0 = time.time()
    detector = DetectorClass(epoch=1, gpu=gpu, verbose=0)
    detector.fit(data)
    t_train = time.time() - t0

    # Inference
    t1 = time.time()
    scores = detector.decision_score_
    t_infer = time.time() - t1

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.nan_to_num(scores, nan=0.0)

    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    try:
        auc = roc_auc_score(y[mask], scores[mask])
    except:
        auc = 0.0

    return t_train, t_infer, auc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()

    tasks = MISSING if args.index < 0 else [MISSING[args.index]]
    out_file = "results/timing_fast.json"
    os.makedirs("results", exist_ok=True)

    results = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)

    for method, ds in tasks:
        key = f"{method}_{ds}"
        if key in results and "estimated_total" in results[key]:
            print(f"  {key}: already done, skipping")
            continue

        print(f"  {method} on {ds} (1 epoch)...", end="", flush=True)
        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)
        try:
            data = load_data(load_name)
            t_train, t_infer, auc = time_one_epoch(method, data, args.device)
            n_epochs = DEFAULT_EPOCHS.get(method, 100)
            estimated = t_train * n_epochs + t_infer

            results[key] = {
                "method": method, "dataset": ds,
                "time_1epoch": round(t_train, 2),
                "time_infer": round(t_infer, 2),
                "default_epochs": n_epochs,
                "estimated_total": round(estimated, 1),
                "auc_1epoch": round(auc, 4),
            }
            print(f" 1ep={t_train:.1f}s est_total={estimated:.0f}s", flush=True)
        except (torch.cuda.OutOfMemoryError, MemoryError):
            results[key] = {"method": method, "dataset": ds, "error": "OOM"}
            print(f" OOM", flush=True)
        except Exception as e:
            results[key] = {"method": method, "dataset": ds,
                            "error": str(e)[:200]}
            print(f" ERROR: {str(e)[:60]}", flush=True)

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults:")
    for k, v in sorted(results.items()):
        if "estimated_total" in v:
            print(f"  {k}: ~{v['estimated_total']:.0f}s "
                  f"({v['time_1epoch']:.1f}s/ep x {v['default_epochs']}ep)")
        else:
            print(f"  {k}: {v.get('error', '?')}")


if __name__ == "__main__":
    main()
