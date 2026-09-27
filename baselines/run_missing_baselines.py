"""Run missing PyGOD baseline timing experiments.

Fills in \tbd entries in the runtime table (Appendix B).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data
from eval_utils import mask_labeled

# What's missing from the table
MISSING = [
    # DOMINANT (all missing except Facebook, Elliptic, Elliptic++)
    ("DOMINANT", "enron"), ("DOMINANT", "weibo"), ("DOMINANT", "reddit"),
    ("DOMINANT", "amazon"), ("DOMINANT", "yelpchi"),
    ("DOMINANT", "blogcatalog"), ("DOMINANT", "acm"), ("DOMINANT", "t_finance"),
    # AnomalyDAE (missing: enron, weibo, reddit, elliptic, elliptic++)
    ("AnomalyDAE", "enron"), ("AnomalyDAE", "weibo"), ("AnomalyDAE", "reddit"),
    ("AnomalyDAE", "elliptic"), ("AnomalyDAE", "elliptic_plus_plus"),
    # CONAD (missing: enron, weibo, reddit, t_finance, elliptic, elliptic++)
    ("CONAD", "enron"), ("CONAD", "weibo"), ("CONAD", "reddit"),
    ("CONAD", "t_finance"),
    ("CONAD", "elliptic"), ("CONAD", "elliptic_plus_plus"),
    # CoLA (missing: reddit, amazon, yelpchi, blogcatalog, facebook, acm, t_finance)
    ("CoLA", "reddit"), ("CoLA", "amazon"), ("CoLA", "yelpchi"),
    ("CoLA", "blogcatalog"), ("CoLA", "facebook"), ("CoLA", "acm"),
    ("CoLA", "t_finance"),
    # ANOMALOUS (missing: blogcatalog, acm, elliptic, elliptic++)
    ("ANOMALOUS", "blogcatalog"), ("ANOMALOUS", "acm"),
    ("ANOMALOUS", "elliptic"), ("ANOMALOUS", "elliptic_plus_plus"),
]


def run_pygod_method(method, data, device="cuda", timeout=7200):
    """Run a PyGOD method and return (auc, elapsed_s)."""
    from pygod.detector import (
        DOMINANT as DOMINANTDetector,
        AnomalyDAE as AnomalyDAEDetector,
        CONAD as CONADDetector,
        CoLA as CoLADetector,
        ANOMALOUS as ANOMALOUSDetector,
    )

    detector_map = {
        "DOMINANT": DOMINANTDetector,
        "AnomalyDAE": AnomalyDAEDetector,
        "CONAD": CONADDetector,
        "CoLA": CoLADetector,
        "ANOMALOUS": ANOMALOUSDetector,
    }

    if method not in detector_map:
        raise ValueError(f"Unknown method: {method}")

    DetectorClass = detector_map[method]

    t0 = time.time()
    try:
        detector = DetectorClass()
        detector.fit(data)
        scores = detector.decision_score_
        elapsed = time.time() - t0

        y, scores_masked = mask_labeled(data.y, torch.tensor(scores))
        if y is not None:
            auc = roc_auc_score(y.numpy(), scores_masked.numpy())
        else:
            auc = 0.0
        return auc, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        raise


def run_tam(data, device="cuda"):
    """Run TAM baseline."""
    from run_tam import run_tam_single
    t0 = time.time()
    auc = run_tam_single(data, device=device)
    elapsed = time.time() - t0
    return auc, elapsed


def main(device="cuda", chunk=0, n_chunks=1):
    out_file = f"results/missing_baselines_{chunk}.json"

    my_tasks = [t for i, t in enumerate(MISSING) if i % n_chunks == chunk]
    print(f"Total: {len(MISSING)}, chunk {chunk}/{n_chunks}: {len(my_tasks)}",
          flush=True)

    results = {}
    for method, ds in my_tasks:
        key = f"{method}_{ds}"
        print(f"  {method} on {ds}...", end="", flush=True)

        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)
        try:
            data = load_data(load_name)
            if method == "TAM":
                auc, elapsed = run_tam(data, device)
            else:
                auc, elapsed = run_pygod_method(method, data, device)

            results[key] = {"method": method, "dataset": ds,
                            "auc": float(auc), "elapsed_s": float(elapsed)}
            print(f" AUC={auc:.3f} time={elapsed:.1f}s", flush=True)
        except Exception as e:
            results[key] = {"method": method, "dataset": ds,
                            "error": str(e)[:100]}
            print(f" ERROR: {str(e)[:60]}", flush=True)

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    args = parser.parse_args()
    main(args.device, args.chunk, args.n_chunks)
