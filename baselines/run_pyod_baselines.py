"""
Non-graph anomaly detection baselines via PyOD.

Runs LOF, ECOD, and DIF (Deep Isolation Forest) on node features only
(ignoring graph structure). Provides a lower bound that any graph-aware
method should beat.

Usage::

    python run_pyod_baselines.py                           # all datasets
    python run_pyod_baselines.py --datasets disney weibo   # specific datasets
    python run_pyod_baselines.py --methods LOF ECOD        # specific methods
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import os
import time
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import auc as sk_auc, precision_recall_curve


def run_pyod_baseline(X: np.ndarray, y: np.ndarray, method: str) -> dict:
    """Run a single PyOD detector and return metrics."""
    if method == "LOF":
        from pyod.models.lof import LOF
        clf = LOF(n_neighbors=20)
    elif method == "ECOD":
        from pyod.models.ecod import ECOD
        clf = ECOD()
    elif method == "DIF":
        from pyod.models.dif import DIF
        clf = DIF()
    elif method == "IForest":
        from pyod.models.iforest import IForest
        clf = IForest(n_estimators=100)
    else:
        raise ValueError(f"Unknown method: {method}")

    clf.fit(X)
    scores = clf.decision_scores_

    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    p, r, _ = precision_recall_curve(y, scores)
    auprc = float(sk_auc(r, p))

    return {"auc": auc, "ap": ap, "auprc": auprc}


def main():
    parser = argparse.ArgumentParser(description="PyOD baselines on GAD datasets")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names (default: all)")
    parser.add_argument("--methods", nargs="+", default=["ECOD"],
                        help="PyOD methods to run (default: LOF ECOD DIF)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials for stochastic methods (DIF, IForest)")
    parser.add_argument("--output", type=str, default="results/pyod_baselines.jsonl")
    args = parser.parse_args()

    from data_utils import load_data

    all_datasets = [
        "disney", "enron", "weibo", "reddit",
        "elliptic", "elliptic_plus_plus", "dgraph",
        "Amazon", "YelpChi", "BlogCatalog", "Facebook", "ACM",
        "t_finance",
    ]
    datasets = args.datasets or all_datasets

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    deterministic_methods = {"LOF", "ECOD"}

    results = []
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        try:
            data = load_data(ds)
        except Exception as e:
            print(f"  SKIP: failed to load: {e}")
            continue

        X = data.x.numpy().astype(np.float64)
        y = data.y.numpy().flatten()
        mask = (y >= 0) & (y <= 1)
        if mask.sum() == 0:
            print(f"  SKIP: no labeled nodes")
            continue
        X = X[mask]
        y_binary = (y[mask] > 0).astype(int)
        n, d = X.shape
        print(f"  n={n}, d={d}, anomalies={y_binary.sum()} ({100*y_binary.mean():.1f}%)")

        for method in args.methods:
            print(f"\n  --- {method} ---")
            n_trials = 1 if method in deterministic_methods else args.trials

            trial_results = []
            t0 = time.time()
            for trial in range(n_trials):
                try:
                    r = run_pyod_baseline(X, y_binary, method)
                    trial_results.append(r)
                    if n_trials > 1:
                        print(f"    trial {trial+1}/{n_trials}: AUC={r['auc']:.4f}")
                except Exception as e:
                    print(f"    trial {trial+1} FAILED: {e}")
            elapsed = time.time() - t0

            if not trial_results:
                print(f"    ALL TRIALS FAILED")
                continue

            aucs = np.array([r["auc"] for r in trial_results])
            auprcs = np.array([r["auprc"] for r in trial_results])
            aps = np.array([r["ap"] for r in trial_results])

            result = {
                "model": method,
                "params": {"_dataset": ds},
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std()),
                "ap_mean": float(aps.mean()),
                "auprc_mean": float(auprcs.mean()),
                "auprc_std": float(auprcs.std()),
                "elapsed_s": elapsed,
                "num_trials": len(trial_results),
            }
            results.append(result)

            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"    {method} on {ds}: AUROC={result['auc_mean']:.4f}±{result['auc_std']:.4f}  "
                  f"AUPRC={result['auprc_mean']:.4f}±{result['auprc_std']:.4f}  ({elapsed:.1f}s)")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY (AUROC)")
    print(f"{'='*80}")
    methods_seen = sorted(set(r["model"] for r in results))
    ds_seen = sorted(set(r["params"]["_dataset"] for r in results),
                     key=lambda d: all_datasets.index(d) if d in all_datasets else 999)

    print(f"{'':15s}", end="")
    for m in methods_seen:
        print(f"  {m:>12s}", end="")
    print()

    lookup = {(r["model"], r["params"]["_dataset"]): r for r in results}
    for ds in ds_seen:
        print(f"{ds:15s}", end="")
        for m in methods_seen:
            key = (m, ds)
            if key in lookup:
                r = lookup[key]
                print(f"  {r['auc_mean']:>5.3f}±{r['auc_std']:.2f}", end="")
            else:
                print(f"  {'—':>12s}", end="")
        print()


if __name__ == "__main__":
    main()
