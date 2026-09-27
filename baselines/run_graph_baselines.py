"""
Graph-based anomaly detection baselines via PyGOD.

Runs DOMINANT, AnomalyDAE, and CONAD on all datasets using PyGOD's
built-in implementations (PyTorch/PyG, no DGL dependency).

Usage::

    python run_graph_baselines.py                                     # all
    python run_graph_baselines.py --datasets disney weibo Amazon      # specific
    python run_graph_baselines.py --methods DOMINANT AnomalyDAE       # specific
    python run_graph_baselines.py --device cuda                       # GPU
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import auc as sk_auc, precision_recall_curve


def run_pygod_baseline(
    data,
    method: str,
    device: str = "cpu",
    epochs: int = 100,
    hid_dim: int | None = None,
    batch_size: int | None = None,
    num_neigh: int | None = None,
) -> dict:
    """Run a single PyGOD detector and return metrics.

    Memory-reducing params (hid_dim, batch_size, num_neigh) apply to
    DOMINANT, AnomalyDAE, CONAD, CoLA. ANOMALOUS uses CUR decomposition
    and has no such params.
    """
    from pygod.detector import DOMINANT, AnomalyDAE, CONAD, CoLA, ANOMALOUS

    gpu = (int(device.split(":")[1]) if ":" in device else 0) if device.startswith("cuda") else -1   # "cuda:k" -> PyGOD gpu index k

    # Memory-reducing kwargs for GNN-based detectors (PyGOD minibatch docs)
    mem_kw = {}
    if hid_dim is not None:
        mem_kw["hid_dim"] = hid_dim
    if batch_size is not None and batch_size > 0:
        mem_kw["batch_size"] = batch_size
    if num_neigh is not None and num_neigh > 0:
        mem_kw["num_neigh"] = num_neigh

    if method == "DOMINANT":
        clf = DOMINANT(epoch=epochs, gpu=gpu, verbose=0, **mem_kw)
    elif method == "AnomalyDAE":
        # AnomalyDAE uses emb_dim and hid_dim
        ae_kw = {k: v for k, v in mem_kw.items() if k in ("hid_dim", "batch_size", "num_neigh")}
        if "hid_dim" in ae_kw:
            ae_kw["emb_dim"] = ae_kw["hid_dim"]
        clf = AnomalyDAE(epoch=epochs, gpu=gpu, verbose=0, **ae_kw)
    elif method == "CONAD":
        clf = CONAD(epoch=epochs, gpu=gpu, verbose=0, **mem_kw)
    elif method == "CoLA":
        clf = CoLA(epoch=epochs, gpu=gpu, verbose=0, **mem_kw)
    elif method == "ANOMALOUS":
        # ANOMALOUS: CUR-based, no hid_dim/batch_size/num_neigh
        clf = ANOMALOUS(epoch=epochs, gpu=gpu, verbose=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    clf.fit(data)
    scores = clf.decision_score_

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    scores = np.nan_to_num(scores, nan=0.0)

    y = data.y.numpy().flatten()
    mask = (y >= 0) & (y <= 1)
    if mask.sum() == 0:
        return {"auc": 0.0, "ap": 0.0, "auprc": 0.0}
    y_binary = (y[mask] > 0).astype(int)
    scores = scores[mask]

    auc = float(roc_auc_score(y_binary, scores))
    ap = float(average_precision_score(y_binary, scores))
    p, r, _ = precision_recall_curve(y_binary, scores)
    auprc = float(sk_auc(r, p))

    return {"auc": auc, "ap": ap, "auprc": auprc}


def main():
    parser = argparse.ArgumentParser(description="PyGOD graph baselines")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=['ANOMALOUS'])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/graph_baselines.jsonl")
    # Memory-reducing params for GNN-based detectors (DOMINANT, AnomalyDAE, CONAD, CoLA)
    parser.add_argument("--hid-dim", type=int, default=None, help="Hidden dim (default 64); smaller=less RAM")
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size; 0=full batch, 64=memory-efficient")
    parser.add_argument("--num-neigh", type=int, default=None, help="Neighbors per layer; -1=all, 10=memory-efficient")
    args = parser.parse_args()

    from data_utils import load_data

    # all_datasets = [
    #     "disney", "enron", "weibo", "reddit",
    #     "Amazon", "YelpChi", "BlogCatalog", "Facebook", "ACM",
    #     "t_finance",
    # ]
    # all_datasets = [
    #     "Amazon", "YelpChi", "BlogCatalog", "Facebook", "ACM", "t_finance",
    # ]

    all_datasets = [
        "weibo", "t_finance",
        "elliptic", "elliptic_plus_plus", "dgraph",
    ]
    
    datasets = args.datasets or all_datasets

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = []
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        try:
            data = load_data(ds)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        n = data.x.shape[0]
        y = data.y.numpy().flatten()
        n_anom = (y > 0).sum()
        print(f"  n={n}, d={data.x.shape[1]}, anomalies={n_anom} ({100*n_anom/n:.1f}%)")

        for method in args.methods:
            print(f"\n  --- {method} ---")
            trial_results = []
            t0 = time.time()

            for trial in range(args.trials):
                try:
                    r = run_pygod_baseline(
                        data, method, device=args.device, epochs=args.epochs,
                        hid_dim=args.hid_dim, batch_size=args.batch_size, num_neigh=args.num_neigh,
                    )
                    trial_results.append(r)
                    print(f"    trial {trial+1}/{args.trials}: AUC={r['auc']:.4f}")
                except Exception as e:
                    print(f"    trial {trial+1} FAILED: {e}")

            elapsed = time.time() - t0

            if not trial_results:
                print(f"    ALL TRIALS FAILED")
                continue

            aucs = np.array([r["auc"] for r in trial_results])
            auprcs = np.array([r["auprc"] for r in trial_results])

            result = {
                "model": method,
                "params": {"_dataset": ds, "epochs": args.epochs},
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std()),
                "auprc_mean": float(auprcs.mean()),
                "auprc_std": float(auprcs.std()),
                "elapsed_s": elapsed,
                "num_trials": len(trial_results),
            }
            results.append(result)

            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"    {method}: AUROC={result['auc_mean']:.4f}±{result['auc_std']:.4f}  "
                  f"AUPRC={result['auprc_mean']:.4f}±{result['auprc_std']:.4f}  ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (AUROC)")
    print(f"{'='*80}")
    methods_seen = sorted(set(r["model"] for r in results))
    ds_seen = [d for d in all_datasets if d in set(r["params"]["_dataset"] for r in results)]

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
