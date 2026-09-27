"""Targeted baseline score-orientation audit.

This script reruns only baseline/dataset pairs whose native AUROC is below
50% and whose sign-flipped AUROC could threaten EB-GAD.  For each trial it
records metrics for the baseline's native anomaly-score direction and for the
negated score.  The latter is diagnostic only: it tells us whether a sub-50
entry is an orientation issue, not a label-free baseline setting.

Examples
--------
    python run_baseline_orientation_audit.py \
        --backend pyod --tasks elliptic:DIF elliptic_plus_plus:DIF

    python run_baseline_orientation_audit.py \
        --backend diffgad --tasks elliptic:DiffGAD elliptic_plus_plus:DiffGAD \
        --device cuda

    python run_baseline_orientation_audit.py \
        --backend pygod --tasks elliptic:DOMINANT elliptic:AnomalyDAE \
        --device cpu --epochs 50
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve


PYGOD_ALIASES = {
    "AnomDAE": "AnomalyDAE",
    "ANOMDAE": "AnomalyDAE",
}


@dataclass(frozen=True)
class Task:
    dataset: str
    method: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    p, r, _ = precision_recall_curve(y, scores)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "auprc": float(sk_auc(r, p)),
    }


def _orientation_record(
    *,
    backend: str,
    dataset: str,
    method: str,
    trial: int,
    seed: int,
    y: np.ndarray,
    scores: np.ndarray,
    elapsed_s: float,
    params: dict | None = None,
) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    native = _metrics(y, scores)
    flipped = _metrics(y, -scores)
    normal = scores[y == 0]
    anomaly = scores[y == 1]
    return {
        "kind": "trial",
        "backend": backend,
        "dataset": dataset,
        "method": method,
        "trial": trial,
        "seed": seed,
        "n_eval": int(y.shape[0]),
        "n_anomaly": int(y.sum()),
        "anomaly_rate": float(y.mean()),
        "auc_native": native["auc"],
        "auc_flipped": flipped["auc"],
        "auprc_native": native["auprc"],
        "auprc_flipped": flipped["auprc"],
        "ap_native": native["ap"],
        "ap_flipped": flipped["ap"],
        "score_mean_normal": float(normal.mean()) if normal.size else float("nan"),
        "score_mean_anomaly": float(anomaly.mean()) if anomaly.size else float("nan"),
        "score_std": float(scores.std()),
        "elapsed_s": float(elapsed_s),
        "params": params or {},
    }


def _summary_record(rows: list[dict]) -> dict:
    first = rows[0]
    keys = [
        "auc_native",
        "auc_flipped",
        "auprc_native",
        "auprc_flipped",
        "ap_native",
        "ap_flipped",
    ]
    out = {
        "kind": "summary",
        "backend": first["backend"],
        "dataset": first["dataset"],
        "method": first["method"],
        "num_trials": len(rows),
        "n_eval": first["n_eval"],
        "n_anomaly": first["n_anomaly"],
        "anomaly_rate": first["anomaly_rate"],
        "elapsed_s": float(sum(r["elapsed_s"] for r in rows)),
        "params": first.get("params", {}),
    }
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    out["flip_would_help"] = bool(out["auc_flipped_mean"] > out["auc_native_mean"])
    return out


def parse_tasks(items: Iterable[str]) -> list[Task]:
    tasks = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Task must be dataset:method, got {item!r}")
        dataset, method = item.split(":", 1)
        method = PYGOD_ALIASES.get(method, method)
        tasks.append(Task(dataset=dataset, method=method))
    return tasks


def _labeled_y(data):
    y = data.y.detach().cpu().numpy().flatten()
    mask = (y >= 0) & (y <= 1)
    return (y[mask] > 0).astype(int), mask


def run_pyod_task(data, task: Task, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if task.method == "LOF":
        from pyod.models.lof import LOF

        clf = LOF(n_neighbors=20)
    elif task.method == "ECOD":
        from pyod.models.ecod import ECOD

        clf = ECOD()
    elif task.method == "DIF":
        from pyod.models.dif import DIF

        kwargs = {}
        if "random_state" in inspect.signature(DIF).parameters:
            kwargs["random_state"] = seed
        clf = DIF(**kwargs)
    elif task.method == "IForest":
        from pyod.models.iforest import IForest

        clf = IForest(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unknown PyOD method: {task.method}")

    y, mask = _labeled_y(data)
    X = data.x.detach().cpu().numpy().astype(np.float64)[mask]
    clf.fit(X)
    return y, np.asarray(clf.decision_scores_, dtype=float), {}


def run_pygod_task(
    data,
    task: Task,
    *,
    seed: int,
    device: str,
    epochs: int,
    hid_dim: int | None,
    batch_size: int | None,
    num_neigh: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from pygod.detector import ANOMALOUS, CONAD, DOMINANT, CoLA, AnomalyDAE

    gpu = 0 if device.startswith("cuda") else -1
    mem_kw = {}
    if hid_dim is not None:
        mem_kw["hid_dim"] = hid_dim
    if batch_size is not None and batch_size > 0:
        mem_kw["batch_size"] = batch_size
    if num_neigh is not None and num_neigh > 0:
        mem_kw["num_neigh"] = num_neigh

    if task.method == "DOMINANT":
        clf = DOMINANT(epoch=epochs, gpu=gpu, verbose=0, **mem_kw)
    elif task.method == "AnomalyDAE":
        ae_kw = {k: v for k, v in mem_kw.items() if k in ("hid_dim", "batch_size", "num_neigh")}
        if "hid_dim" in ae_kw:
            ae_kw["emb_dim"] = ae_kw["hid_dim"]
        clf = AnomalyDAE(epoch=epochs, gpu=gpu, verbose=0, **ae_kw)
    elif task.method == "CONAD":
        clf = CONAD(epoch=epochs, gpu=gpu, verbose=0, **mem_kw)
    elif task.method == "CoLA":
        clf = CoLA(epoch=epochs, gpu=gpu, verbose=0, **mem_kw)
    elif task.method == "ANOMALOUS":
        clf = ANOMALOUS(epoch=epochs, gpu=gpu, verbose=0)
    else:
        raise ValueError(f"Unknown PyGOD method: {task.method}")

    clf.fit(data)
    scores = clf.decision_score_
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    y, mask = _labeled_y(data)
    params = {"epochs": epochs}
    if mem_kw:
        params.update(mem_kw)
    return y, np.asarray(scores, dtype=float)[mask], params


def run_diffgad_task(
    data,
    task: Task,
    *,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from bridge.diffgad_baseline import run_diffgad_trial

    params = {
        "ae_dropout": 0.3,
        "ae_lr": 0.01,
        "ae_alpha": 0.8,
        "proto_alpha": 0.001,
        "weight": 0.0,
        "lr": 0.005,
        "diff_epochs": 800,
        "ae_epochs": 300,
        "patience": 100,
    }
    result = run_diffgad_trial(data, device=device, return_scores=True, verbose=False, **params)
    y = np.asarray(result.get("y", []), dtype=int)
    scores = np.asarray(result.get("scores", []), dtype=float)
    if y.size == 0:
        raise RuntimeError("DiffGAD did not return labeled scores")
    return y, scores, params


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted baseline orientation audit")
    parser.add_argument("--backend", choices=["pyod", "pygod", "diffgad"], required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="dataset:method pairs")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hid-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-neigh", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/baseline_orientation_audit.jsonl")
    args = parser.parse_args()

    from data_utils import load_data

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tasks = parse_tasks(args.tasks)

    with open(args.output, "a") as f:
        for task in tasks:
            print(f"\n=== {args.backend} {task.dataset}:{task.method} ===", flush=True)
            data = load_data(task.dataset)
            trial_rows = []
            for trial in range(args.trials):
                seed = args.seed + trial
                _set_seed(seed)
                t0 = time.time()
                if args.backend == "pyod":
                    y, scores, params = run_pyod_task(data, task, seed)
                elif args.backend == "pygod":
                    y, scores, params = run_pygod_task(
                        data,
                        task,
                        seed=seed,
                        device=args.device,
                        epochs=args.epochs,
                        hid_dim=args.hid_dim,
                        batch_size=args.batch_size,
                        num_neigh=args.num_neigh,
                    )
                elif args.backend == "diffgad":
                    y, scores, params = run_diffgad_task(data, task, device=args.device, seed=seed)
                else:
                    raise AssertionError(args.backend)
                elapsed_s = time.time() - t0
                row = _orientation_record(
                    backend=args.backend,
                    dataset=task.dataset,
                    method=task.method,
                    trial=trial,
                    seed=seed,
                    y=y,
                    scores=scores,
                    elapsed_s=elapsed_s,
                    params=params,
                )
                f.write(json.dumps(row, default=_jsonable) + "\n")
                f.flush()
                trial_rows.append(row)
                print(
                    f"trial {trial + 1}/{args.trials}: "
                    f"native={row['auc_native']:.4f} flipped={row['auc_flipped']:.4f} "
                    f"mean0={row['score_mean_normal']:.4g} mean1={row['score_mean_anomaly']:.4g} "
                    f"({elapsed_s:.1f}s)",
                    flush=True,
                )

            summary = _summary_record(trial_rows)
            f.write(json.dumps(summary, default=_jsonable) + "\n")
            f.flush()
            print(
                f"summary: native={summary['auc_native_mean']:.4f}±{summary['auc_native_std']:.4f}, "
                f"flipped={summary['auc_flipped_mean']:.4f}±{summary['auc_flipped_std']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
