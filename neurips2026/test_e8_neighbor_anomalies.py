"""E8: do neighboring (clustered) anomalies evade the contextual score?

Inject the same number of feature-swap anomalies either scattered uniformly
over normal nodes or as connected BFS clusters, refit EB, and compare the
label-free selected score's ability to rank the injected nodes. If clustered
anomalies define their own local context, a purely contextual score should
degrade on the clustered placement; the equilibrium/global families should not.
"""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
from collections import deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

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


def adjacency_lists(edge_index, n):
    adj = [[] for _ in range(n)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for a, b in zip(src, dst):
        adj[a].append(b)
    return adj


def pick_nodes(data, n_inject: int, placement: str, rng,
               cluster_size: int | None = None) -> np.ndarray:
    y = data.y.detach().cpu().numpy()
    normal = np.where(y == 0)[0]
    if placement == "scattered":
        return rng.choice(normal, size=n_inject, replace=False)
    # clustered: grow BFS balls over normal nodes from random seeds.
    # cluster_size caps each ball (many small neighbor groups); None grows
    # one contiguous region (adversarial extreme).
    adj = adjacency_lists(data.edge_index, data.num_nodes)
    normal_set = set(normal.tolist())
    chosen: set[int] = set()
    while len(chosen) < n_inject:
        seed = int(rng.choice(normal))
        if seed in chosen:
            continue
        cap = n_inject if cluster_size is None else min(
            n_inject, len(chosen) + cluster_size)
        queue = deque([seed])
        while queue and len(chosen) < cap:
            u = queue.popleft()
            if u in chosen or u not in normal_set:
                continue
            chosen.add(u)
            nbrs = [v for v in adj[u] if v in normal_set and v not in chosen]
            rng.shuffle(nbrs)
            queue.extend(nbrs)
    return np.fromiter(chosen, dtype=np.int64)


def inject(data, nodes: np.ndarray, rng):
    x = data.x.clone()
    n = x.shape[0]
    for i in nodes:
        j = int(rng.integers(0, n))
        x[int(i)] = data.x[j]
    out = data.clone()
    out.x = x
    return out


def run(ds: str, placement: str, frac: float, seed: int, device: str,
        cluster_size: int | None = None) -> dict:
    base = load_data(ds)
    rng = np.random.default_rng(seed)
    y0 = base.y.detach().cpu().numpy()
    n_inject = int(round(frac * len(y0)))
    nodes = pick_nodes(base, n_inject, placement, rng, cluster_size)
    data = inject(base, nodes, rng)

    injected = np.zeros(len(y0), bool)
    injected[nodes] = True
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
    # injected vs untouched normals; original anomalies excluded
    y_inj = injected.astype(np.int64)
    mask_inj = (y0 == 0)
    summary = evaluate_score_bank(y_inj, mask_inj, bank, references=refs)
    row = {
        "dataset": ds, "placement": placement, "frac": frac, "seed": seed,
        "rho": fit["rho"], "kappa": fit["kappa"], "entry": fit["entry_key"],
        "score_ks": summary.get("score_ks"),
        "auc_injected": summary.get("auc_ks_selected"),
        "auc_injected_oracle": summary.get("auc_oracle"),
    }
    print("  %-8s %-9s eps=%.2f seed=%d rho=%.3f kappa=%-6g sel=%s inj=%.1f%% oracle=%.1f%%" % (
        ds, placement, frac, seed, row["rho"], row["kappa"], row["score_ks"],
        100 * (row["auc_injected"] or 0),
        100 * (row["auc_injected_oracle"] or 0)), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="weibo,reddit")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cluster-size", type=int, default=None,
                    help="cap each BFS ball (many small neighbor groups); "
                         "default None grows one contiguous region")
    ap.add_argument("--placements", default="scattered,clustered")
    ap.add_argument("--out", default="results/e8/e8_neighbor.json")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets.split(","):
        for placement in args.placements.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                try:
                    rows.append(run(ds.strip(), placement, args.frac, seed,
                                    args.device, args.cluster_size))
                except Exception as exc:
                    print("  ERROR %s %s: %s" % (ds, placement, str(exc)[:200]), flush=True)
                    rows.append({"dataset": ds, "placement": placement,
                                 "error": str(exc)[:300]})
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out, "w") as fh:
                    json.dump(rows, fh, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
