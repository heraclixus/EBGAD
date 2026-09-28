"""Worked example on Zachary's karate club: how structure and an uncertain edge enter the score.

Features are the connectivity profiles (rows of the adjacency matrix). The pipeline is the one of
the paper (z-scored features, full spectrum, EB fit, equilibrium energy J*). The edge between
individuals 23 and 34, on which Zachary's published matrix is inconsistent, is toggled and the
scores are compared.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
from test_dynamic_score_bank import (  # noqa: E402
    entry_for_fit, fit_best_entry, horizon_score_bank, prepare_model_entries, q_for_fit,
)

KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
EDGE = (22, 33)  # individuals 23 and 34, zero-based


def score(A):
    n = A.shape[0]
    src, dst = np.nonzero(A)
    data = SimpleNamespace(x=torch.tensor(A, dtype=torch.float32),
                           edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
                           y=torch.zeros(n, dtype=torch.long), num_nodes=n)
    entries = prepare_model_entries(data, graph_types=["original"], template_gammas=[0.5, 1.0, 2.0],
                                    pca_dim=None, truncated_k=None, modeled_subspace="drop_nullspace",
                                    template_type="low", device="cpu")
    fit = fit_best_entry(entries, [np.inf], KAPPAS, False)
    entry = entry_for_fit(entries, fit)
    bank = horizon_score_bank(entry, q_for_fit(entry, fit), [np.inf])
    return np.asarray(bank["J@inf"], np.float64), fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/camera_ready/studies/karate.json")
    args = ap.parse_args()
    G = nx.karate_club_graph()
    A = (nx.to_numpy_array(G, weight=None) > 0).astype(np.float64)
    present = bool(A[EDGE])
    A_with, A_without = A.copy(), A.copy()
    A_with[EDGE] = A_with[EDGE[::-1]] = 1.0
    A_without[EDGE] = A_without[EDGE[::-1]] = 0.0
    j1, fit1 = score(A_with)
    j0, fit0 = score(A_without)
    delta = np.abs(j1 - j0)
    hop = set(EDGE)
    for e in EDGE:
        hop |= set(np.nonzero(A_with[e])[0].tolist())
    hop = sorted(hop)
    r1, r0 = rankdata(-j1, method="average"), rankdata(-j0, method="average")
    outside = [i for i in range(len(j1)) if i not in hop]
    out = {
        "edge_in_networkx_graph": present, "n": int(A.shape[0]),
        "fit_with": {k: float(fit1[k]) for k in ("rho", "kappa", "template_gamma")},
        "fit_without": {k: float(fit0[k]) for k in ("rho", "kappa", "template_gamma")},
        "top3_with": [int(i) + 1 for i in np.argsort(-j1)[:3]],
        "top3_without": [int(i) + 1 for i in np.argsort(-j0)[:3]],
        "largest_changes": [int(i) + 1 for i in np.argsort(-delta)[:3]],
        "mass_on_endpoints_and_neighbors": 100 * float(delta[hop].sum() / delta.sum()),
        "n_endpoints_and_neighbors": len(hop),
        "max_rank_shift_outside": float(np.max(np.abs(r1 - r0)[outside])) if outside else 0.0,
        "n_outside_shifted": int(np.sum(np.abs(r1 - r0)[outside] > 0)) if outside else 0,
        "max_rank_shift_inside": float(np.max(np.abs(r1 - r0)[hop])),
        "spearman": float(np.corrcoef(r1, r0)[0, 1]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
