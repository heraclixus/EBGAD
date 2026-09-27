"""E13: do the edge cross-terms Q_ij <delta_i, delta_j> detect anomalous
edges directly? (Pairwise-detection extension.)

Reuses the E8 clustered-injection protocol verbatim (reddit, frac 0.05,
BFS clusters capped at 10, seeds 0-4, EB refit from scratch on the
contaminated graph). For every undirected edge we compute the magnitude of
its cross-term contribution to the equilibrium quadratic form,
|Q_ij <delta_i, delta_j>| with Q_ij = sum_k q_k V_ik V_jk, and ask whether
edges joining two injected anomalies rank above edges joining two untouched
normal nodes.
"""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import roc_auc_score

from data_utils import load_data
from test_dynamic_score_bank import (
    entry_for_fit,
    fit_best_entry,
    prepare_model_entries,
    q_for_fit,
)
from test_e8_neighbor_anomalies import (
    DS_GAMMAS,
    ERA_HORIZONS,
    ERA_KAPPAS,
    inject,
    pick_nodes,
)


def unique_edges(edge_index, n):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    keep = src < dst
    return src[keep], dst[keep]


def edge_quantities(V, q, delta, src, dst, batch=4096):
    Wq = V * q[None, :]
    Ws = V * (1.0 / np.maximum(q, 1e-12))[None, :]
    sig_diag = np.einsum("nk,nk->n", V, Ws)
    q_e = np.empty(len(src), dtype=np.float64)
    s_e = np.empty(len(src), dtype=np.float64)
    r_e = np.empty(len(src), dtype=np.float64)
    for lo in range(0, len(src), batch):
        hi = min(lo + batch, len(src))
        q_e[lo:hi] = np.einsum("ek,ek->e", Wq[src[lo:hi]], V[dst[lo:hi]])
        s_e[lo:hi] = np.einsum("ek,ek->e", Ws[src[lo:hi]], V[dst[lo:hi]])
        r_e[lo:hi] = np.einsum("ed,ed->e", delta[src[lo:hi]], delta[dst[lo:hi]])
    return q_e, s_e, r_e, sig_diag


def run(ds: str, frac: float, seed: int, cluster_size: int, device: str) -> dict:
    base = load_data(ds)
    rng = np.random.default_rng(seed)
    y0 = base.y.detach().cpu().numpy()
    n_inject = int(round(frac * len(y0)))
    nodes = pick_nodes(base, n_inject, "clustered", rng, cluster_size)
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

    src, dst = unique_edges(data.edge_index, data.num_nodes)
    q_e, s_e, r_e, sig_diag = edge_quantities(entry["V"], q, entry["delta"], src, dst)
    null_scale = np.sqrt(np.maximum(
        sig_diag[src] * sig_diag[dst] + s_e * s_e, 1e-12))
    # node equilibrium energies: E_i = <delta_i, (Q delta)_i>
    A = entry["V"].T @ entry["delta"]              # (k, d) spectral residuals
    B = entry["V"] * q[None, :]                    # (n, k)
    je = np.einsum("nd,nd->n", B @ A, entry["delta"])

    readouts = {
        "cross_raw": np.abs(q_e * r_e),
        "cross_std": np.abs(r_e) / null_scale,
        "align_raw": np.abs(r_e),
        "min_node_energy": np.minimum(je[src], je[dst]),
    }

    inj_s, inj_d = injected[src], injected[dst]
    norm_s = (y0[src] == 0) & ~inj_s
    norm_d = (y0[dst] == 0) & ~inj_d
    pos = inj_s & inj_d
    neg = norm_s & norm_d
    mixed = (inj_s ^ inj_d) & (norm_s | norm_d)

    row = {
        "dataset": ds, "frac": frac, "seed": seed, "cluster_size": cluster_size,
        "rho": fit["rho"], "kappa": fit["kappa"], "entry": fit["entry_key"],
        "n_edges": int(len(src)), "n_pos": int(pos.sum()),
        "n_mixed": int(mixed.sum()), "n_neg": int(neg.sum()),
    }
    sel = pos | neg
    sel_mix = mixed | neg
    for name, score in readouts.items():
        row["auc_%s" % name] = float(
            roc_auc_score(pos[sel].astype(int), score[sel]))
        row["auc_mixed_%s" % name] = float(
            roc_auc_score(mixed[sel_mix].astype(int), score[sel_mix]))
    print("  %-8s seed=%d rho=%.3f kappa=%g pos=%d | " % (
        ds, seed, row["rho"], row["kappa"], row["n_pos"]) +
        " ".join("%s=%.3f" % (k, row["auc_%s" % k]) for k in readouts), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reddit")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cluster-size", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/e13/e13_edge_crossterms.json")
    args = ap.parse_args()

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        try:
            rows.append(run(args.dataset, args.frac, seed,
                            args.cluster_size, args.device))
        except Exception as exc:
            print("  ERROR seed %d: %s" % (seed, str(exc)[:300]), flush=True)
            rows.append({"seed": seed, "error": str(exc)[:300]})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
