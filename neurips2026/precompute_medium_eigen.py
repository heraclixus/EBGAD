"""Precompute eigendecomposition for medium-sized datasets.

Caches eigenpairs so scoring is instant (no eigendecomp per config).
Datasets: T-Finance (39K), YelpChi (46K), ACM (16K), BlogCatalog (5K).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, time
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data


def build_laplacian(data):
    n = data.num_nodes
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]),
         (edge_index[0], edge_index[1])), shape=(n, n))
    adj = (adj + adj.T) / 2
    deg = np.array(adj.sum(1)).flatten()
    L = sp.diags(deg) - adj
    return L.tocsc()


def main():
    datasets = {
        "t_finance": ("t_finance", 500),
        "yelpchi": ("YelpChi", 500),
        "acm": ("ACM", 500),
        "blogcatalog": ("BlogCatalog", 500),
    }

    out_dir = "cache/medium_eigen"
    os.makedirs(out_dir, exist_ok=True)

    for ds_key, (load_name, k) in datasets.items():
        fname = f"{ds_key}_k{k}.pt"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"  {fname} exists, skipping", flush=True)
            continue

        print(f"=== {ds_key} ===", flush=True)
        data = load_data(load_name)
        n = data.num_nodes
        print(f"  {n} nodes, {data.x.shape[1]} features, "
              f"{data.edge_index.shape[1]} edges", flush=True)

        L = build_laplacian(data)
        k_actual = min(k, n - 2)

        t0 = time.time()
        print(f"  eigsh k={k_actual}...", flush=True)
        lam_L, V = eigsh(L, k=k_actual, which='SM')
        idx = np.argsort(lam_L)
        lam_L = np.maximum(lam_L[idx], 0)
        V = V[:, idx]
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

        torch.save({
            "lam_L": torch.from_numpy(lam_L).float(),
            "V": torch.from_numpy(V).float(),
            "x": data.x, "y": data.y,
            "edge_index": data.edge_index,
            "n": n, "k": k_actual,
        }, fpath)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  Saved {fname} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
