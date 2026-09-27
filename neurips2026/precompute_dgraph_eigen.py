"""Precompute and cache DGraph eigendecomposition.

The Laplacian eigendecomposition depends only on (graph_type, k).
Cache it once, reuse for all hyperparameter configs.

Saves: {lam_L, V, edge_index, x, y, n} as .pt files.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, time
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg, eigsh

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data, extract_labeled_subgraph


def build_laplacian(data, graph_type="original"):
    """Build sparse Laplacian from data."""
    n = data.num_nodes
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]),
         (edge_index[0], edge_index[1])),
        shape=(n, n))
    adj = (adj + adj.T) / 2

    if graph_type == "affinity":
        # Affinity weighting: downweight edges between dissimilar nodes
        x = data.x.numpy()
        src, dst = edge_index[0], edge_index[1]
        # Cosine similarity per edge
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = np.sum(x_norm[src] * x_norm[dst], axis=1)
        sim = np.maximum(sim, 0)  # ReLU
        # Per-node affinity (mean similarity to neighbors)
        from scipy.sparse import csr_matrix
        w = np.zeros(n)
        cnt = np.zeros(n)
        np.add.at(w, src, sim)
        np.add.at(cnt, src, 1)
        np.add.at(w, dst, sim)
        np.add.at(cnt, dst, 1)
        w = w / np.maximum(cnt, 1)
        # Min-max normalize
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        # Refined adjacency
        edge_weights = w[src] * w[dst] * sim
        adj = sp.coo_matrix(
            (edge_weights, (src, dst)), shape=(n, n))
        adj = (adj + adj.T) / 2

    deg = np.array(adj.sum(1)).flatten()
    D = sp.diags(deg)
    L = D - adj
    return L.tocsc()


def compute_eigen(L, k, method="lobpcg"):
    """Truncated eigendecomposition."""
    n = L.shape[0]
    k = min(k, n - 2)

    print(f"    Eigendecomposition: n={n}, k={k}, method={method}...",
          flush=True)
    t0 = time.time()

    if method == "lobpcg":
        np.random.seed(42)
        X0 = np.random.randn(n, k).astype(np.float64)
        try:
            lam, V = lobpcg(L, X0, largest=False, maxiter=1000, tol=1e-6)
        except Exception as e:
            print(f"    LOBPCG failed ({e}), falling back to eigsh",
                  flush=True)
            lam, V = eigsh(L, k=k, which='SM', maxiter=2000)
    else:
        lam, V = eigsh(L, k=k, which='SM', maxiter=2000)

    # Sort by eigenvalue
    idx = np.argsort(lam)
    lam = np.maximum(lam[idx], 0)
    V = V[:, idx]

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. λ range: [{lam[0]:.6f}, {lam[-1]:.4f}]",
          flush=True)

    return torch.from_numpy(lam).float(), torch.from_numpy(V).float()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cache/dgraph_eigen")
    parser.add_argument("--k-values", type=str, default="128,256,500")
    parser.add_argument("--graph-types", type=str, default="original,affinity")
    parser.add_argument("--labeled-only", action="store_true",
                        help="Also compute for labeled subgraph")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["all", "full", "labeled"],
                        help="Which subset to compute: all, full, or labeled")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k_values = [int(k) for k in args.k_values.split(",")]
    graph_types = args.graph_types.split(",")

    print("=== Loading DGraph ===", flush=True)
    data = load_data("dgraph")
    print(f"  {data.num_nodes} nodes, {data.x.shape[1]} features, "
          f"{data.edge_index.shape[1]} edges", flush=True)

    datasets = []
    if args.subset in ("all", "full"):
        datasets.append(("full", data))
    if args.subset in ("all", "labeled") or args.labeled_only:
        sub_data, _ = extract_labeled_subgraph(data)
        print(f"  Labeled subgraph: {sub_data.num_nodes} nodes, "
              f"{sub_data.edge_index.shape[1]} edges", flush=True)
        datasets.append(("labeled", sub_data))

    for subset_name, d in datasets:
        for gt in graph_types:
            print(f"\n=== {subset_name} / {gt} ===", flush=True)
            L = build_laplacian(d, graph_type=gt)

            for k in k_values:
                fname = f"{subset_name}_{gt}_k{k}.pt"
                fpath = os.path.join(args.output_dir, fname)

                if os.path.exists(fpath):
                    print(f"  {fname} already exists, skipping", flush=True)
                    continue

                print(f"  Computing k={k}...", flush=True)
                try:
                    lam_L, V = compute_eigen(L, k)
                    torch.save({
                        "lam_L": lam_L,
                        "V": V,
                        "x": d.x,
                        "y": d.y,
                        "edge_index": d.edge_index,
                        "n": d.num_nodes,
                        "k": k,
                        "graph_type": gt,
                        "subset": subset_name,
                    }, fpath)
                    size_mb = os.path.getsize(fpath) / 1e6
                    print(f"    Saved {fname} ({size_mb:.1f} MB)", flush=True)
                except Exception as e:
                    print(f"    FAILED k={k}: {str(e)[:80]}", flush=True)

    print("\n=== All done ===")
    print(f"Cached files in {args.output_dir}/:")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
        print(f"  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
