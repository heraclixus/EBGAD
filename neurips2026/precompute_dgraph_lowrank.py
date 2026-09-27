"""Precompute DGraph eigendecomposition.

Supports two methods:
- eigsh: scipy ARPACK eigsh with which='SM' (most accurate, CPU-only)
- svd_lowrank: torch randomized SVD with Rayleigh quotient (GPU, approximate)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, time
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data, extract_labeled_subgraph


def build_sparse_laplacian(data, graph_type="original"):
    """Build sparse Laplacian."""
    n = data.num_nodes
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]),
         (edge_index[0], edge_index[1])), shape=(n, n))
    adj = (adj + adj.T) / 2

    if graph_type == "affinity":
        x = data.x.numpy()
        src, dst = edge_index[0], edge_index[1]
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = np.sum(x_norm[src] * x_norm[dst], axis=1)
        sim = np.maximum(sim, 0)
        w = np.zeros(n)
        cnt = np.zeros(n)
        np.add.at(w, src, sim)
        np.add.at(cnt, src, 1)
        np.add.at(w, dst, sim)
        np.add.at(cnt, dst, 1)
        w = w / np.maximum(cnt, 1)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        edge_weights = w[src] * w[dst] * sim
        adj = sp.coo_matrix(
            (edge_weights, (src, dst)), shape=(n, n))
        adj = (adj + adj.T) / 2

    deg = np.array(adj.sum(1)).flatten()
    D = sp.diags(deg)
    L = D - adj
    return L.tocsr()


def eigen_arpack(L_sparse, k):
    """Compute k smallest eigenvalues/vectors using ARPACK (scipy eigsh).

    Uses which='SM' (smallest magnitude) which is accurate but requires
    implicit shift-and-invert. For large matrices, uses which='SA'
    (smallest algebraic) which avoids LU factorization.
    """
    n = L_sparse.shape[0]
    k = min(k, n - 2)

    print(f"    ARPACK eigsh: n={n}, k={k}...", flush=True)
    t0 = time.time()

    L_csc = L_sparse.tocsc().astype(np.float64)

    try:
        # Try shift-invert mode (most accurate for smallest eigenvalues)
        # sigma=0 finds eigenvalues near 0
        print(f"    Trying shift-invert mode (sigma=0)...", flush=True)
        lam_np, V_np = eigsh(L_csc, k=k, sigma=0, which='LM',
                             maxiter=1000, tol=1e-6)
    except Exception as e:
        print(f"    Shift-invert failed: {str(e)[:80]}", flush=True)
        print(f"    Falling back to which='SA'...", flush=True)
        lam_np, V_np = eigsh(L_csc, k=k, which='SA',
                             maxiter=5000, tol=1e-6)

    lam = torch.from_numpy(lam_np.astype(np.float32))
    V_out = torch.from_numpy(V_np.astype(np.float32))

    # Sort by eigenvalue (ascending)
    idx = torch.argsort(lam)
    lam = torch.clamp(lam[idx], min=0)
    V_out = V_out[:, idx]

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. "
          f"lambda range: [{lam[0]:.6f}, {lam[-1]:.4f}]", flush=True)

    return lam, V_out


def eigen_lobpcg(L_sparse, k):
    """Compute k smallest eigenvalues/vectors using LOBPCG.

    Memory-efficient iterative method for finding smallest eigenvalues.
    Uses random initial vectors and diagonal preconditioner.
    """
    n = L_sparse.shape[0]
    k = min(k, n - 2)

    print(f"    LOBPCG: n={n}, k={k}...", flush=True)
    t0 = time.time()

    L_csc = L_sparse.tocsc().astype(np.float64)

    # Random initial guess
    rng = np.random.RandomState(42)
    X0 = rng.randn(n, k).astype(np.float64)

    # Diagonal preconditioner (inverse of diagonal)
    diag = np.array(L_csc.diagonal()).flatten()
    diag_inv = 1.0 / np.maximum(diag, 1e-10)
    M_precond = sp.diags(diag_inv)

    lam_np, V_np = lobpcg(L_csc, X0, M=M_precond, largest=False,
                           maxiter=500, tol=1e-6, verbosityLevel=0)

    lam = torch.from_numpy(lam_np.astype(np.float32))
    V_out = torch.from_numpy(V_np.astype(np.float32))

    idx = torch.argsort(lam)
    lam = torch.clamp(lam[idx], min=0)
    V_out = V_out[:, idx]

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. "
          f"lambda range: [{lam[0]:.6f}, {lam[-1]:.4f}]", flush=True)

    return lam, V_out


def eigen_lowrank(L_sparse, k, device="cuda"):
    """Compute k smallest eigenvalues/vectors using torch.svd_lowrank.

    Strategy: compute on shifted matrix (sigma*I - L) to get largest
    singular values, which correspond to smallest eigenvalues of L.
    """
    n = L_sparse.shape[0]
    k = min(k, n - 2)

    print(f"    Low-rank SVD: n={n}, k={k}, device={device}...", flush=True)
    t0 = time.time()

    # Convert sparse L to torch sparse tensor
    L_coo = L_sparse.tocoo()
    indices = torch.stack([
        torch.from_numpy(L_coo.row.astype(np.int64)),
        torch.from_numpy(L_coo.col.astype(np.int64)),
    ])
    values = torch.from_numpy(L_coo.data.astype(np.float32))
    L_torch = torch.sparse_coo_tensor(indices, values, (n, n)).to(device)

    # Estimate max eigenvalue for shift
    diag = L_sparse.diagonal()
    print(f"    L stats: nnz={L_sparse.nnz}, diag_max={diag.max():.2f}",
          flush=True)
    sigma = float(diag.max() * 2)
    print(f"    Shift sigma={sigma:.1f}", flush=True)

    # Create shifted sparse matrix
    shift_diag = torch.full((n,), sigma, device=device, dtype=torch.float32)
    I_indices = torch.arange(n, device=device).unsqueeze(0).expand(2, -1)
    I_sparse = torch.sparse_coo_tensor(I_indices, shift_diag, (n, n))
    M = I_sparse - L_torch  # sigma*I - L
    M = M.coalesce()

    # Randomized SVD with more iterations for accuracy
    q = min(k + 50, n)  # more oversampling
    U, S, V = torch.svd_lowrank(M, q=q, niter=15)

    V_out = U[:, :k].cpu()

    # Compute eigenvalues via Rayleigh quotient
    print(f"    Computing eigenvalues via Rayleigh quotient...", flush=True)
    V_np = V_out.numpy()
    LV = L_sparse @ V_np
    lam_np = np.sum(V_np * LV, axis=0) / np.maximum(np.sum(V_np**2, axis=0), 1e-12)
    lam = torch.from_numpy(lam_np.astype(np.float32))

    # Sort by eigenvalue (ascending)
    idx = torch.argsort(lam)
    lam = torch.clamp(lam[idx], min=0)
    V_out = V_out[:, idx]

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. "
          f"lambda range: [{lam[0]:.6f}, {lam[-1]:.4f}]", flush=True)

    return lam, V_out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cache/dgraph_eigen")
    parser.add_argument("--k-values", type=str, default="128,256,500")
    parser.add_argument("--graph-types", type=str, default="original")
    parser.add_argument("--subset", type=str, default="labeled",
                        choices=["all", "full", "labeled"])
    parser.add_argument("--method", type=str, default="eigsh",
                        choices=["eigsh", "lobpcg", "svd_lowrank"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k_values = [int(k) for k in args.k_values.split(",")]
    graph_types = args.graph_types.split(",")

    eigen_fn = {
        "eigsh": eigen_arpack,
        "lobpcg": eigen_lobpcg,
        "svd_lowrank": lambda L, k, **kw: eigen_lowrank(L, k, device=args.device),
    }[args.method]

    print(f"=== Loading DGraph (method={args.method}) ===", flush=True)
    data = load_data("dgraph")
    print(f"  {data.num_nodes} nodes, {data.x.shape[1]} features, "
          f"{data.edge_index.shape[1]} edges", flush=True)

    datasets = []
    if args.subset in ("all", "full"):
        datasets.append(("full", data))
    if args.subset in ("all", "labeled"):
        sub_data, _ = extract_labeled_subgraph(data)
        print(f"  Labeled subgraph: {sub_data.num_nodes} nodes, "
              f"{sub_data.edge_index.shape[1]} edges", flush=True)
        datasets.append(("labeled", sub_data))

    for subset_name, d in datasets:
        for gt in graph_types:
            print(f"\n=== {subset_name} / {gt} ===", flush=True)
            L = build_sparse_laplacian(d, graph_type=gt)

            for k in k_values:
                fname = f"{subset_name}_{gt}_k{k}_{args.method}.pt"
                fpath = os.path.join(args.output_dir, fname)

                if os.path.exists(fpath):
                    print(f"  {fname} already exists, skipping", flush=True)
                    continue

                print(f"  Computing k={k}...", flush=True)
                try:
                    lam_L, V = eigen_fn(L, k)
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
                        "method": args.method,
                    }, fpath)
                    size_mb = os.path.getsize(fpath) / 1e6
                    print(f"    Saved {fname} ({size_mb:.1f} MB)", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"    FAILED k={k}: {str(e)[:200]}", flush=True)

                if args.device == "cuda":
                    torch.cuda.empty_cache()

    print("\n=== All done ===")
    if os.path.isdir(args.output_dir):
        for f in sorted(os.listdir(args.output_dir)):
            if f.endswith('.pt'):
                size = os.path.getsize(
                    os.path.join(args.output_dir, f)) / 1e6
                print(f"  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
