"""
TAM (Truncated Affinity Maximization) baseline runner.

Adapted from the original TAM codebase (Qiao et al., 2024) to work with
our data_utils pipeline.  Removes DGL dependency (unused in original
training loop) and uses PyG data objects as input.

Usage::

    python run_tam.py --dataset disney --device cuda --trials 5
    python run_tam.py --dataset Weibo --device cuda --trials 5
    python run_tam.py --dataset Amazon --device cuda --trials 10
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import auc as sk_auc, precision_recall_curve

# The TAM authors' model code is not redistributed here: clone https://github.com/mala-lab/TAM-master into
# third_party/TAM-master (or point TAM_DIR at a clone) and apply baselines/patches/tam.patch.
sys.path.insert(0, os.environ.get("TAM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "TAM-master")))
from model import Model  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions (adapted from the TAM utils.py, DGL-free)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_adj_tensor(raw_adj):
    adj = raw_adj[0, :, :]
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    return adj.unsqueeze(0)


def normalize_score(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)


def calc_distance(adj, seq):
    """Pairwise Euclidean distance for connected nodes (O(n^2) memory). Legacy path for small graphs."""
    n = adj.shape[0]
    dis = torch.zeros(n, n, device=adj.device)
    for i in range(n):
        nbrs = torch.argwhere(adj[i, :] > 0).flatten()
        if len(nbrs) > 0:
            diff = seq[i].unsqueeze(0) - seq[nbrs]
            dis[i, nbrs] = torch.sqrt((diff * diff).sum(dim=-1))
    return dis


def calc_distance_edges(edge_index, seq):
    """Compute pairwise distances for edges only. O(E) memory instead of O(n^2)."""
    src, dst = edge_index[0], edge_index[1]
    diff = seq[src] - seq[dst]
    return torch.sqrt((diff * diff).sum(dim=-1).clamp(min=0))


def graph_nsgt(dis_array, adj):
    """Randomly cut distant edges (NSGT procedure). Uses dense dis_array. Legacy for small graphs."""
    dis_array = dis_array.to(adj.device)
    n = dis_array.shape[0]
    dis_u = dis_array * adj
    nonzero = dis_u[dis_u != 0]
    if len(nonzero) == 0:
        return adj
    mean_dis = nonzero.mean()
    for i in range(n):
        nbrs = torch.argwhere(adj[i, :] > 0)
        if nbrs.shape[0] == 0:
            continue
        max_dis = dis_array[i, nbrs].max()
        if max_dis > mean_dis:
            thresh = (max_dis - mean_dis) * np.random.random_sample() + mean_dis
            cut = torch.argwhere(dis_array[i, nbrs[:, 0]] > thresh)
            if cut.shape[0] > 0:
                adj[i, nbrs[cut[:, 0]]] = 0
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


def graph_nsgt_edges(edge_index, edge_dist, adj):
    """NSGT using O(E) edge distances. Cuts distant edges without allocating n×n dis matrix."""
    n = adj.shape[0]
    if edge_index.shape[1] == 0:
        return adj
    mean_dis = edge_dist.mean().item()
    # Group edges by source node for O(E) iteration
    src = edge_index[0]
    dst = edge_index[1]
    # Sort by src for efficient grouping
    perm = torch.argsort(src)
    src_s = src[perm]
    dst_s = dst[perm]
    dist_s = edge_dist[perm]
    # Find boundaries: unique src values
    uniq_src, counts = torch.unique_consecutive(src_s, return_counts=True)
    start = 0
    for i, cnt in zip(uniq_src.tolist(), counts.tolist()):
        nbrs = dst_s[start : start + cnt]
        dists = dist_s[start : start + cnt]
        start += cnt
        if len(nbrs) == 0:
            continue
        max_dis = dists.max().item()
        if max_dis > mean_dis:
            thresh = (max_dis - mean_dis) * np.random.random_sample() + mean_dis
            cut_mask = dists > thresh
            if cut_mask.any():
                cut_nbrs = nbrs[cut_mask]
                adj[i, cut_nbrs] = 0
                adj[cut_nbrs, i] = 0  # symmetric
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


def _build_rev_edge_index(edge_index, n):
    """Build reverse edge index: rev_idx[i] = j where edge j is (v,u) when edge i is (u,v).
    O(E) memory, avoids dict for large graphs."""
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    E = edge_index.shape[1]
    keys = row.astype(np.int64) * n + col
    rev_keys = col.astype(np.int64) * n + row
    perm = np.argsort(keys)
    keys_sorted = keys[perm]
    order = np.arange(E, dtype=np.int64)[perm]
    rev_idx = np.full(E, -1, dtype=np.int64)
    pos = np.searchsorted(keys_sorted, rev_keys)
    valid = (pos < E) & (keys_sorted[pos] == rev_keys)
    rev_idx[valid] = order[pos[valid]]
    return torch.from_numpy(rev_idx).to(edge_index.device)


def graph_nsgt_edges_sparse(edge_index, edge_dist, edge_mask, n, rev_edge_idx=None):
    """NSGT for sparse: updates edge_mask in place. Only considers edges where edge_mask is True.
    rev_edge_idx: optional precomputed reverse edge index (avoids O(E) dict for large graphs)."""
    active = edge_mask.nonzero(as_tuple=True)[0]
    if len(active) == 0:
        return
    ei = edge_index[:, active]
    ed = edge_dist[active]
    mean_dis = ed.mean().item()
    src = ei[0]
    dst = ei[1]
    perm = torch.argsort(src)
    src_s = src[perm]
    dst_s = dst[perm]
    dist_s = ed[perm]
    active_perm = active[perm]
    if rev_edge_idx is None:
        keys = edge_index[0] * n + edge_index[1]
        rev_keys = edge_index[1] * n + edge_index[0]
        key_to_idx = {rev_keys[e].item(): e for e in range(edge_index.shape[1])}
    uniq_src, counts = torch.unique_consecutive(src_s, return_counts=True)
    start = 0
    for i, cnt in zip(uniq_src.tolist(), counts.tolist()):
        dists = dist_s[start : start + cnt]
        edge_indices = active_perm[start : start + cnt]
        start += cnt
        if len(dists) == 0:
            continue
        max_dis = dists.max().item()
        if max_dis > mean_dis:
            thresh = (max_dis - mean_dis) * np.random.random_sample() + mean_dis
            cut_mask = dists > thresh
            if cut_mask.any():
                cut_edge_idx = edge_indices[cut_mask]
                for e in cut_edge_idx.tolist():
                    edge_mask[e] = False
                    if rev_edge_idx is not None:
                        rev_e = rev_edge_idx[e].item()
                        if rev_e >= 0:
                            edge_mask[rev_e] = False
                    else:
                        rev_e = key_to_idx.get(keys[e].item())
                        if rev_e is not None:
                            edge_mask[rev_e] = False


def normalize_adj_sparse(edge_index, edge_mask, n, device):
    """Symmetric normalization D^{-1/2} A D^{-1/2} for sparse adj. Returns sparse tensor."""
    ei = edge_index[:, edge_mask]
    if ei.shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, device=device),
            (n, n),
            device=device,
        )
    row, col = ei[0], ei[1]
    degree = torch.zeros(n, device=device, dtype=torch.float32)
    degree.scatter_add_(0, row, torch.ones(ei.shape[1], device=device, dtype=torch.float32))
    degree = degree.clamp(min=1).pow(-0.5)
    values = degree[row] * degree[col]
    return torch.sparse_coo_tensor(ei, values, (n, n), device=device).coalesce()


def max_message(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.mm(feature, feature.T) * adj_matrix
    sim[torch.isinf(sim)] = 0
    sim[torch.isnan(sim)] = 0
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    message = torch.sum(sim, 1) * r_inv
    return -torch.sum(message), message


def _edge_sim_chunked(feature, row, col, chunk_size=1_000_000):
    """Compute edge similarities in chunks to avoid OOM on large graphs."""
    n_edges = row.shape[0]
    if n_edges <= chunk_size:
        return (feature[row] * feature[col]).sum(dim=1)
    out = []
    for i in range(0, n_edges, chunk_size):
        end = min(i + chunk_size, n_edges)
        r, c = row[i:end], col[i:end]
        sim = (feature[r] * feature[c]).sum(dim=1)
        out.append(sim)
    return torch.cat(out, dim=0)


def max_message_sparse(feature, edge_index, n, chunk_size=1_000_000):
    """Edge-wise message passing. O(E) instead of O(n^2). Chunked for large graphs."""
    feature = feature / torch.norm(feature, dim=-1, keepdim=True).clamp(min=1e-8)
    row, col = edge_index[0], edge_index[1]
    edge_sim = _edge_sim_chunked(feature, row, col, chunk_size)
    degree = torch.zeros(n, device=feature.device, dtype=feature.dtype)
    degree.scatter_add_(0, row, torch.ones_like(row, dtype=feature.dtype, device=feature.device))
    message = torch.zeros(n, device=feature.device, dtype=feature.dtype)
    message.scatter_add_(0, row, edge_sim)
    message = message / degree.clamp(min=1)
    return -torch.sum(message), message


def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.mm(feature, feature.T) * adj_matrix
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    return torch.sum(sim, 1) * r_inv


def inference_sparse(feature, edge_index, n, chunk_size=1_000_000):
    """Edge-wise inference. O(E) instead of O(n^2). Chunked for large graphs."""
    feature = feature / torch.norm(feature, dim=-1, keepdim=True).clamp(min=1e-8)
    row, col = edge_index[0], edge_index[1]
    edge_sim = _edge_sim_chunked(feature, row, col, chunk_size)
    degree = torch.zeros(n, device=feature.device, dtype=feature.dtype)
    degree.scatter_add_(0, row, torch.ones_like(row, dtype=feature.dtype, device=feature.device))
    message = torch.zeros(n, device=feature.device, dtype=feature.dtype)
    message.scatter_add_(0, row, edge_sim)
    return message / degree.clamp(min=1)


def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.mm(emb, emb.T) * (1 - adj)
    row_sum = torch.sum(1 - adj, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    return torch.sum(torch.sum(sim, 1) * r_inv)


# ═══════════════════════════════════════════════════════════════════════════
# Data conversion: PyG Data → TAM format
# ═══════════════════════════════════════════════════════════════════════════

def pyg_to_tam_format(data, device="cpu", preprocess_feats=False, use_sparse=False):
    """Convert PyG Data to TAM format. use_sparse=True avoids n×n dense adj allocation."""
    n = data.x.shape[0]

    # Features
    x = data.x.numpy() if not isinstance(data.x, np.ndarray) else data.x
    if sp.issparse(x):
        x = np.array(x.todense())
    if preprocess_feats and sp.issparse(data.x):
        rowsum = np.array(x.sum(1)).flatten()
        r_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
        x = np.diag(r_inv) @ x

    # Adjacency (with self-loops)
    ei = data.edge_index.numpy()
    adj_sp = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    adj_sp = adj_sp + sp.eye(n)
    adj_coo = adj_sp.tocoo()
    edge_index = torch.LongTensor(np.stack([adj_coo.row, adj_coo.col])).to(device)

    features = torch.FloatTensor(x[np.newaxis]).to(device)
    labels = data.y.numpy().flatten()

    if use_sparse:
        return features, edge_index, labels
    adj_dense = np.array(adj_sp.todense(), dtype=np.float32)
    raw_adj = torch.FloatTensor(adj_dense[np.newaxis]).to(device)
    return features, raw_adj, labels, edge_index


# ═══════════════════════════════════════════════════════════════════════════
# Single TAM trial
# ═══════════════════════════════════════════════════════════════════════════

def run_tam_trial(
    data,
    device="cpu",
    cutting=18,
    n_tree=3,
    num_epoch=500,
    lr=1e-5,
    embedding_dim=128,
    lamda=0,
    verbose=False,
    return_scores=False,
):
    """Run one TAM trial.  Returns dict with auc, auprc. If return_scores=True, returns (scores, y) instead."""
    dev = torch.device(device)

    # Threshold: use sparse + O(E) for large graphs to avoid n×n allocation
    _LARGE_GRAPH_THRESHOLD = 10_000
    _HUGE_GRAPH_THRESHOLD = 500_000  # Use rev_edge_idx + chunked ops to avoid OOM

    preprocess = hasattr(data, '_dataset_name') and data._dataset_name in ('Amazon', 'YelpChi')
    nb_nodes = data.x.shape[0]
    use_sparse = nb_nodes >= _LARGE_GRAPH_THRESHOLD
    use_memory_savers = nb_nodes >= _HUGE_GRAPH_THRESHOLD

    if use_sparse:
        features, edge_index, ano_label = pyg_to_tam_format(data, device=dev, use_sparse=True)
    else:
        features, raw_adj, ano_label, edge_index = pyg_to_tam_format(data, device=dev)

    nb_nodes = features.shape[1]
    ft_size = features.shape[2]
    use_edge_dist = nb_nodes >= _LARGE_GRAPH_THRESHOLD

    # Distance: O(E) for large graphs, O(n^2) for small
    if verbose:
        print(f"  Computing distances (n={nb_nodes}, {'sparse' if use_sparse else 'dense'} mode)...")
    if use_edge_dist:
        edge_dist = calc_distance_edges(edge_index, features[0])
    else:
        dis_array = calc_distance(raw_adj[0], features[0])

    # Initialize models
    models, optims = [], []
    for _ in range(cutting * n_tree):
        m = Model(ft_size, embedding_dim, "prelu", 2, "avg").to(dev)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        models.append(m)
        optims.append(o)

    if use_sparse:
        all_cut_edge_masks = [torch.ones(edge_index.shape[1], dtype=torch.bool, device=dev) for _ in range(n_tree)]
        rev_edge_idx = _build_rev_edge_index(edge_index, nb_nodes) if use_memory_savers else None
        chunk_size = 500_000 if use_memory_savers else 1_000_000
    else:
        all_cut_adj = torch.cat([raw_adj.clone() for _ in range(n_tree)])
    idx = 0
    message_mean_list = []

    for n_cut in range(cutting):
        message_list = []
        for n_t in range(n_tree):
            if use_sparse:
                graph_nsgt_edges_sparse(edge_index, edge_dist, all_cut_edge_masks[n_t], nb_nodes, rev_edge_idx)
                adj_norm = normalize_adj_sparse(edge_index, all_cut_edge_masks[n_t], nb_nodes, dev)
            else:
                cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t])
                cut_adj_b = cut_adj.unsqueeze(0)
                adj_norm = normalize_adj_tensor(cut_adj_b)
                all_cut_adj[n_t] = cut_adj

            optims[idx].zero_grad()
            models[idx].train()

            for epoch in range(num_epoch):
                if use_sparse:
                    node_emb, feat1, feat2 = models[idx](features, adj_norm, sparse=True)
                    loss, _ = max_message_sparse(node_emb[0], edge_index, nb_nodes, chunk_size)
                    msg = inference_sparse(node_emb[0], edge_index, nb_nodes, chunk_size)
                    rl = torch.tensor(0.0, device=dev)
                else:
                    node_emb, feat1, feat2 = models[idx](features, adj_norm)
                    loss, _ = max_message(node_emb[0], raw_adj[0])
                    msg = inference(node_emb[0], raw_adj[0])
                    rl = reg_edge(feat1[0], raw_adj[0]) if lamda != 0 else torch.tensor(0.0, device=dev)
                total_loss = loss + lamda * rl
                total_loss.backward()
                optims[idx].step()
                optims[idx].zero_grad()

            message_list.append(msg.unsqueeze(0))
            idx += 1

        message_agg = torch.mean(torch.cat(message_list), 0)
        message_mean_list.append(message_agg.unsqueeze(0))

    final_msg = torch.mean(torch.cat(message_mean_list), 0)
    score_np = 1.0 - normalize_score(final_msg.cpu().detach().numpy())

    # Mask to labeled nodes only (0=normal, 1=anomaly); -1=unlabeled
    mask = (ano_label >= 0) & (ano_label <= 1)
    if mask.sum() == 0:
        return {"auc": 0.0, "ap": 0.0, "auprc": 0.0}
    y_eval = (ano_label[mask] > 0).astype(int)
    score_eval = score_np[mask]

    auc = float(roc_auc_score(y_eval, score_eval))
    ap = float(average_precision_score(y_eval, score_eval))
    p, r, _ = precision_recall_curve(y_eval, score_eval)
    auprc = float(sk_auc(r, p))

    if verbose:
        print(f"  TAM trial: AUROC={auc:.4f}  AUPRC={auprc:.4f}")

    if return_scores:
        return {"auc": auc, "ap": ap, "auprc": auprc, "scores": score_eval, "y": y_eval}
    return {"auc": auc, "ap": ap, "auprc": auprc}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run TAM baseline")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cutting", type=int, default=18)
    parser.add_argument("--n-tree", type=int, default=3)
    parser.add_argument("--num-epoch", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None, help="Override embedding dim (default 64 for dgraph)")
    args = parser.parse_args()

    # DGraph-specific defaults to fit in 80GB GPU (with labeled subgraph)
    if args.dataset.lower() == "dgraph":
        if args.cutting == 18:
            args.cutting = 6
        if args.n_tree == 3:
            args.n_tree = 2
        if args.embedding_dim is None:
            args.embedding_dim = 64
    if args.embedding_dim is None:
        args.embedding_dim = 128

    from data_utils import load_data, extract_labeled_subgraph

    data = load_data(args.dataset)
    n = data.x.shape[0]
    # DGraph: use labeled subgraph (~1.2M nodes) to avoid OOM on 80GB GPU
    if args.dataset.lower() == "dgraph" and n > 1_000_000:
        data, _ = extract_labeled_subgraph(data)
        n = data.x.shape[0]
        print(f"Dataset: {args.dataset} (labeled subgraph: n={n}, d={data.x.shape[1]})")
    else:
        print(f"Dataset: {args.dataset}  (n={n}, d={data.x.shape[1]})")

    output_file = args.output or f"results/tam_{args.dataset}.jsonl"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    aucs, auprcs = [], []
    t0 = time.time()
    for trial in range(args.trials):
        print(f"Trial {trial + 1}/{args.trials}")
        r = run_tam_trial(
            data, device=args.device,
            cutting=args.cutting, n_tree=args.n_tree,
            num_epoch=args.num_epoch, embedding_dim=args.embedding_dim,
            verbose=True,
        )
        aucs.append(r["auc"])
        auprcs.append(r["auprc"])
    elapsed = time.time() - t0

    result = {
        "model": "tam",
        "params": {"dataset": args.dataset, "cutting": args.cutting,
                   "n_tree": args.n_tree, "num_epoch": args.num_epoch,
                   "embedding_dim": args.embedding_dim},
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "elapsed_s": elapsed,
        "num_trials": args.trials,
    }

    with open(output_file, "w") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\nTAM on {args.dataset}:  "
          f"AUROC={result['auc_mean']:.4f}±{result['auc_std']:.4f}  "
          f"AUPRC={result['auprc_mean']:.4f}±{result['auprc_std']:.4f}  "
          f"({elapsed:.1f}s)")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
