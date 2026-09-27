"""
Shared data loading utilities.

Supports three dataset sources:

1. **PyGOD datasets** (disney, weibo, reddit, books, enron, inj_cora, inj_amazon, ...)
   Loaded via ``pygod.utils.load_data``.

2. **TAM .mat datasets** (ACM, Amazon, BlogCatalog, Facebook, YelpChi, Amazon-all,
   YelpChi-all, t_finance, ogbn_proteins)
   Loaded from ``data/<name>.mat`` files containing adjacency matrices, features,
   and labels.  Converted to PyG ``Data`` objects for compatibility.

3. **Fraud datasets** (elliptic, elliptic_plus_plus, dgraph)
   - elliptic, elliptic_plus_plus: CSV files in ``fraud_data/<name>/``
   - dgraph: ``fraud_data/dgraph/dgraphfin.npz`` (x, y, edge_index, etc.)

All datasets are returned as ``torch_geometric.data.Data`` with at minimum:
``data.x`` (node features), ``data.edge_index`` (COO edges), ``data.y`` (labels).
"""

import os
import torch

# ── Register safe globals for PyTorch 2.6+ / PyG data objects ──
try:
    import torch_geometric.data.storage as _pyg_storage
    import torch_geometric.data.data as _pyg_data
    import torch_geometric.data.batch as _pyg_batch

    torch.serialization.add_safe_globals([
        _pyg_storage.GlobalStorage,
        _pyg_storage.NodeStorage,
        _pyg_storage.EdgeStorage,
        _pyg_data.Data,
        _pyg_batch.Batch,
    ])
except Exception:
    pass

# .mat datasets that live in data/<name>.mat
MAT_DATASETS = {
    "ACM", "Amazon", "BlogCatalog", "Facebook", "YelpChi",
    "Amazon-all", "YelpChi-all", "t_finance", "ogbn_proteins",
    "Cora", "CiteSeer", "PubMed", "Flickr",   # CoLA-protocol injected anomalies (files from the PREM-GAD repo; ACM.mat there is byte-identical to ours)
}
# Case-insensitive lookup
_MAT_LOWER = {n.lower(): n for n in MAT_DATASETS}

# Fraud datasets in fraud_data/
FRAUD_DATASETS = {"elliptic", "elliptic_plus_plus", "dgraph", "tsocial"}
_FRAUD_LOWER = {n.lower(): n for n in FRAUD_DATASETS}

# Heterophilous benchmark datasets with organic binary anomaly labels
# (Platonov et al.; used by GADBench). Loaded via torch_geometric.
HETERO_DATASETS = {"tolokers": "Tolokers", "questions": "Questions",
                   "minesweeper": "Minesweeper"}


def load_hetero_data(name: str):
    """Load a HeterophilousGraphDataset as a plain GAD Data object.

    y is binary (1 = anomalous class: banned toloker / unanswered question).
    Split masks are dropped; the pipeline is unsupervised.
    """
    from torch_geometric.data import Data
    from torch_geometric.datasets import HeterophilousGraphDataset

    canonical = HETERO_DATASETS[name.lower()]
    root = os.path.join(os.path.dirname(__file__), "data", "heterophilous")
    raw = HeterophilousGraphDataset(root, name=canonical)[0]
    data = Data(x=raw.x.float(), edge_index=raw.edge_index, y=raw.y.long())
    data.num_nodes = raw.x.shape[0]
    return data


def load_mat_data(name: str):
    """Load a .mat dataset and return a PyG-compatible Data object.

    Looks for ``data/<name>.mat`` with keys:
    - ``Network`` or ``A``: adjacency (sparse)
    - ``Attributes`` or ``X``: node features (sparse or dense)
    - ``Label`` or ``gnd``: anomaly labels
    """
    import numpy as np
    import scipy.io as sio
    import scipy.sparse as sp
    from torch_geometric.data import Data

    canonical = _MAT_LOWER.get(name.lower(), name)
    path = os.path.join(os.path.dirname(__file__), "data", f"{canonical}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data_dict = sio.loadmat(path)

    label = data_dict["Label"] if "Label" in data_dict else data_dict["gnd"]
    attr = data_dict["Attributes"] if "Attributes" in data_dict else data_dict["X"]
    network = data_dict["Network"] if "Network" in data_dict else data_dict["A"]

    # Labels
    y = torch.from_numpy(np.squeeze(np.array(label))).long()

    # Features: sparse → dense, row-normalize for large sparse feature matrices
    if sp.issparse(attr):
        if canonical in ("Amazon", "YelpChi", "Amazon-all", "YelpChi-all"):
            rowsum = np.array(attr.sum(1)).flatten()
            r_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
            attr = sp.diags(r_inv).dot(attr)
        x = torch.from_numpy(np.array(attr.todense())).float()
    else:
        x = torch.from_numpy(np.array(attr)).float()

    # Adjacency → edge_index (COO)
    adj = sp.coo_matrix(network)
    edge_index = torch.stack([
        torch.from_numpy(adj.row.astype(np.int64)),
        torch.from_numpy(adj.col.astype(np.int64)),
    ], dim=0)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = x.shape[0]

    # Store sub-type labels if available
    if "str_anomaly_label" in data_dict:
        data.str_anomaly_label = torch.from_numpy(
            np.squeeze(np.array(data_dict["str_anomaly_label"]))
        ).long()
        data.attr_anomaly_label = torch.from_numpy(
            np.squeeze(np.array(data_dict["attr_anomaly_label"]))
        ).long()

    return data


def load_dgraph_data():
    """Load DGraph-Fin from fraud_data/dgraph/dgraphfin.npz.

    Labels: 0=normal, 1=fraud (anomaly), -1=background (classes 2,3).
    Also stores train_mask, valid_mask, test_mask for semi-supervised splits.
    """
    import numpy as np
    from torch_geometric.data import Data

    path = os.path.join(os.path.dirname(__file__), "fraud_data", "dgraph", "dgraphfin.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DGraph file not found: {path}")

    z = np.load(path, allow_pickle=True)
    x = torch.from_numpy(z["x"]).float()
    raw_y = z["y"]
    # 0=normal, 1=fraud; 2,3=background -> -1 (unlabeled for GAD eval)
    y = np.where((raw_y == 0) | (raw_y == 1), raw_y, -1)
    y = torch.from_numpy(y.astype(np.int64)).long()

    edge_index = torch.from_numpy(z["edge_index"].T.astype(np.int64)).contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = x.shape[0]
    data.train_mask = torch.from_numpy(z["train_mask"])
    data.valid_mask = torch.from_numpy(z["valid_mask"])
    data.test_mask = torch.from_numpy(z["test_mask"])
    return data


def load_tsocial_data():
    """Load T-Social (BWGNN release converted by tools/convert_tsocial.py): all nodes labeled, 0 normal / 1 anomaly."""
    import numpy as np
    from torch_geometric.data import Data

    path = os.path.join(os.path.dirname(__file__), "fraud_data", "tsocial", "tsocial.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"T-Social file not found: {path}")
    z = np.load(path)
    x = torch.from_numpy(z["x"]).float(); y = torch.from_numpy(z["y"].astype(np.int64)).long()
    edge_index = torch.from_numpy(z["edge_index"].astype(np.int64)).contiguous()
    data = Data(x=x, edge_index=edge_index, y=y); data.num_nodes = x.shape[0]
    return data


def load_fraud_data(name: str):
    """Load a fraud dataset from fraud_data/.

    - dgraph: dgraphfin.npz (0=normal, 1=fraud, -1=background)
    - elliptic, elliptic_plus_plus: CSV files (0=licit, 1=illicit, -1=unknown)
    """
    if name.lower() == "dgraph":
        return load_dgraph_data()
    if name.lower() == "tsocial":
        return load_tsocial_data()

    import csv
    import numpy as np
    from torch_geometric.data import Data

    canonical = _FRAUD_LOWER.get(name.lower(), name)
    base = os.path.join(os.path.dirname(__file__), "fraud_data", canonical)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Fraud dataset directory not found: {base}")

    if canonical == "elliptic":
        feat_path = os.path.join(base, "elliptic_txs_features.csv")
        class_path = os.path.join(base, "elliptic_txs_classes.csv")
        edge_path = os.path.join(base, "elliptic_txs_edgelist.csv")
        has_header = False
        feat_col_start = 1  # col 0 = txId
    else:  # elliptic_plus_plus
        feat_path = os.path.join(base, "txs_features.csv")
        class_path = os.path.join(base, "txs_classes.csv")
        edge_path = os.path.join(base, "txs_edgelist.csv")
        has_header = True
        feat_col_start = 2  # col 0 = txId, col 1 = Time step

    # Load features and build txId -> index mapping
    txid_to_idx = {}
    rows = []
    with open(feat_path) as f:
        r = csv.reader(f)
        if has_header:
            next(r)
        for idx, row in enumerate(r):
            txid = row[0]
            txid_to_idx[txid] = idx
            feat = []
            for x in row[feat_col_start:]:
                try:
                    feat.append(float(x) if x.strip() else 0.0)
                except (ValueError, AttributeError):
                    feat.append(0.0)
            rows.append(feat)
    x = torch.from_numpy(np.array(rows, dtype=np.float32))
    n_nodes = x.shape[0]

    # Load labels: 0=licit, 1=illicit, -1=unknown
    y = torch.full((n_nodes,), -1, dtype=torch.long)
    with open(class_path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            txid, cls = row[0], row[1]
            if txid not in txid_to_idx:
                continue
            idx = txid_to_idx[txid]
            if cls in ("1", "illicit"):
                y[idx] = 1
            elif cls in ("2", "licit"):
                y[idx] = 0
            # else: unknown, keep -1 (elliptic uses "unknown"; elliptic++ uses "3")

    # Load edges (only include if both endpoints exist)
    edges = []
    with open(edge_path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            a, b = row[0], row[1]
            if a in txid_to_idx and b in txid_to_idx:
                edges.append([txid_to_idx[a], txid_to_idx[b]])
    if not edges:
        raise ValueError(f"No valid edges found in {edge_path}")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = n_nodes
    return data


def extract_labeled_subgraph(data):
    """Extract the subgraph of labeled nodes (y >= 0) from a PyG Data object.

    For datasets like DGraph where 67% of nodes are background (y=-1),
    training on the full graph wastes compute and dilutes the spectral
    decomposition.  This returns a new Data with only labeled nodes and
    edges between them, plus a ``labeled_mask`` on the original graph.

    Returns (sub_data, labeled_mask) where labeled_mask is a bool tensor
    of shape (n_original,) mapping back to the original node indices.
    """
    from torch_geometric.data import Data

    y = data.y
    labeled_mask = (y >= 0) & (y <= 1)
    labeled_idx = labeled_mask.nonzero(as_tuple=True)[0]
    n_sub = labeled_idx.shape[0]

    # Re-index: old_idx -> new_idx (-1 if not labeled)
    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    remap[labeled_idx] = torch.arange(n_sub, dtype=torch.long)

    # Filter edges: keep only edges where both endpoints are labeled
    src, dst = data.edge_index[0], data.edge_index[1]
    keep = labeled_mask[src] & labeled_mask[dst]
    new_src = remap[src[keep]]
    new_dst = remap[dst[keep]]
    new_edge_index = torch.stack([new_src, new_dst], dim=0)

    sub_data = Data(
        x=data.x[labeled_idx],
        edge_index=new_edge_index,
        y=data.y[labeled_idx],
    )
    sub_data.num_nodes = n_sub
    return sub_data, labeled_mask


def extract_labeled_with_background(data):
    """Extract labeled subgraph but KEEP background neighbors (H2).

    Unlike :func:`extract_labeled_subgraph`, this keeps background nodes
    (y=-1) that are directly connected to labeled nodes.  This restores
    neighborhood context for the 50% of fraud nodes that would otherwise
    be isolated.

    Background nodes get y=-1 in the subgraph and are masked at eval time.
    """
    from torch_geometric.data import Data

    y = data.y
    labeled_mask = (y >= 0) & (y <= 1)
    src, dst = data.edge_index[0], data.edge_index[1]

    bg_mask = (y < 0) | (y > 1)
    bg_with_labeled_neighbor = torch.zeros(data.num_nodes, dtype=torch.bool)
    bg_with_labeled_neighbor[dst[labeled_mask[src] & bg_mask[dst]]] = True
    bg_with_labeled_neighbor[src[labeled_mask[dst] & bg_mask[src]]] = True

    keep_mask = labeled_mask | bg_with_labeled_neighbor
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    n_sub = keep_idx.shape[0]

    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    remap[keep_idx] = torch.arange(n_sub, dtype=torch.long)

    keep_edges = keep_mask[src] & keep_mask[dst]
    new_edge_index = torch.stack([remap[src[keep_edges]], remap[dst[keep_edges]]], dim=0)

    new_y = data.y[keep_idx].clone()
    new_y[~labeled_mask[keep_idx]] = -1

    sub_data = Data(x=data.x[keep_idx], edge_index=new_edge_index, y=new_y)
    sub_data.num_nodes = n_sub
    sub_data.labeled_mask_sub = labeled_mask[keep_idx]
    return sub_data, labeled_mask


def apply_edge_type_weights(data, edge_type_weights=None):
    """Weight edges by type for edge-type-aware Laplacian (H3).

    Loads ``edge_type`` from the DGraph .npz file and stores
    ``data.edge_weight`` as a float tensor.  If ``edge_type_weights``
    is a dict mapping type_id -> weight, those weights are applied;
    otherwise all types get weight 1.0.
    """
    import numpy as np
    path = os.path.join(os.path.dirname(__file__), "fraud_data", "dgraph", "dgraphfin.npz")
    z = np.load(path, allow_pickle=True)
    edge_type_raw = z["edge_type"].flatten()

    if edge_type_weights is None:
        edge_type_weights = {}

    weights = np.ones(len(edge_type_raw), dtype=np.float32)
    for etype, w in edge_type_weights.items():
        weights[edge_type_raw == int(etype)] = float(w)

    data.edge_weight_raw = torch.from_numpy(weights).float()
    return data


def augment_structural_features(data, include_background_edges=False):
    """Append structural features to node feature matrix (H4).

    Computes per-node:
    - degree (in labeled subgraph)
    - neighbor feature mean (per feature dim)
    - neighbor feature std (per feature dim)

    If ``include_background_edges`` is True and the original DGraph data
    is available, also computes:
    - background-neighbor degree
    - background-neighbor ratio

    Resulting features are appended to ``data.x``.
    """
    n = data.num_nodes
    x = data.x
    d = x.shape[1]
    src, dst = data.edge_index[0], data.edge_index[1]

    degree = torch.zeros(n, dtype=torch.float32)
    neighbor_sum = torch.zeros(n, d, dtype=torch.float32)
    neighbor_sq_sum = torch.zeros(n, d, dtype=torch.float32)

    degree.index_add_(0, src, torch.ones(src.shape[0]))
    degree.index_add_(0, dst, torch.ones(dst.shape[0]))

    neighbor_sum.index_add_(0, src, x[dst])
    neighbor_sum.index_add_(0, dst, x[src])

    neighbor_sq_sum.index_add_(0, src, x[dst] ** 2)
    neighbor_sq_sum.index_add_(0, dst, x[src] ** 2)

    safe_deg = degree.clamp(min=1).unsqueeze(-1)
    neighbor_mean = neighbor_sum / safe_deg
    neighbor_var = (neighbor_sq_sum / safe_deg - neighbor_mean ** 2).clamp(min=0)
    neighbor_std = neighbor_var.sqrt()

    new_feats = [
        x,
        degree.unsqueeze(-1) / degree.max().clamp(min=1),
        neighbor_mean,
        neighbor_std,
    ]

    data.x = torch.cat(new_feats, dim=-1)
    return data


def load_data(name: str):
    """Load a dataset by name.  Dispatches to .mat, fraud CSV, or PyGOD.

    Parameters
    ----------
    name : dataset name (case-insensitive).
        .mat datasets: ACM, Amazon, BlogCatalog, Facebook, YelpChi,
            Amazon-all, YelpChi-all, t_finance, ogbn_proteins
        Fraud CSV: elliptic, elliptic_plus_plus (from fraud_data/)
        PyGOD datasets: disney, weibo, reddit, books, enron,
            inj_cora, inj_amazon, ...
    """
    if name.lower() in _MAT_LOWER:
        return load_mat_data(name)
    if name.lower() in _FRAUD_LOWER:
        return load_fraud_data(name)
    if name.lower() in HETERO_DATASETS:
        return load_hetero_data(name)

    from pygod.utils import load_data as _load
    try:
        data = _load(name)
        if name.lower().startswith("inj_"):      # BOND's injected graphs: y is a bit mask (1 contextual, 2 structural, 3 both) -> outlier / not
            data.y = (data.y != 0).long()
        return data
    except (KeyError, RuntimeError, EOFError) as e:
        import shutil
        print(f"[WARN] load_data('{name}') failed: {e}")
        print("       Clearing PyGOD cache and retrying...")
        # Only PyGOD's own download caches. ./data holds the .mat, T-Finance and
        # heterophilous datasets and must never be removed here (it was, on
        # 2026-09-18, by a mistyped dataset name).
        for cache_dir in ["~/.cache/pygod", "~/.pygod"]:
            p = os.path.expanduser(cache_dir)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                print(f"       Removed {p}")
        return _load(name)
