"""Unlabeled graph statistics read by the selector: feature homophily on raw and on z-scored features."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
from data_utils import load_data

NAMES = {"yelpchi": "YelpChi", "facebook": "Facebook", "blogcatalog": "BlogCatalog"}


def homophily(x, edge_index, seed=42, n_sample=100000):
    ei = edge_index
    rng = np.random.RandomState(seed)
    idx = rng.choice(ei.shape[1], min(n_sample, ei.shape[1]), replace=False)
    a, b = x[ei[0, idx]], x[ei[1, idx]]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(num / den))


out = {}
for ds in sys.argv[1].split(","):
    data = load_data(NAMES.get(ds, ds))
    x = data.x.float().numpy().astype(np.float64)
    ei = data.edge_index.numpy()
    z = (x - x.mean(0)) / (x.std(0) + 1e-8)
    n = x.shape[0]
    und = np.unique(np.sort(ei, axis=0), axis=1).shape[1]
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    adj = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    n_comp, lab = connected_components(adj, directed=False)
    out[ds] = {"components": int(n_comp), "largest_component": int(np.bincount(lab).max()),
               "n": n, "D_x": x.shape[1], "edges_directed": int(ei.shape[1]), "edges_undirected": int(und),
               "d_e": ei.shape[1] // 2 / n, "h_raw": homophily(x, ei), "h_z": homophily(z, ei)}
    print(ds, out[ds], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
