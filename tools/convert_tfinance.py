"""Convert the BWGNN T-Finance DGL graph to data/t_finance.mat.

Run with an environment that has dgl (torch 2.3.1 with dgl 2.2.1 works on macOS arm64):

    python tools/convert_tfinance.py <path/to/tfinance> data/t_finance.mat

Output keys match data_utils.load_mat_data: Network (sparse adjacency,
directed edge list as stored by DGL, both directions present), Attributes
(dense n x 10 float32), Label (argmax of the one-hot label, 1 = fraud).
"""
import sys

import numpy as np

# DGL 2.2.1 still references numpy aliases removed in numpy 1.24.
for _alias, _target in (("long", int), ("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import scipy.io as sio
import scipy.sparse as sp
from dgl.data.utils import load_graphs

src, dst_path = sys.argv[1], sys.argv[2]
graphs, _ = load_graphs(src)
g = graphs[0]
n = g.num_nodes()
feat = g.ndata["feature"].numpy().astype(np.float32)
label_raw = g.ndata["label"]
label = label_raw.argmax(1).numpy() if label_raw.dim() == 2 else label_raw.numpy()
u, v = (t.numpy() for t in g.edges())
A = sp.coo_matrix((np.ones(len(u), dtype=np.float32), (u, v)), shape=(n, n)).tocsr()
print(f"nodes {n} feat {feat.shape[1]} directed edges {A.nnz} fraud {int(label.sum())} ({100 * label.mean():.2f}%)")
sio.savemat(dst_path, {"Network": A, "Attributes": feat, "Label": label.reshape(-1, 1).astype(np.int64)}, do_compression=True)
print("wrote", dst_path)
