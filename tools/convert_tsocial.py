"""Convert the BWGNN release of T-Social (a DGL save_graphs binary, cache/tsocial/.../tsocial) to
fraud_data/tsocial/tsocial.npz with x (float32), y (int64: 0 normal, 1 anomaly) and edge_index (2 x E, int64),
the format of our DGraph loader.  Run in an environment with DGL:
  python tools/convert_tsocial.py <path/to/tsocial>"""
import sys, os, numpy as np
from dgl import load_graphs
src = sys.argv[1]; out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fraud_data", "tsocial", "tsocial.npz")
g = load_graphs(src)[0][0]
print(g)
x = g.ndata["feature"].numpy().astype(np.float32); y = g.ndata["label"].numpy().astype(np.int64)
u, v = g.edges(); ei = np.vstack([u.numpy(), v.numpy()]).astype(np.int64)
os.makedirs(os.path.dirname(out), exist_ok=True); np.savez(out, x=x, y=y, edge_index=ei)
print("saved", out, "n", x.shape[0], "d", x.shape[1], "edges", ei.shape[1], "anomalies", int(y.sum()), "%.2f%%" % (100 * y.mean()))
