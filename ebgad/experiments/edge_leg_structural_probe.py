"""Edge leg on the connections source: the same statistic (ebgad/edgeleg.py) applied to the 64-column Gaussian
sketch X_s = A_n R of the normalized-adjacency rows (structural_leg_test.py's second source).  The cosine of two
neighbors' sketch rows is the overlap of their neighborhoods, so the leg asks whether a node's neighbors share
its neighborhood more or less than typical edges do.  Usage: edge_leg_structural_probe.py weibo yelpchi ...
"""
import sys, numpy as np, scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.edgeleg import edge_leg, undirected_edges
from run_ebgad import eval_mask

A = lambda y, z: 100 * roc_auc_score(y, z)
print("%-14s | %6s | %-13s | %-13s" % ("dataset", "deg", "struct-edge D/C", "two-sided"))
for name in sys.argv[1:]:
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]; n = data.num_nodes
    ia, ib = undirected_edges(data.edge_index.numpy(), n)
    W = sp.coo_matrix((np.ones(2 * len(ia)), (np.concatenate([ia, ib]), np.concatenate([ib, ia]))), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(1)).ravel(); dis = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    An = sp.diags(dis) @ W @ sp.diags(dis)
    rng = np.random.default_rng(0); R = rng.standard_normal((n, 64)) / 8.0
    Xs = np.asarray(An @ R); Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    res = edge_leg(Xs, data.edge_index.numpy()); e = res["e"]
    print("%-14s | %6.1f | %5.1f / %5.1f | %5.1f" % (name, deg.mean(), A(y, -e[m]), A(y, e[m]), A(y, np.abs(e[m]))), flush=True)
