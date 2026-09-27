"""Probe: the edge leg in the graph-aligned feature subspace.

The edge leg reads the cosine between neighbors' feature vectors. On BlogCatalog the features carry almost no
homophily as a whole (median neighbor cosine -0.002 in PCA-256 space), so the leg is blind, while TAM's learned
affinity (82.5) is not. A label-free, closed-form analogue of a learned affinity space: the feature directions u
that are most homophilous, i.e. the top generalized eigenvectors of  X^T A_n X u = mu X^T X u  (A_n the
normalized adjacency), the graph-aligned principal directions. Projecting the features on the top-k such
directions and reading the edge leg there asks whether a node agrees with its neighbors along the directions
where normal nodes do. Controls: the plain edge leg (k = all), and plain PCA top-k (not aligned).
Usage: edge_leg_aligned_probe.py blogcatalog:256 acm:256 facebook t_finance weibo amazon [--ks 8 16 32 64]
"""
import sys, argparse, numpy as np, scipy.sparse as sp, scipy.linalg as sla
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from ebgad.edgeleg import edge_leg, undirected_edges
from run_ebgad import eval_mask

A = lambda y, z: 100 * roc_auc_score(y, z)
ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
args = ap.parse_args()
print("%-16s | %-13s | %s" % ("dataset", "edge leg D/C", " | ".join("aligned k=%d D/C | pca k=%d D/C" % (k, k) for k in args.ks)), flush=True)
for arg in args.datasets:
    name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    x = preprocess_features(data.x, data.edge_index, pca=pca).double().numpy(); n, d = x.shape
    ia, ib = undirected_edges(data.edge_index.numpy(), n)
    W = sp.coo_matrix((np.ones(2 * len(ia)), (np.concatenate([ia, ib]), np.concatenate([ib, ia]))), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(1)).ravel(); dis = 1.0 / np.sqrt(np.maximum(deg, 1.0)); An = sp.diags(dis) @ W @ sp.diags(dis)
    G = x.T @ (An @ x); G = 0.5 * (G + G.T); C = x.T @ x + 1e-6 * np.trace(x.T @ x) / d * np.eye(d)
    mu, U = sla.eigh(G, C)                                 # generalized eigenproblem; ascending mu
    order = np.argsort(-mu); mu, U = mu[order], U[:, order]
    Cx = x.T @ x; ev, P = np.linalg.eigh(Cx); P = P[:, np.argsort(-ev)]
    e0 = edge_leg(x, data.edge_index.numpy())["e"]
    cols = [name.ljust(16), "%5.1f / %5.1f" % (A(y, -e0[m]), A(y, e0[m]))]
    for k in args.ks:
        k = min(k, d)
        za = x @ U[:, :k]; ea = edge_leg(za, data.edge_index.numpy())["e"]
        zp = x @ P[:, :k]; ep = edge_leg(zp, data.edge_index.numpy())["e"]
        cols.append("%5.1f / %5.1f | %5.1f / %5.1f" % (A(y, -ea[m]), A(y, ea[m]), A(y, -ep[m]), A(y, ep[m])))
    print(" | ".join(cols) + "  | top mu %.3f, #mu>0.1: %d" % (mu[0], int((mu > 0.1).sum())), flush=True)
