"""Feasibility probe for an *edge leg*: is a node dissimilar from its neighbors, edge by edge?

Motivation: the injected-outlier benchmarks (ACM, BlogCatalog, Amazon) are half
contextual anomalies (features swapped with a far node: the attribute leg catches them) and half
structural anomalies (cliques among random nodes: normal features, edges to dissimilar nodes).
The local leg s_i compares x_i with the *mean* of its neighborhood, which is unremarkable for a
clique member; TAM / DOMINANT see cliques through per-edge dissimilarity / unreconstructable edges.
This probe measures, uncalibrated, whether per-edge dissimilarity separates the anomalies at all
before the null-calibrated version (per-edge chi^2_d under the fitted model) is built.

Statistics per node i (standardized features x, d columns, sym-normalized geometry x~_i = x_i / sqrt(deg_i)):
  rough    : mean_j ||x_i - x_j||^2 / d                      (plain roughness)
  energy   : sum_j ||x~_i - x~_j||^2 / d                     (node's share of x^T L x)
  affinity : mean_j cos(x_i, x_j)                             (TAM-like; anomalies expected LOW)
  ratio    : rough / (||x_i||^2/d + mean_j ||x_j||^2/d)       (roughness relative to the norms; shape-like)
AUROC in both tails on the evaluated nodes.  Usage: edge_leg_probe.py acm:256 blogcatalog:256 amazon ...
"""
import sys, numpy as np, scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from run_ebgad import eval_mask


def au(y, z):
    return 100 * roc_auc_score(y, z), 100 * average_precision_score(y, z)


def edges_undirected(edge_index, n):
    ei = np.asarray(edge_index)
    a = np.minimum(ei[0], ei[1]); b = np.maximum(ei[0], ei[1])
    m = a != b
    a, b = a[m], b[m]
    key = a.astype(np.int64) * n + b
    key = np.unique(key)
    return key // n, key % n


print("%-14s | %6s | %-13s | %-13s | %-13s | %-13s | %-13s" % ("dataset", "deg", "rough DEV/CAM", "energy DEV/CAM", "affin DEV/CAM", "ratio DEV/CAM", "no-graph DEV/CAM"))
for arg in sys.argv[1:]:
    name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
    data = load_data(name); y_all = data.y.numpy().astype(int); m_ = eval_mask(y_all); y = y_all[m_]
    xt = preprocess_features(data.x, data.edge_index, pca=pca); x = xt.double().numpy(); n, d = x.shape
    a, b = edges_undirected(data.edge_index.numpy(), n)
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    deg_s = np.maximum(deg, 1).astype(float)
    xn = x / np.sqrt(deg_s)[:, None]
    norm2 = (x * x).sum(1) / d
    rough_sum = np.zeros(n); energy = np.zeros(n); aff_sum = np.zeros(n); nbr_norm = np.zeros(n)
    xu = x / np.maximum(np.sqrt(norm2 * d), 1e-12)[:, None]
    CH = 2_000_000
    for s in range(0, len(a), CH):
        ia, ib = a[s:s + CH], b[s:s + CH]
        dd = ((x[ia] - x[ib]) ** 2).sum(1) / d
        ee = ((xn[ia] - xn[ib]) ** 2).sum(1) / d
        cs = (xu[ia] * xu[ib]).sum(1)
        for u, v in ((ia, ib), (ib, ia)):
            np.add.at(rough_sum, u, dd); np.add.at(energy, u, ee); np.add.at(aff_sum, u, cs); np.add.at(nbr_norm, u, norm2[v])
    rough = rough_sum / deg_s; aff = aff_sum / deg_s; nbr = nbr_norm / deg_s
    ratio = rough / (norm2 + nbr + 1e-12)
    out = [name.ljust(14), "%6.1f" % deg.mean()]
    for z in (rough, energy, aff, ratio, norm2):
        zz = z[m_]
        out.append("%5.1f / %5.1f" % (au(y, zz)[0], au(y, -zz)[0]))
    print(" | ".join(out), flush=True)
