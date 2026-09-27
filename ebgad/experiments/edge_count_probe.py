"""Aggregation probe for the edge leg: mean of standardized edge z's (current, e_i) vs a count of extreme edges
with an exact binomial null.  A clique member on a high-degree graph (BlogCatalog: mean degree 66, cliques of
~15) has a minority of anomalous edges, which the mean dilutes; the count of edges below the population's
q-quantile is Binomial(deg_i, q) under independence, so c_i = -log P(Bin(deg_i, q) >= count_i) is a surprise
with the same population calibration.  DEV: count of edges with z~ below the q-quantile (dissimilar);
CAM: above the (1-q)-quantile (too similar).  Usage: edge_count_probe.py acm:256 blogcatalog:256 facebook ...
"""
import sys, numpy as np
from scipy.stats import binom, norm
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from ebgad.edgeleg import undirected_edges
from run_ebgad import eval_mask

A = lambda y, z: 100 * roc_auc_score(y, z)
print("%-14s | %5s | %-13s | %-13s | %-13s | %-13s | %-13s" % ("dataset", "deg", "mean e D/C", "count5 D/C", "count10 D/C", "count20 D/C", "top-k D/C"))
for arg in sys.argv[1:]:
    name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    x = preprocess_features(data.x, data.edge_index, pca=pca).double().numpy(); n, d = x.shape
    ia, ib = undirected_edges(data.edge_index.numpy(), n)
    deg = (np.bincount(ia, minlength=n) + np.bincount(ib, minlength=n)).astype(float); degs = np.maximum(deg, 1.0)
    xu = x / (np.sqrt((x * x).sum(1)) + 1e-12)[:, None]
    r = np.clip((xu[ia] * xu[ib]).sum(1), -0.999, 0.999); z = np.arctanh(r)
    med = np.median(z); mad = 1.4826 * np.median(np.abs(z - med)) + 1e-12; zs = (z - med) / mad
    acc = np.zeros(n); np.add.at(acc, ia, zs); np.add.at(acc, ib, zs); e = acc / np.sqrt(degs)
    cols = [name.ljust(14), "%5.1f" % deg.mean(), "%5.1f / %5.1f" % (A(y, -e[m]), A(y, e[m]))]
    for q in (0.05, 0.10, 0.20):
        lo, hi = np.quantile(zs, q), np.quantile(zs, 1 - q)
        cl = np.zeros(n); np.add.at(cl, ia, zs < lo); np.add.at(cl, ib, zs < lo)
        ch = np.zeros(n); np.add.at(ch, ia, zs > hi); np.add.at(ch, ib, zs > hi)
        s_dev = -binom.logsf(cl - 1, degs, q); s_cam = -binom.logsf(ch - 1, degs, q)     # P(Bin >= count)
        s_dev = np.nan_to_num(s_dev, posinf=700); s_cam = np.nan_to_num(s_cam, posinf=700)
        cols.append("%5.1f / %5.1f" % (A(y, s_dev[m]), A(y, s_cam[m])))
    # top-k scan: the most negative standardized partial sum over a node's sorted edges (k = 1..deg)
    order = np.argsort(zs); best_lo = np.zeros(n); best_hi = np.zeros(n)
    src = np.concatenate([ia, ib]); dst = np.concatenate([ib, ia]); zz = np.concatenate([zs, zs])
    o = np.lexsort((zz, src)); src_o, zz_o = src[o], zz[o]
    starts = np.searchsorted(src_o, np.arange(n)); ends = np.searchsorted(src_o, np.arange(n), side="right")
    for i in range(n):
        v = zz_o[starts[i]:ends[i]]
        if len(v) == 0:
            continue
        cs = np.cumsum(v); k = np.arange(1, len(v) + 1)
        best_lo[i] = -np.min(cs / np.sqrt(k)); best_hi[i] = np.max(np.cumsum(v[::-1]) / np.sqrt(k))
    cols.append("%5.1f / %5.1f" % (A(y, best_lo[m]), A(y, best_hi[m])))
    print(" | ".join(cols), flush=True)
