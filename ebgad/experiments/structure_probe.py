"""Do the deviation benchmarks carry structural anomalies? Node statistics from the graph alone:
log degree, degree z-score, local clustering, neighbor-degree ratio, and the low-rank adjacency reconstruction
error (A ~ U_k Lambda_k U_k^T, k eigenpairs of the normalized adjacency; AnomalyDAE/ANOMALOUS-style structure
residual). DEV AUROC (upper tail) / CAM. Usage: ... structure_probe.py weibo acm ..."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from run_ebgad import eval_mask
print("%-12s | %-11s | %-11s | %-11s | %-11s | %s" % ("dataset", "log degree", "clustering", "nbr-deg ratio", "2-hop count", "adjacency recon. error k=8 / 32 / 128 (DEV/CAM)"))
for name in sys.argv[1:]:
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]; n = data.num_nodes
    ei = data.edge_index.numpy(); A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64); A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel(); R = lambda z: "%5.1f/%5.1f" % (100 * roc_auc_score(y, z[m]), 100 * roc_auc_score(y, -z[m]))
    logdeg = np.log1p(deg)
    A2 = A @ A; tri = np.asarray(A2.multiply(A).sum(1)).ravel() / 2.0; clust = np.where(deg > 1, 2 * tri / np.maximum(deg * (deg - 1), 1), 0.0)
    nbr_deg = np.asarray(A @ deg[:, None]).ravel() / np.maximum(deg, 1); ratio = np.log1p(deg) - np.log1p(nbr_deg)
    two_hop = np.asarray((A2 > 0).sum(1)).ravel().astype(float)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1)); An = sp.diags(dinv) @ A @ sp.diags(dinv)
    outs = []
    for k in (8, 32, 128):
        if k >= n - 1: outs.append("  -  "); continue
        vals, vecs = eigsh(An, k=k, which="LA")
        low = (vecs * vals[None, :]) @ vecs.T if n <= 20000 else None
        if low is not None:
            err = np.asarray(((An - low) if False else (An.toarray() - low))).__pow__(2).sum(1)
        else:
            # row-wise residual without materializing: ||An_i||^2 - 2 An_i (U L U^T)_i + ||(U L U^T)_i||^2
            UL = vecs * vals[None, :]; proj = (An @ vecs)                     # (n, k) = An U
            err = np.asarray(An.multiply(An).sum(1)).ravel() - 2 * (proj * UL).sum(1) + ((UL @ (vecs.T @ vecs)) * UL).sum(1)
        outs.append(R(err))
    print("%-12s | %s | %s | %s | %s | %s" % (name, R(logdeg), R(clust), R(ratio), R(two_hop), " / ".join(outs)), flush=True)
