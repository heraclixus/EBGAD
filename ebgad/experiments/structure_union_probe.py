"""Structural channel and its union with the feature channel.
Structure: low-rank signal plus noise on the normalized adjacency, A_n ~ U_k Lambda_k U_k^T; per-node residual
r_i = ||(A_n - U L U^T)_i||^2 (k eigenpairs by LOBPCG-free eigsh with generous tolerance). Feature channel: the
v4 local leg s_i (from results/ebgad_v4_final/seed0 scores). Both are turned into upper-tail surprises on the
evaluated population (empirical calibration, rank-based) and combined by the maximum ("either type").
Reports DEV AUROC: feature leg, structure leg (k = 8 / 32), union (k = 8 / 32), and the population-calibrated
structural z. Usage: ... structure_union_probe.py weibo acm ..."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from run_ebgad import eval_mask

def emp_surprise(v, mask):
    ref = np.sort(v[mask]); F = (np.searchsorted(ref, v, side="right") + 0.5) / (ref.size + 1.0)
    return -np.log(np.clip(1.0 - F, 1.0 / (ref.size + 1.0), 1.0))       # upper-tail empirical -log p

print("%-12s | %-8s | %-13s | %-13s | %s" % ("dataset", "feature", "struct k=8/32", "union k=8/32", "degree(DEV/CAM)"))
for name in sys.argv[1:]:
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]; n = data.num_nodes
    ei = data.edge_index.numpy(); A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64); A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel(); dinv = 1.0 / np.sqrt(np.maximum(deg, 1)); An = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    z = np.load(f"results/ebgad_v4_final/seed0/{name}_scores.npz"); s = z["s"]
    au = lambda v: 100 * roc_auc_score(y, v[m])
    feat = emp_surprise(np.log(np.maximum(s, 1e-12)), m)
    row_sq = np.asarray(An.multiply(An).sum(1)).ravel()
    st, un = [], []
    for k in (8, 32):
        k_eff = min(k, n - 2)
        try:
            vals, vecs = eigsh(An, k=k_eff, which="LA", tol=1e-4, ncv=min(n - 1, 4 * k_eff + 20), maxiter=5000)
        except Exception as e:
            vals, vecs = e.eigenvalues, e.eigenvectors
        UL = vecs * vals[None, :]; proj = An @ vecs
        err = row_sq - 2 * (proj * UL).sum(1) + ((UL @ (vecs.T @ vecs)) * UL).sum(1)
        err = np.maximum(err, 0) / np.maximum(row_sq, 1e-12)                  # relative residual: share of the row not explained
        st.append(au(err)); un.append(au(np.maximum(feat, emp_surprise(err, m))))
    print("%-12s | %8.1f | %6.1f / %5.1f | %6.1f / %5.1f | %5.1f/%5.1f" % (name, au(s), st[0], st[1], un[0], un[1], au(np.log1p(deg)), 100 * roc_auc_score(y, -np.log1p(deg)[m])), flush=True)
