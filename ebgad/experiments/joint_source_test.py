"""One model, two data sources: X_all = [X | A_n R] (attributes and a d_s-column random sketch of the
normalized-adjacency row), EB smoothing fit with column scales (each column its own EB scale, closed form),
one residual scale s_i with its exact null, the declared tail per family. Compared with attributes only.
Usage: ... joint_source_test.py weibo acm ... [--ds 64]"""
import sys, numpy as np, scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features, permute_data
from ebgad.ebsmooth import EBSmoother, laplacian_csr, rcm_order
from run_ebgad import eval_mask
ds_cols = int(sys.argv[sys.argv.index("--ds") + 1]) if "--ds" in sys.argv else 64
ds_equal = "--ds-equal" in sys.argv                # one structural column per attribute column
names = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
FRAUD = {"elliptic", "elliptic_plus_plus", "t_finance", "dgraph", "yelpchi", "tolokers"}
print("%-12s | tail | %-13s | %-13s | %-13s | %-13s | joint fit" % ("dataset", "attributes", "structure", "joint (col.sc.)", "joint (equal)"))
for name in names:
    data = load_data(name); y_all = data.y.numpy().astype(int); n = data.num_nodes
    perm = rcm_order(laplacian_csr(data.edge_index, n)); data = permute_data(data, perm); y_all = y_all[perm]
    m = eval_mask(y_all); y = y_all[m]; tail = "CAM" if name in FRAUD else "DEV"
    x = preprocess_features(data.x, data.edge_index, pca=(256 if name in ("acm", "blogcatalog") else None)).double().numpy()
    ei = data.edge_index.numpy(); A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64); A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel(); dinv = 1.0 / np.sqrt(np.maximum(deg, 1)); An = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    dsc = x.shape[1] if ds_equal else ds_cols
    R = np.random.default_rng(0).standard_normal((n, dsc)) / np.sqrt(dsc); Xs = np.asarray(An @ R); Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    L = laplacian_csr(data.edge_index, n)
    def run(X, col_scales):
        sm = EBSmoother(L, X, n_probes=20, m=80, seed=0, lam_max=2.0)
        fit = sm.fit_column_scales(n_starts=3) if col_scales else sm.fit(n_starts=3)
        return sm.residual_and_scale(fit, n_hutch=128, seed=1, shape=False)["s"], fit
    sign = -1.0 if tail == "CAM" else 1.0
    R2 = lambda s: "%5.1f / %4.1f" % (100 * roc_auc_score(y, sign * s[m]), 100 * average_precision_score(y, sign * s[m]))
    s_a, _ = run(x, False); s_s, _ = run(Xs, False); s_j, fj = run(np.hstack([x, Xs]), True); s_je, _ = run(np.hstack([x, Xs]), False)
    print("%-12s | %s  | %s | %s | %s | %s | s2=%.2f gain=%.2f" % (name, tail, R2(s_a), R2(s_s), R2(s_j), R2(s_je), fj.sigma2, fj.g2 / (fj.g2 + fj.kappa2 + 1)), flush=True)
