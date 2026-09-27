"""Structural leg: the same EB smoothing model applied to structural features X_s = A_n R, a Gaussian random
sketch (d_s columns) of the node's normalized-adjacency row (row geometry preserved, columns ~ i.i.d. by
construction, O(|E| d_s), no rank choice). Reports both tails of the structural residual scale, the no-graph
structural statistic (sketch-row norm ~ degree), and the union (max of empirical upper-tail surprises on the
evaluated population) of the feature leg (v4 s_i, declared tail) with the structural leg in each tail.
Usage: ... structural_leg_test.py weibo acm ... [--ds 64]"""
import sys, numpy as np, scipy.sparse as sp, torch
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import permute_data
from ebgad.ebsmooth import EBSmoother, laplacian_csr, rcm_order
from run_ebgad import eval_mask
from ebgad.hier import fit_scale_prior
from scipy.stats import norm
ds_cols = int(sys.argv[sys.argv.index("--ds") + 1]) if "--ds" in sys.argv else 64
names = [a for a in sys.argv[1:] if not a.startswith("--") and not a.isdigit()]
FRAUD = {"elliptic", "elliptic_plus_plus", "t_finance", "dgraph", "yelpchi", "tolokers"}

def emp_up(v, mask):
    ref = np.sort(v[mask]); F = (np.searchsorted(ref, v, side="right") + 0.5) / (ref.size + 1.0)
    return -np.log(np.clip(1.0 - F, 1.0 / (ref.size + 1.0), 1.0))

print("%-12s | %-8s | %-13s | %-9s | %-13s | %-13s | %s" % ("dataset", "feature", "struct DEV/CAM", "struct 2S", "no-graph s.", "union DEV/CAM", "fit (sigma2, gain typ)"))
for name in names:
    data = load_data(name); y_all = data.y.numpy().astype(int); n = data.num_nodes
    perm = rcm_order(laplacian_csr(data.edge_index, n)); data = permute_data(data, perm); y_all = y_all[perm]
    m = eval_mask(y_all); y = y_all[m]
    ei = data.edge_index.numpy(); A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64); A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel(); dinv = 1.0 / np.sqrt(np.maximum(deg, 1)); An = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    rng = np.random.default_rng(0); R = rng.standard_normal((n, ds_cols)) / np.sqrt(ds_cols)
    Xs = np.asarray(An @ R); Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    L = laplacian_csr(data.edge_index, n)
    sm = EBSmoother(L, Xs, n_probes=20, m=80, seed=0, lam_max=2.0); fit = sm.fit(n_starts=3)
    res = sm.residual_and_scale(fit, n_hutch=128, seed=1, shape=False); s_struct = res["s"]
    z = np.load(f"results/ebgad_v4_final/seed0/{name}_scores.npz"); s_feat = np.empty(n); s_feat[:] = z["s"][perm]
    tail = "CAM" if name in FRAUD else "DEV"
    feat_sur = emp_up(-np.log(np.maximum(s_feat, 1e-12)) if tail == "CAM" else np.log(np.maximum(s_feat, 1e-12)), m)
    au = lambda v: 100 * roc_auc_score(y, v[m])
    ls = np.log(np.maximum(s_struct, 1e-12)); ng = (Xs * Xs).sum(1)
    union_dev = au(np.maximum(feat_sur, emp_up(ls, m))); union_cam = au(np.maximum(feat_sur, emp_up(-ls, m)))
    mu_s, sig2, m_d, v_d = fit_scale_prior(s_struct[m], Xs.shape[1]); zz = (ls - mu_s - m_d) / np.sqrt(sig2 + v_d)
    two_sided = -np.minimum(norm.logcdf(zz), norm.logsf(zz))
    g1 = fit.g2 / (fit.g2 + fit.kappa2 + 1.0)
    print("%-12s | %5.1f %s | %5.1f / %5.1f | %9.1f | %5.1f / %5.1f | %5.1f / %5.1f | %.2f, %.2f" % (name, au(s_feat) if tail == "DEV" else au(-s_feat), tail, au(s_struct), au(-s_struct), au(two_sided), au(ng), au(-ng), union_dev, union_cam, fit.sigma2, g1), flush=True)
