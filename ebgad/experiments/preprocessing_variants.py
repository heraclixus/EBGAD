"""Feature preprocessing under the iid-column model: z-score (v4), PCA-k (the NeurIPS choice), PCA-whitened all
components, and EB column scales (each PCA component keeps its own variance, profiled in closed form).
Reports DEV / CAM AUROC of the marginal residual scale s_i with the EB template. Dense-spectrum datasets.
Usage: ... preprocessing_variants.py weibo facebook ..."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.experiments.eb_smoothing import fit_eb
from run_ebgad import eval_mask

def scores(x, V, lam, d):
    xhat = V.T @ x; S = (xhat ** 2).sum(1)
    f = fit_eb(lam, S, d); s2, a, k2 = f["sigma2"], f["a"], f["kappa2"]
    v = s2 + 1.0 / (a * (k2 + lam)); t = (v - s2) / v
    delta = x - V @ (t[:, None] * xhat); tau2 = s2 * np.maximum(1.0 - (V * V) @ t, 1e-12)
    return (delta ** 2).sum(1) / (d * tau2), f

def col_scaled(x, V, lam):
    """EB column scales: x_f ~ N(0, c_f v(L)); profile c_f = mean_j xhat_jf^2 / v_j; iterate with the shared fit."""
    xhat = V.T @ x; d = x.shape[1]; c = np.ones(d)
    for _ in range(4):
        S = ((xhat / np.sqrt(c)[None, :]) ** 2).sum(1)
        f = fit_eb(lam, S, d); v = f["sigma2"] + 1.0 / (f["a"] * (f["kappa2"] + lam))
        c = ((xhat ** 2) / v[:, None]).mean(0)
    return x / np.sqrt(c)[None, :]

print("%-12s %4s | %-15s | %-15s | %-15s | %-15s" % ("dataset", "d", "z-score", "PCA-64", "PCA-whiten all", "EB col. scales"))
for name in sys.argv[1:]:
    data = load_data(name)
    y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    xt = preprocess_features(data.x, data.edge_index); n, d = xt.shape
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad", cache_tag=f"{name}_pcaNone_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64); x = xt.double().numpy()
    out = []
    # 1. z-score
    s, _ = scores(x, V, lam, d); out.append(s)
    # 2. PCA-64 (as the paper: PCA on raw then z-score)
    k = min(64, d)
    xc = x - x.mean(0); U, sv, Wt = np.linalg.svd(xc, full_matrices=False); p = xc @ Wt[:k].T; p = (p - p.mean(0)) / (p.std(0) + 1e-12)
    s, _ = scores(p, V, lam, k); out.append(s)
    # 3. PCA-whitened, all components with non-negligible variance
    keep = sv > 1e-6 * sv[0]; pw = (xc @ Wt[keep].T) / (sv[keep] / np.sqrt(n - 1))[None, :]
    s, _ = scores(pw, V, lam, pw.shape[1]); out.append(s)
    # 4. EB column scales on the z-scored features
    xs = col_scaled(x, V, lam); s, _ = scores(xs, V, lam, d); out.append(s)
    r = lambda z: "%5.1f / %5.1f" % (100 * roc_auc_score(y, z[m]), 100 * roc_auc_score(y, -z[m]))
    print("%-12s %4d | %s | %s | %s | %s" % (name, d, *[r(z) for z in out]), flush=True)
