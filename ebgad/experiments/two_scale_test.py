"""Two legs of the relaxation under the EB model: local surprise s_i (observation vs template, exact chi^2_d/d
null) and regional surprise g_i = ||m_i||^2 / (d Var(m_i)) (template vs population; m = S x is Gaussian with
Cov(m) = v(L) - 2 sigma^2 I + sigma^4 v(L)^{-1}, so g_i is chi^2_d/d too). Combination: the larger of the two
surprises after each is standardized by its null (max of chi^2 upper-tail -log p), read in the declared tail.
Usage: ... two_scale_test.py weibo amazon ..."""
import sys, numpy as np
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.experiments.eb_smoothing import fit_eb
from run_ebgad import eval_mask

def au(y, z): return 100 * roc_auc_score(y, z), 100 * average_precision_score(y, z)
print("%-12s | %-13s | %-13s | %-13s | %-13s | %-13s | %-13s" % ("dataset", "s DEV/CAM", "g DEV/CAM", "max(s,g) DEV", "no-graph DEV", "s+g Fisher DEV", "joint q DEV/CAM"))
for arg in sys.argv[1:]:
    name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
    data = load_data(name); y_all = data.y.numpy().astype(int); m_ = eval_mask(y_all); y = y_all[m_]
    xt = preprocess_features(data.x, data.edge_index, pca=pca); n, d = xt.shape; x = xt.double().numpy()
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad", cache_tag=f"{name}_pca{pca}_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
    xhat = V.T @ x; f = fit_eb(lam, (xhat ** 2).sum(1), d); s2, a, k2 = f["sigma2"], f["a"], f["kappa2"]
    v = s2 + 1.0 / (a * (k2 + lam)); t = (v - s2) / v; V2 = V * V
    m = V @ (t[:, None] * xhat); delta = x - m
    tau2 = s2 * np.maximum(1.0 - V2 @ t, 1e-12); s = (delta ** 2).sum(1) / (d * tau2)
    var_m = np.maximum(V2 @ (v * t * t), 1e-12)                     # Cov(m)_ii = sum_j t_j^2 v_j V_ij^2
    g = (m ** 2).sum(1) / (d * var_m)
    lp_s = np.nan_to_num(-chi2.logsf(d * s, d), posinf=1e6); lp_g = np.nan_to_num(-chi2.logsf(d * g, d), posinf=1e6)   # upper-tail surprises
    # joint two-leg surprise: per node, (m_if, delta_if) is bivariate Gaussian over f with a common 2x2 covariance
    c_mm = V2 @ (v * t * t); c_dd = V2 @ (v * (1 - t) ** 2); c_md = V2 @ (v * t * (1 - t))
    det = np.maximum(c_mm * c_dd - c_md ** 2, 1e-18)
    q = ((m * m).sum(1) * c_dd - 2 * (m * delta).sum(1) * c_md + (delta * delta).sum(1) * c_mm) / det / (2 * d)   # chi^2_{2d} / 2d
    comb = np.maximum(lp_s, lp_g); fisher = lp_s + lp_g
    xn = (x * x).sum(1)
    R = lambda z: "%5.1f / %5.1f" % (au(y, z[m_])[0], au(y, -z[m_])[0])
    print("%-12s | %s | %s | %5.1f         | %5.1f         | %5.1f         | %s" % (arg, R(s), R(g), au(y, comb[m_])[0], au(y, xn[m_])[0], au(y, fisher[m_])[0], R(q)), flush=True)
