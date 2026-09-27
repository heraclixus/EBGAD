"""Robust EB template: x_i = h_i + eps_i, eps_i ~ N(0, sigma^2 zeta_i I_d), zeta_i ~ InvGamma(nu/2, nu/2)
(row-wise Student-t noise). Given node weights w_i = E[1/zeta_i | x], the posterior mean of the field is the
reweighted smoother m = (cK + W)^{-1} W x (c = sigma^2 a, W = diag(w)); the E-step gives
w_i = (nu + d) / (nu + r_i) with r_i = (||x_i - m_i||^2 + d Var_i) / sigma^2, Var_i = sigma^2 [(cK + W)^{-1}]_ii.
(a, kappa^2) come from the Gaussian EB fit; sigma^2 is re-estimated (generalized M-step); nu by EB from the
F(d, nu) marginal of r_i / d. Score: r_i / d (deviation = upper tail, camouflage = lower).
Usage: ... robust_template_test.py weibo amazon ... [--no-sigma-update]"""
import sys, numpy as np, scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.stats import f as f_dist
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.experiments.eb_smoothing import fit_eb
from ebgad.ebsmooth import laplacian_csr, block_cg
from run_ebgad import eval_mask

update_sigma = "--no-sigma-update" not in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith("--")]
print("%-12s | %-13s | %-13s | %-13s | %-13s | nu_EB  sigma2 gauss->robust  weights<0.5" % ("dataset", "gauss DEV/CAM", "t nu=4", "t nu=EB", "t nu=EB, no s2 upd"))

def robust(x, K, c, s2, nu, n_iter=10, n_hutch=64, update_s2=True, seed=0):
    n, d = x.shape; w = np.ones(n); rng = np.random.default_rng(seed)
    Z = rng.choice([-1.0, 1.0], size=(n, n_hutch))
    for it in range(n_iter):
        mv = lambda Y: c * (K @ Y) + w[:, None] * Y
        m, _ = block_cg(mv, w[:, None] * x, tol=1e-6)
        Ginv_Z, _ = block_cg(mv, Z, tol=1e-5)
        var = s2 * np.maximum((Z * Ginv_Z).mean(axis=1), 1e-12)     # sigma^2 [(cK+W)^{-1}]_ii
        delta = x - m
        r = ((delta ** 2).sum(1) + d * var) / s2
        if isinstance(nu, str):                                       # EB for nu from the F(d, nu) marginal of r/d
            z = np.clip(r / d, 1e-9, None)
            nu_hat = np.exp(minimize_scalar(lambda ln: -f_dist.logpdf(z, d, np.exp(ln)).sum(), bounds=(np.log(0.5), np.log(500)), method="bounded").x)
        else:
            nu_hat = nu
        w = (nu_hat + d) / (nu_hat + r)
        if update_s2:
            s2 = float((w * ((delta ** 2).sum(1) + d * var)).sum() / (n * d))
    return r / d, w, nu_hat, s2

for name in names:
    pca = None
    if ":" in name: name, pca = name.split(":"); pca = int(pca)
    data = load_data(name); y_all = data.y.numpy().astype(int); msk = eval_mask(y_all); y = y_all[msk]
    xt = preprocess_features(data.x, data.edge_index, pca=pca); n, d = xt.shape; x = xt.double().numpy()
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad", cache_tag=f"{name}_pca{pca}_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
    xhat = V.T @ x; f = fit_eb(lam, ((xhat ** 2).sum(1)), d); s2, a, k2 = f["sigma2"], f["a"], f["kappa2"]
    v = s2 + 1.0 / (a * (k2 + lam)); t = (v - s2) / v
    delta = x - V @ (t[:, None] * xhat); tau2 = s2 * np.maximum(1.0 - (V * V) @ t, 1e-12); s = (delta ** 2).sum(1) / (d * tau2)
    L = laplacian_csr(data.edge_index, n); K = (k2 * sp.eye(n) + L).tocsr(); c = s2 * a
    s2_use = max(s2, 1e-6)
    R = lambda z: "%5.1f / %5.1f" % (100 * roc_auc_score(y, z[msk]), 100 * roc_auc_score(y, -z[msk]))
    r4, w4, _, _ = robust(x, K, c, s2_use, 4.0, update_s2=update_sigma)
    rE, wE, nuE, s2E = robust(x, K, c, s2_use, "eb", update_s2=update_sigma)
    rN, wN, nuN, _ = robust(x, K, c, s2_use, "eb", update_s2=False)
    print("%-12s | %s | %s | %s | %s | %5.2g  %.3f->%.3f  %.3f" % (name + (f":{pca}" if pca else ""), R(s), R(r4), R(rE), R(rN), nuE, s2, s2E, (wE < 0.5).mean()), flush=True)
