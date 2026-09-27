"""Node share of the evidence: e_i = ||(v_theta(L)^{-1/2} X)_i||^2 / d, the whitened energy under the EB
signal-plus-noise model (exact chi^2_d/d null, no leverage correction), against the marginal residual
scale s_i of the v4 runner. Dense-spectrum datasets only. Usage: ... whitened_energy_test.py ds[:pca] ..."""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.experiments.eb_smoothing import fit_eb
from run_ebgad import eval_mask

def au(y, s): return 100 * roc_auc_score(y, s), 100 * average_precision_score(y, s)
print("%-14s %5s | %-17s | %-17s | %-17s | %-17s | fit" % ("dataset", "pca", "s: DEV / CAM", "e: DEV / CAM", "e_hi: DEV / CAM", "e_lo: DEV / CAM"))
for arg in sys.argv[1:]:
    name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
    data = load_data(name); xt = preprocess_features(data.x, data.edge_index, pca=pca); n, d = xt.shape
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad",
                            cache_tag=f"{name}_pca{pca}_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64); x = xt.double().numpy()
    xhat = V.T @ x; S = (xhat ** 2).sum(1)
    f = fit_eb(lam, S, d); s2, a, k2 = f["sigma2"], f["a"], f["kappa2"]
    v = s2 + 1.0 / (a * (k2 + lam)); t = (v - s2) / v
    y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    # marginal residual scale (v4)
    delta = x - V @ (t[:, None] * xhat); tau2 = s2 * np.maximum(1.0 - (V * V) @ t, 1e-12)
    s = (delta ** 2).sum(1) / (d * tau2)
    # whitened energy: node share of the evidence
    W = V @ (xhat / np.sqrt(v)[:, None]); e = (W ** 2).sum(1) / d
    # its split into the smooth-signal part and the noise part of the spectrum (for diagnosis)
    hi = lam > np.median(lam); W_hi = V[:, hi] @ (xhat[hi] / np.sqrt(v[hi])[:, None]); e_hi = (W_hi ** 2).sum(1) / hi.sum()
    W_lo = V[:, ~hi] @ (xhat[~hi] / np.sqrt(v[~hi])[:, None]); e_lo = (W_lo ** 2).sum(1) / (~hi).sum()
    r = lambda z: "%5.1f / %5.1f" % (au(y, z[m])[0], au(y, -z[m])[0])
    print("%-14s %5s | %s | %s | %s | %s | s2=%.2f a=%.2g k2=%.2g" % (name, pca or "-", r(s), r(e), r(e_hi), r(e_lo), s2, a, k2), flush=True)
