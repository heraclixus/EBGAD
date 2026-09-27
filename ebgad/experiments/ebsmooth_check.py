"""Validate the matrix-free EB smoother against the dense spectral fit on full-spectrum datasets.
Usage: PYTHONPATH=. python ebgad/experiments/ebsmooth_check.py yelpchi weibo reddit amazon"""
import sys, time
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.ebsmooth import EBSmoother, laplacian_csr
from ebgad.experiments.eb_smoothing import fit_eb, node_stat
from run_ebgad import eval_mask

for name in sys.argv[1:]:
    data = load_data(name)
    xt = preprocess_features(data.x, data.edge_index); n, d = xt.shape
    x = xt.double().numpy()
    y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    # dense reference
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad",
                            cache_tag=f"{name}_pcaNone_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
    xhat = V.T @ x; S = (xhat ** 2).sum(1)
    fd = fit_eb(lam, S, d)
    t = fd["g2"] / (fd["g2"] + fd["kappa2"] + lam)
    s_dense = node_stat(x, V, xhat, t, fd["sigma2"]); S_ii_dense = (V * V) @ t
    nll_dense_at = lambda s2, a, k2: float(np.sum(d * np.log(s2 + 1 / (a * (k2 + lam))) + S / (s2 + 1 / (a * (k2 + lam)))))
    # matrix-free
    t0 = time.time()
    L = laplacian_csr(data.edge_index, n)
    sm = EBSmoother(L, x, n_probes=20, m=80, seed=0)
    fm = sm.fit()
    out = sm.residual_and_scale(fm, n_hutch=128)
    el = time.time() - t0
    print(f"[{name}] n={n} d={d}  dense: sigma2={fd['sigma2']:.4f} a={fd['a']:.4g} kappa2={fd['kappa2']:.4g} g2={fd['g2']:.4g} nll={fd['nll']:.1f}")
    print(f"        mfree: sigma2={fm.sigma2:.4f} a={fm.a:.4g} kappa2={fm.kappa2:.4g} g2={fm.g2:.4g} nll(slq)={fm.nll:.1f} "
          f"dense-nll at mfree params={nll_dense_at(fm.sigma2, fm.a, fm.kappa2):.1f}  ({el:.0f}s, {fm.n_eval} nll evals)")
    print(f"        S_ii: dense median {np.median(S_ii_dense):.3f}  mfree median {np.median(out['S_ii']):.3f}  corr {np.corrcoef(S_ii_dense, out['S_ii'])[0,1]:.3f}")
    s_dense_at_mfree = node_stat(x, V, xhat, fm.g2 / (fm.g2 + fm.kappa2 + lam), fm.sigma2)
    print(f"        s: spearman(dense, mfree) {spearmanr(s_dense, out['s']).correlation:.4f}; at the same params {spearmanr(s_dense_at_mfree, out['s']).correlation:.4f}")
    print(f"        AUROC DEV dense {100*roc_auc_score(y, s_dense[m]):.1f} mfree {100*roc_auc_score(y, out['s'][m]):.1f} | CAM dense {100*roc_auc_score(y, -s_dense[m]):.1f} mfree {100*roc_auc_score(y, -out['s'][m]):.1f}")
