"""Marginal-likelihood profile in the noise variance sigma^2 (a, kappa^2 re-optimized at each value),
with the two tails' AUROC along the profile. Shows whether a boundary fit (sigma^2 -> 0) is a true
optimum or a flat direction. Usage: PYTHONPATH=. python ebgad/experiments/ebsmooth_profile.py <dataset> [sym|comb]"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from ebgad.ebsmooth import EBSmoother, laplacian_csr
from run_ebgad import eval_mask
name = sys.argv[1]; variant = sys.argv[2] if len(sys.argv) > 2 else "sym"
data = load_data(name); x = preprocess_features(data.x, data.edge_index).double().numpy(); n, d = x.shape
y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
sm = EBSmoother(laplacian_csr(data.edge_index, n, variant=variant), x, n_probes=20, m=80, seed=0)
free = sm.fit()
print(f"[{name}/{variant}] free fit: sigma2={free.sigma2:.4g} a={free.a:.4g} kappa2={free.kappa2:.4g} nll={free.nll:.1f}")
print("  sigma2     nll - nll_free      a       kappa2     g2      gain(typ)   CAM    DEV")
lam_typ = 1.0 if variant == "sym" else float(np.median(sm.L.diagonal()))
for s2 in [1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]:
    f = sm.fit(fixed_sigma2=s2)
    r = sm.residual_and_scale(f, n_hutch=64)
    s = r["s"]
    print(f"  {s2:7.4f}  {f.nll - free.nll:14.1f}  {f.a:8.3g}  {f.kappa2:8.3g}  {f.g2:8.3g}  {f.g2 / (f.g2 + f.kappa2 + lam_typ):8.3f}  {100*roc_auc_score(y, -s[m]):5.1f}  {100*roc_auc_score(y, s[m]):5.1f}")
