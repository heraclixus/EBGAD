import sys, numpy as np
sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import prepare
from ebgad.fit import fit_prepared
data = load_data('weibo'); y = data.y.numpy()
auc = lambda s: 100*roc_auc_score(y, s)
cache = {}
pu = prepare('weibo', data, 1.0, template='low_unit', pca=64, cache_dir='cache/ebgad', _spectrum_cache=cache)
pp = prepare('weibo', data, 0.5, template='low', pca=64, cache_dir='cache/ebgad', _spectrum_cache=cache)
V, lam, x = pu.V, pu.lam, pu.x
xh = V.T @ x                       # (k, d) spectral coefficients of features
print('feature-only  ||x_i||^2            AUROC %.2f' % auc((x*x).sum(1)))
print('paper J* (gamma .5, rho 1, kappa 1)  AUROC %.2f' % auc(((V @ (np.sqrt(1+lam)[:,None] * (V.T @ pp.delta)))**2).sum(1)))
print('unit residual ||delta_i||^2 (g=1)   AUROC %.2f' % auc((pu.delta**2).sum(1)))
for g in [0.05, 0.2, 0.5, 1.0, 2.0]:
    s = g**2/(g**2+lam); sm = V @ (s[:,None]*xh)
    print('smoothed-feature energy gamma=%-4g   AUROC %.2f' % (g, auc((sm*sm).sum(1))))
edges = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.01]
print('\nband energies of x (features) and of the unit residual, by Laplacian eigenvalue band')
print('%-14s %6s %8s %8s %10s' % ('band', 'modes', 'x-band', 'res-band', 'x-band/||x||'))
for a, b in zip(edges[:-1], edges[1:]):
    w = ((lam >= a) & (lam < b)).astype(float)
    if w.sum() == 0: continue
    ex = ((V @ (w[:,None]*xh))**2).sum(1)
    er = ((V @ (w[:,None]*(V.T @ pu.delta)))**2).sum(1)
    print('[%.2f,%.2f)   %6d %8.2f %8.2f %10.2f' % (a, b, int(w.sum()), auc(ex), auc(er), auc(ex/np.maximum((x*x).sum(1),1e-12))))
# per-mode calibration of the two fits: q_j S_j / d  (1 = calibrated)
fu = fit_prepared(pu); fp = fit_prepared(pp)
for name, p, f in [('unit g=1', pu, fu), ('paper g=.5', pp, fp)]:
    r = f.q * p.S / p.d
    qs = np.percentile(lam, [0, 10, 25, 50, 75, 90, 100])
    print('\n%s fit rho=%.3f kappa=%g: per-mode calibration q_j S_j/d by eigenvalue decile' % (name, f.rho, f.kappa))
    for a, b in zip(qs[:-1], qs[1:]):
        m = (lam >= a) & (lam <= b)
        print('  lam in [%.3f, %.3f]: median ratio %.3f' % (a, b, np.median(r[m])))
