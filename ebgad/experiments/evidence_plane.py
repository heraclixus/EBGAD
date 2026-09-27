"""Selection-free 2-D score: low density in the calibrated (magnitude, shape) plane.

Coordinates per node: z_mag (hierarchical scale z) and z_shape (max over
bulk-standardized ratio members, HSCAN-R). Score = -log of a smoothed 2-D
histogram density of all nodes at the node's position. No channel choice,
no direction assumed: anomalies are wherever the normal population is not.
Also reports the 1-D variants and Mahalanobis distance in the plane.
"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
res_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ebgad_v2_hier'
order = ['weibo', 'reddit', 'amazon', 'yelpchi', 'blogcatalog', 'facebook', 'acm', 'elliptic', 'elliptic_plus_plus', 't_finance', 'tolokers', 'questions']
def lowdens(X, bins=80, sigma=2.0):
    X = np.nan_to_num(X); lo = np.percentile(X, 0.2, axis=0); hi = np.percentile(X, 99.8, axis=0)
    Xc = np.clip(X, lo, hi)
    H, edges = np.histogramdd(Xc, bins=bins, range=[(l, h) for l, h in zip(lo, hi)])
    H = gaussian_filter(H, sigma) + 1e-12
    idx = [np.clip(np.searchsorted(e, Xc[:, i], side='right') - 1, 0, bins - 1) for i, e in enumerate(edges)]
    return -np.log(H[tuple(idx)])
print('%-12s %6s | %6s %6s | %8s %8s %8s | %8s' % ('dataset', 'oracle', 'MAG', 'HSC-R', 'dens2D', 'mahal2D', 'dens3D', '|z_mag|'))
for name in order:
    try:
        z = np.load(f'{res_dir}/{name}_scores.npz')
    except FileNotFoundError:
        continue
    y = z['y']; m = (y == 0) | (y == 1)
    auc = lambda s: 100 * roc_auc_score(y[m], np.nan_to_num(s)[m])
    members = [k for k in z.files if k != 'y' and not k.startswith(('SCAN', 'MAG', 'HSCAN', 'H-'))]
    oracle = max(auc(z[k]) for k in members)
    zm, zr, ze = z['MAG'], z['HSCAN-R'], z['HSCAN-E']
    X2 = np.column_stack([zm, zr]); X3 = np.column_stack([zm, zr, ze])
    mu = np.median(X2, axis=0); C = np.cov(X2[np.all(np.abs(X2 - mu) < 3, axis=1)].T) + 1e-6 * np.eye(2)
    mah = np.einsum('ij,jk,ik->i', X2 - mu, np.linalg.inv(C), X2 - mu)
    print('%-12s %6.1f | %6.1f %6.1f | %8.1f %8.1f %8.1f | %8.1f' % (name[:12], oracle, auc(zm), auc(zr), auc(lowdens(X2)), auc(mah), auc(lowdens(X3, bins=40)), auc(np.abs(zm))))
