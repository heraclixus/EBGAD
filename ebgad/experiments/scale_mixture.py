"""Can a two-component mixture with a FREE majority component find the anomalous
scale group label-free? 1-D on the calibrated scale statistic (standardized on
the evaluated population) and 2-D with the shape scan. Score = posterior of the
minority component. Label values are never used in fitting."""
import sys, os, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
order = ['elliptic', 'elliptic_plus_plus', 't_finance', 'dgraph', 'yelpchi', 'facebook', 'tolokers', 'weibo', 'reddit', 'amazon', 'blogcatalog', 'acm', 'questions']
def std(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return np.clip((v - med) / mad, -8, 8)
print('%-12s | %6s %6s | %8s %6s %7s %7s | %8s %6s | %8s' % ('dataset', '-z', '+z', 'GMM1D', 'pi', 'mu_min', 'mu_maj', 'GMM2D', 'pi', 'GMM3D-min'))
for ds in order:
    p = f'results/ebgad_v2_fraud/seed0/{ds}_scores.npz'
    if not os.path.exists(p): p = f'results/ebgad_v2_tg/{ds}_scores.npz'
    if not os.path.exists(p): continue
    z = np.load(p); y = z['y']; m = (y == 0) | (y == 1)
    mag = std(np.nan_to_num(z['MAG'])[m]); sh = std(np.nan_to_num(z['SCAN-R'])[m])
    auc = lambda s: 100 * roc_auc_score(y[m], s)
    out = []
    for X in (mag[:, None], np.column_stack([mag, sh])):
        g = GaussianMixture(2, covariance_type='full', n_init=5, random_state=0).fit(X)
        k = int(np.argmin(g.weights_)); post = g.predict_proba(X)[:, k]
        out.append((auc(post), g.weights_[k], g.means_[k], g.means_[1 - k]))
    g3 = GaussianMixture(3, covariance_type='full', n_init=5, random_state=0).fit(mag[:, None])
    k3 = int(np.argmin(g3.weights_)); post3 = g3.predict_proba(mag[:, None])[:, k3]
    a1, pi1, mu1, mu0 = out[0]; a2, pi2, mu2, _ = out[1]
    print('%-12s | %6.1f %6.1f | %8.1f %6.3f %7.2f %7.2f | %8.1f %6.3f | %8.1f' % (ds[:12], auc(-mag), auc(mag), a1, pi1, mu1[0], mu0[0], a2, pi2, auc(post3)))
