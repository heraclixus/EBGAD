"""Two-groups EB with a LEARNED alternative on the calibrated magnitude statistic.

Null: MAG ~ N(0,1) (fixed by the hierarchical fit). Alternative: N(mu1, s1^2)
with mu1, s1, pi free, fitted by EM. The SIGN of mu1 is learned, so a
cluster of anomalies with unusually SMALL residual scale (camouflaged,
connected fraud) is found without labels. Also the 2-D version on
(MAG, standardized shape scan). Reports AUROC of the posterior and of the
raw statistic in both directions.
"""
import sys, os, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
res = sys.argv[1] if len(sys.argv) > 1 else 'results/ebgad_v2_tg'
def em(Z, iters=300):
    n, d = Z.shape
    r = (np.linalg.norm(Z, axis=1) > np.percentile(np.linalg.norm(Z, axis=1), 90)).astype(float)
    pi = r.mean(); mu1 = (r[:, None] * Z).sum(0) / r.sum(); s1 = 1.0
    for _ in range(iters):
        l0 = norm.logpdf(Z, 0, 1).sum(1)
        l1 = norm.logpdf(Z, mu1, s1).sum(1)
        la = np.log(pi) + l1; ln = np.log(1 - pi) + l0
        r = np.exp(la - np.logaddexp(la, ln))
        pi_new = float(np.clip(r.mean(), 1e-4, 0.5))
        mu_new = (r[:, None] * Z).sum(0) / max(r.sum(), 1e-9)
        s_new = float(np.sqrt(((r[:, None] * (Z - mu_new) ** 2).sum() / max(r.sum() * d, 1e-9))))
        s_new = float(np.clip(s_new, 0.05, 10))
        if abs(pi_new - pi) < 1e-7 and np.abs(mu_new - mu1).max() < 1e-7: break
        pi, mu1, s1 = pi_new, mu_new, s_new
    return r, pi, mu1, s1, (la - ln)
order = ['elliptic', 'elliptic_plus_plus', 't_finance', 'dgraph', 'yelpchi', 'facebook', 'tolokers', 'weibo', 'reddit', 'amazon', 'blogcatalog', 'acm', 'questions']
print('%-12s | %6s %6s | %8s %6s %6s | %8s %6s | %s' % ('dataset', 'MAG', '-MAG', 'EM1D', 'pi', 'mu1', 'EM2D', 'pi', 'mu1 (mag, shape)'))
for ds in order:
    p = f'{res}/{ds}_scores.npz'
    if not os.path.exists(p): continue
    z = np.load(p); y = z['y']; m = (y == 0) | (y == 1)
    auc = lambda s: 100 * roc_auc_score(y[m], np.nan_to_num(s)[m])
    mag = np.nan_to_num(z['MAG']); mag = np.clip(mag, -8, 8)
    r1, pi1, mu1, s1, llr1 = em(mag[:, None])
    sh = np.nan_to_num(z['SCAN-R']); sh = (sh - np.median(sh)) / (1.4826 * np.median(np.abs(sh - np.median(sh))) + 1e-9); sh = np.clip(sh, -8, 8)
    r2, pi2, mu2, s2, llr2 = em(np.column_stack([mag, sh]))
    print('%-12s | %6.1f %6.1f | %8.1f %6.3f %6.2f | %8.1f %6.3f %s' % (ds[:12], auc(mag), auc(-mag), auc(llr1), pi1, mu1[0], auc(llr2), pi2, np.round(mu2, 2)))
