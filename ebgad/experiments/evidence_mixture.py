"""Feasibility: two-groups EB with a learned Gaussian alternative in the calibrated evidence space.

Evidence per node: bulk-standardized log statistics of the scale (MAG) and
the shape profile members (R, DR@t). Standardization and EM on the evaluated
population (labeled nodes only; label VALUES never used). Null: N(0, C) with
C the robust correlation of the evidence; alternative: N(mu1, C) (mean shift)
with pi <= 1/2. EM from many initializations (each member, each sign, top or
bottom 5 percent); the fit with the highest mixture log-likelihood wins.
Score = log-likelihood ratio. This is the EB counterpart of prototype-anchor
scoring: the anomalous group is a second mode found by likelihood, its
direction is learned, not declared.
"""
import sys, os, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal as mvn
res = sys.argv[1] if len(sys.argv) > 1 else 'results/ebgad_v2_fraud/seed0'
alt = sys.argv[2] if len(sys.argv) > 2 else 'results/ebgad_v2_tg'
order = ['elliptic', 'elliptic_plus_plus', 't_finance', 'dgraph', 'yelpchi', 'facebook', 'tolokers', 'weibo', 'reddit', 'amazon', 'blogcatalog', 'acm', 'questions']
def robust_std(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return np.clip((v - med) / mad, -8, 8)
def em_fit(E, r0, C, iters=200):
    n, d = E.shape
    pi = float(np.clip(r0.mean(), 1e-3, 0.5)); mu1 = (r0[:, None] * E).sum(0) / max(r0.sum(), 1e-9)
    l0 = mvn.logpdf(E, np.zeros(d), C)
    prev = -np.inf
    for _ in range(iters):
        l1 = mvn.logpdf(E, mu1, C)
        la = np.log(pi) + l1; ln = np.log(1 - pi) + l0
        lm = np.logaddexp(la, ln); r = np.exp(la - lm)
        pi = float(np.clip(r.mean(), 1e-3, 0.5)); mu1 = (r[:, None] * E).sum(0) / max(r.sum(), 1e-9)
        ll = float(lm.sum())
        if ll - prev < 1e-6 * max(1, abs(prev)): break
        prev = ll
    return ll, pi, mu1, la - ln
print('%-12s | %6s %6s %6s | %8s %6s | %s' % ('dataset', 'oracle', 'scan-R', '-MAG', 'EM-LLR', 'pi', 'mu1 direction (top members)'))
for ds in order:
    p = f'{res}/{ds}_scores.npz'
    if not os.path.exists(p): p = f'{alt}/{ds}_scores.npz'
    if not os.path.exists(p): continue
    z = np.load(p); y = z['y']; m = (y == 0) | (y == 1)
    names = ['MAG'] + [k for k in z.files if k == 'R' or k.startswith('DR@')]
    cols, keep = [], []
    for k in names:
        v = np.nan_to_num(z[k][m]).astype(float)
        if k != 'MAG': v = np.log(np.maximum(v, 1e-300))
        if np.subtract(*np.percentile(v, [75, 25])) < 1e-6: continue
        cols.append(robust_std(v)); keep.append(k)
    E = np.column_stack(cols); n, d = E.shape
    core = np.all(np.abs(E) < 3, axis=1); C = np.corrcoef(E[core].T) + 1e-3 * np.eye(d)
    best = None
    for j in range(d):
        for sign in (1, -1):
            thr = np.percentile(sign * E[:, j], 95); r0 = (sign * E[:, j] >= thr).astype(float)
            fit = em_fit(E, r0, C)
            if best is None or fit[0] > best[0]: best = fit + (j, sign)
    ll, pi, mu1, llr, j0, s0 = best
    auc = lambda s: 100 * roc_auc_score(y[m], s)
    members = [k for k in z.files if k != 'y' and not k.startswith(('SCAN', 'MAG', 'HSCAN', 'H-', 'TG-'))]
    oracle = max(auc(np.nan_to_num(z[k][m])) for k in members)
    top = np.argsort(-np.abs(mu1))[:3]
    print('%-12s | %6.1f %6.1f %6.1f | %8.1f %6.3f | %s' % (ds[:12], oracle, auc(np.nan_to_num(z['SCAN-R'][m])), auc(-np.nan_to_num(z['MAG'][m])), auc(llr), pi, ', '.join('%s %+.2f' % (keep[t], mu1[t]) for t in top)))
