"""Label-efficient selection: pick the pool member by the mean rank of L labeled anomalies."""
import sys, glob, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
runs = [('weibo (unit)', 'results/ebgad_v2_scan/weibo_scores.npz'),
        ('weibo (paper)', 'results/ebgad_v2_scan_paper/weibo_scores.npz'),
        ('reddit (unit)', 'results/ebgad_v2_scan/reddit_scores.npz'),
        ('questions (unit)', 'results/ebgad_v2_scan/questions_scores.npz'),
        ('tolokers (unit)', 'results/ebgad_v2_scan/tolokers_scores.npz')]
Ls = [1, 3, 5, 10, 25, 50]
print('%-18s %6s %6s | %s' % ('dataset', 'oracle', 'J*/P', ' '.join('L=%-2d(mean/p10)' % L for L in Ls)))
for tag, path in runs:
    z = np.load(path); y = z['y']
    names = [k for k in z.files if k != 'y' and not k.startswith('SCAN')]
    S = np.vstack([z[k] for k in names]); m = len(names)
    R = np.vstack([rankdata(s) / len(s) for s in S])
    aucs = np.array([100 * roc_auc_score(y, s) for s in S])
    pos = np.where(y == 1)[0]
    fixed = aucs[names.index('P')] if 'P' in names else aucs[names.index('J*')]
    cells = []
    for L in Ls:
        got = []
        for _ in range(300):
            idx = rng.choice(pos, size=min(L, len(pos)), replace=False)
            pick = int(np.argmax(R[:, idx].mean(axis=1)))
            got.append(aucs[pick])
        got = np.array(got)
        cells.append('%5.1f/%5.1f' % (got.mean(), np.percentile(got, 10)))
    print('%-18s %6.2f %6.2f | %s' % (tag, aucs.max(), fixed, '  '.join(cells)))
