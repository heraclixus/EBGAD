"""Consolidated suite table from results/ebgad_v2_scan (unit template, EB gamma).

Columns: single-member oracle, best energy member, best ratio member, fixed
J*/R/P, selection-free scans per channel, and label-efficient variants:
  pick-L      : the pool member with the best mean rank of L labeled anomalies
  channel-L   : L labeled anomalies choose only the channel (scan-E vs scan-R)
                by mean rank; the score is that channel's selection-free scan
Means over 300 random label draws.
"""
import json, os, sys, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
res_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ebgad_v2_scan'
paper_eq = {'weibo': 95.1, 'reddit': 61.8, 'amazon': 77.9, 'yelpchi': 70.7, 'blogcatalog': 78.8, 'facebook': 85.9,
            'acm': 83.5, 'elliptic': 73.0, 'elliptic_plus_plus': 72.4, 't_finance': 83.3, 'tolokers': None, 'questions': None}
paper_bank = {'weibo': 94.2, 'reddit': 56.7, 'amazon': 78.1, 'yelpchi': 70.7, 'blogcatalog': 78.6, 'facebook': 91.4,
              'acm': 86.0, 'elliptic': 73.8, 'elliptic_plus_plus': 72.8, 't_finance': 86.8, 'tolokers': None, 'questions': None}
order = ['weibo', 'reddit', 'amazon', 'yelpchi', 'blogcatalog', 'facebook', 'acm', 'elliptic', 'elliptic_plus_plus', 't_finance', 'tolokers', 'questions']
Ls = [1, 3, 5, 10]
hdr = '%-12s %6s %6s %6s | %6s %6s %6s | %6s %6s | %s | %s | %s | %6s %6s' % ('dataset', 'oracle', 'bestE', 'bestR', 'J*', 'R', 'P', 'scanE', 'scanR',
      ' '.join('pick%-2d' % L for L in Ls), ' '.join('chan%-2d' % L for L in Ls),
      ' '.join('%6s' % k for k in ('MAG', 'HSC-E', 'HSC-R', 'HSCAN', 'H-SUM', 'H-CHAN')), 'pap.eq', 'pap.bk')
print(hdr)
rows = []
for name in order:
    p = os.path.join(res_dir, f'{name}_scores.npz')
    if not os.path.exists(p):
        continue
    z = np.load(p); y = z['y']; mask = (y == 0) | (y == 1)
    names = [k for k in z.files if k != 'y']
    auc = {k: 100 * roc_auc_score(y[mask], np.nan_to_num(z[k][mask])) for k in names}
    is_e = lambda k: k in ('J*', 'P') or k.startswith(('D@', 'C@'))
    is_r = lambda k: k in ('R', 'NP') or k.startswith(('DR@', 'CR@'))
    members = [k for k in names if not k.startswith(('SCAN', 'MAG', 'HSCAN', 'H-'))]
    bestE = max(auc[k] for k in members if is_e(k)); bestR = max(auc[k] for k in members if is_r(k))
    pos = np.where(y == 1)[0]
    R = {k: rankdata(np.nan_to_num(z[k])) / len(y) for k in members + ['SCAN-E', 'SCAN-R'] if k in names}
    picks, chans = [], []
    for L in Ls:
        got_p, got_c = [], []
        for _ in range(300):
            idx = rng.choice(pos, size=min(L, len(pos)), replace=False)
            best = max(members, key=lambda k: R[k][idx].mean()); got_p.append(auc[best])
            ch = 'SCAN-E' if R['SCAN-E'][idx].mean() >= R['SCAN-R'][idx].mean() else 'SCAN-R'; got_c.append(auc[ch])
        picks.append(np.mean(got_p)); chans.append(np.mean(got_c))
    fmt = lambda v: '%6.1f' % v if v is not None else '   -  '
    hier = ' '.join('%6.1f' % auc[k] if k in auc else '   -  ' for k in ('MAG', 'HSCAN-E', 'HSCAN-R', 'HSCAN', 'H-SUM', 'H-CHAN'))
    print('%-12s %6.1f %6.1f %6.1f | %6.1f %6.1f %6s | %6.1f %6.1f | %s | %s | %s | %s %s' % (
        name[:12], max(auc[k] for k in members), bestE, bestR, auc['J*'], auc['R'], ('%6.1f' % auc['P']) if 'P' in auc else '  dup ',
        auc['SCAN-E'], auc['SCAN-R'], ' '.join('%6.1f' % v for v in picks), ' '.join('%6.1f' % v for v in chans), hier,
        fmt(paper_eq[name]), fmt(paper_bank[name])))
