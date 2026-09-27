"""Split-half reproducibility of the energy and ratio channels (channel diagnostic).

For each dataset, features are split into two random halves; each channel's
statistic is computed on each half; the top-5% node sets of the two halves
are compared (Jaccard). For the shape (ratio) channel the null is
exchangeable across halves, so chance Jaccard is ~0.026; agreement well
above that means reproducible shape signal. For the magnitude (energy)
channel the node scale s_i correlates the halves for every node, so its
agreement is not a null-calibrated quantity. Printed next to the oracle
channel from the label-based results for comparison.
"""
import json, sys, numpy as np
sys.path.insert(0, '.')
from data_utils import load_data
from ebgad.prep import prepare
from ebgad.fit import q_eigs
from ebgad.scores import profile_specs

runs = [('weibo', 64, None, 'low_unit'), ('reddit', None, None, 'low_unit'), ('amazon', None, None, 'low_unit'),
        ('blogcatalog', None, None, 'low_unit'), ('facebook', None, None, 'low_unit'), ('yelpchi', None, 500, 'low_unit'),
        ('t_finance', None, 500, 'low_unit'), ('elliptic', None, 300, 'low_unit'), ('elliptic_plus_plus', None, 300, 'low_unit'),
        ('tolokers', None, None, 'low_unit'), ('questions', None, 500, 'low_unit')]
rng = np.random.default_rng(0)
def topset(s, frac=0.05):
    k = max(int(frac * len(s)), 10); return set(np.argsort(-s)[:k].tolist())
def jac(a, b): return len(a & b) / len(a | b)
print('%-20s %6s %6s | %8s %8s %8s %8s | oracle channel (E / R AUROC)' % ('dataset', 'n', 'd', 'J-half', 'R-half', 'Dl-half', 'DRl-half'))
for name, pca, k, tmpl in runs:
    try:
        r = json.load(open(f'results/ebgad_v2_scan/{name}.json'))
    except FileNotFoundError:
        print(name, 'no result json'); continue
    data = load_data(name)
    prep = prepare(name, data, r['gamma'], template=tmpl, pca=pca, k=k, cache_dir='cache/ebgad')
    q = q_eigs(prep.lam, r['rho'], r['kappa'])
    d = prep.d; perm = rng.permutation(d); A, B = perm[: d // 2], perm[d // 2:]
    V = prep.V; c_late = profile_specs(q)[-1].c
    out = []
    for c in (q, c_late):
        e, rr = {}, {}
        for tag, cols in (('A', A), ('B', B)):
            u = V @ (np.sqrt(c)[:, None] * prep.Xi[:, cols]); en = (u * u).sum(1)
            den = np.maximum((prep.delta[:, cols] ** 2).sum(1), 1e-12)
            e[tag], rr[tag] = en, en / den
        out += [jac(topset(e['A']), topset(e['B'])), jac(topset(rr['A']), topset(rr['B']))]
    m = r['metrics']
    eE = max(v['auroc'] for kk, v in m.items() if kk in ('J*', 'P') or kk.startswith(('D@', 'C@')))
    eR = max(v['auroc'] for kk, v in m.items() if kk == 'R' or kk.startswith(('DR@', 'CR@')))
    print('%-20s %6d %6d | %8.3f %8.3f %8.3f %8.3f | %s (%.1f / %.1f)' % (name, prep.n, d, out[0], out[1], out[2], out[3], 'ratio' if eR > eE else 'energy', eE, eR), flush=True)
