"""Final fraud-family table from results/ebgad_v2_fraud/seed{0,1,2}.

Per dataset: scan-R AUROC and AUPRC (mean and range over eigenbasis seeds),
two-groups readout with validity, equilibrium R, best Table 1 baseline,
permutation gate (seed 0), null-check median KS (seed 0), pipeline
seconds, precision at k = number of labeled anomalies, and lift over the
base rate.
"""
import json, os, sys, numpy as np
STAT = os.environ.get('STAT', 'SCAN-R')
from sklearn.metrics import roc_auc_score, average_precision_score
root = sys.argv[1] if len(sys.argv) > 1 else 'results/ebgad_v2_fraud'
baselines = {  # best unsupervised baseline in Table 1 / AUPRC table (AUROC, AUPRC, name)
    'elliptic': (65.3, 15.3, 'CoLA'), 'elliptic_plus_plus': (62.9, 12.2, 'CoLA'),
    't_finance': (63.1, 6.6, 'CONAD'), 'yelpchi': (57.2, 7.8, 'LOF/TAM'), 'dgraph': (64.2, 1.9, 'DIF')}
def metr(y, s):
    m = (y == 0) | (y == 1); s = np.nan_to_num(s)
    return 100 * roc_auc_score(y[m], s[m]), 100 * average_precision_score(y[m], s[m])
def prec_at_k(y, s):
    m = (y == 0) | (y == 1); s = np.where(m, np.nan_to_num(s), -np.inf)
    k = int((y == 1).sum()); top = np.argsort(-s)[:k]
    p = (y[top] == 1).mean(); base = (y[m] == 1).mean()
    return 100 * p, p / base
print('%-18s | %-14s %-14s | %-10s %-6s | %-14s | %6s %6s %6s | %6s %6s' % (
    'dataset', STAT + ' AUROC', STAT + ' AUPRC', 'TG-R', 'valid', 'best baseline', 'perm', 'KS', 'sec', 'P@k', 'lift'))
for ds in ['elliptic', 'elliptic_plus_plus', 't_finance', 'yelpchi', 'dgraph']:
    aucs, aps, tgs, valid = [], [], [], []
    r0 = None
    for seed in (0, 1, 2):
        pj = os.path.join(root, f'seed{seed}', f'{ds}.json'); pz = os.path.join(root, f'seed{seed}', f'{ds}_scores.npz')
        if not (os.path.exists(pj) and os.path.exists(pz)):
            continue
        r = json.load(open(pj)); z = np.load(pz); y = z['y']
        a, p = metr(y, z[STAT]); aucs.append(a); aps.append(p)
        if 'TG-R' in z.files:
            tgs.append(metr(y, z['TG-R'])[0]); valid.append(r['two_groups']['TG-R']['pi'] < 0.499)
        if seed == 0:
            r0, z0, y0 = r, z, y
    if not aucs:
        print('%-18s | (no runs)' % ds); continue
    perm = r0.get('permute_check', {}).get('spearman', {}).get(STAT, float('nan')) if r0 else float('nan')
    ks = float(np.median([v['ks'] for v in r0['null_check'].values()])) if r0 and r0.get('null_check') else float('nan')
    pk, lift = prec_at_k(y0, z0[STAT])
    bl = baselines.get(ds, (float('nan'), float('nan'), '?'))
    print('%-18s | %5.1f [%4.1f-%4.1f] %5.1f [%4.1f-%4.1f] | %5.1f %-5s %-6s | %5.1f/%4.1f %-5s | %6.3f %6.3f %6.0f | %6.1f %6.2f' % (
        ds, np.mean(aucs), min(aucs), max(aucs), np.mean(aps), min(aps), max(aps),
        np.mean(tgs) if tgs else float('nan'), '', ('yes' if all(valid) else 'no') if valid else '-',
        bl[0], bl[1], bl[2], perm, ks, r0['seconds'], pk, lift))
