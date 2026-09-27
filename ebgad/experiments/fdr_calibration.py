"""Calibration of the two-groups posterior: realized precision vs nominal.

For a run's TG-R posterior r_i (probability non-null), flag nodes with
r_i >= 1 - q for q in a grid. Nominal precision among flagged nodes is the
mean of r_i over the flagged set (1 - local fdr averaged = 1 - FDR);
realized precision is the fraction of flagged nodes with label 1, over
labeled nodes only. Also reports the number flagged and recall.
"""
import sys, glob, os, numpy as np
res_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/ebgad_v2_fraud/seed0'
for p in sorted(glob.glob(os.path.join(res_dir, '*_scores.npz'))):
    z = np.load(p); y = z['y']
    if 'TG-R-post' not in z.files:
        continue
    r = z['TG-R-post']; m = (y == 0) | (y == 1)
    name = os.path.basename(p).replace('_scores.npz', '')
    print('\n%s: labeled %d, anomalies %d (%.2f%%), fitted pi = %.3f' % (name, m.sum(), (y == 1).sum(), 100 * (y == 1).sum() / m.sum(), r[m].mean()))
    print('  %8s %8s %10s %10s %8s' % ('q', 'flagged', 'nominal', 'realized', 'recall'))
    for q in (0.05, 0.1, 0.2, 0.3, 0.5):
        f = (r >= 1 - q) & m
        if f.sum() == 0:
            print('  %8.2f %8d %10s %10s %8s' % (q, 0, '-', '-', '-')); continue
        print('  %8.2f %8d %10.3f %10.3f %8.3f' % (q, f.sum(), r[f].mean(), (y[f] == 1).mean(), (y[f] == 1).sum() / max((y[m] == 1).sum(), 1)))
