import sys, json, numpy as np
sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
qs = [1, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
def show(tag, path, names):
    z = np.load(path); y = z['y']
    r = json.load(open(path.replace('_scores.npz', '.json')))
    print('\n== %s ==' % tag)
    print('%-10s %6s | %s | anomalies: median pct-rank, frac in top 5%%, top 1%%' % ('stat', 'AUROC', ' '.join('%6s' % ('q%g' % q) for q in qs)))
    for nm in names:
        s = z[nm]
        w = np.log(np.maximum(s, 1e-300))
        w = w - np.median(w)
        qv = np.percentile(w, qs)
        ranks = (np.argsort(np.argsort(s)) + 1) / len(s)
        a = ranks[y == 1]
        auc = 100 * roc_auc_score(y, s)
        print('%-10s %6.2f | %s | %.3f  %.2f  %.2f' % (nm, auc, ' '.join('%6.2f' % v for v in qv), np.median(a), (a > 0.95).mean(), (a > 0.99).mean()))
show('weibo, unit template (calibrated fit)', 'results/ebgad_v2_profile/weibo_scores.npz', ['D@0.2', 'D@0.01', 'D@0.001', 'D@1e-05', 'DR@1e-05', 'DR@0.01'])
show('weibo, paper template', 'results/ebgad_v2_profile_paper/weibo_scores.npz', ['J*', 'P', 'D@0.1', 'DR@0.01', 'CR@0.5,1'])
show('reddit, unit', 'results/ebgad_v2_profile/reddit_scores.npz', ['J*', 'D@0.2', 'DR@0.01', 'D@1e-05'])
show('questions, unit', 'results/ebgad_v2_profile/questions_scores.npz', ['P', 'D@1e-05', 'D@0.01', 'J*', 'DR@1e-05'])
show('tolokers, unit', 'results/ebgad_v2_profile/tolokers_scores.npz', ['DR@1e-05', 'D@0.001', 'J*', 'DR@0.01'])
