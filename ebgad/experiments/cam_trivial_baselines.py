"""Is CAM a degree or feature-norm artifact? Usage: PYTHONPATH=. python ebgad/experiments/cam_trivial_baselines.py <results_dir> <dataset>.
 Trivial baselines vs CAM on the evaluated population."""
import sys, json, glob, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import degree
from data_utils import load_data
root = sys.argv[1]; ds = sys.argv[2]
z = np.load(f"{root}/{ds}_scores.npz"); y_all = z["y"]
data = load_data(ds); n = data.num_nodes
deg = degree(data.edge_index[0], n).numpy()
x = data.x.numpy().astype(np.float64)
x = (x - x.mean(0)) / (x.std(0) + 1e-12)          # the pipeline z-scores after PCA-on-raw; no PCA here (pca=None)
xn = (x * x).sum(1)
assert y_all.shape[0] == n, (y_all.shape, n)
m = y_all >= 0
y = y_all[m]
def au(s): return 100 * roc_auc_score(y, s[m]), 100 * average_precision_score(y, s[m])
print(f"{ds}: n={n} n_eval={m.sum()} prev={y.mean():.3f}  deg0 frac (eval)={np.mean(deg[m]==0):.4f}  deg0 frac among positives={np.mean(deg[m][y==1]==0):.4f}")
rows = [("CAM", z["CAM"]), ("-degree", -deg), ("+degree", deg), ("-||x||^2 (no-graph CAM)", -xn), ("-J*", -z["J*"]), ("-P", -z["P"]), ("-R", -z["R"]), ("SCAN-R", z["SCAN-R"])]
for name, s in rows:
    a, p = au(s); print(f"  {name:26s} AUROC {a:5.1f}  AUPRC {p:5.1f}")
print(f"  spearman(CAM,-deg)={spearmanr(z['CAM'][m], -deg[m]).correlation:+.3f}  spearman(CAM,-||x||^2)={spearmanr(z['CAM'][m], -xn[m]).correlation:+.3f}  spearman(CAM,-J*)={spearmanr(z['CAM'][m], -z['J*'][m]).correlation:+.3f}")
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]:
    sel = m & (deg >= lo) & (deg < hi)
    if sel.sum() > 50 and 0 < y_all[sel].sum() < sel.sum():
        print(f"  deg in [{lo},{hi:g}): n={sel.sum():6d} prev={y_all[sel].mean():.3f}  CAM AUROC {100*roc_auc_score(y_all[sel], z['CAM'][sel]):5.1f}   -||x||^2 AUROC {100*roc_auc_score(y_all[sel], -xn[sel]):5.1f}")
for f in sorted(glob.glob(f"cache/ebgad/spec_{ds}_*k*_s0*.npz"))[:1]:
    c = np.load(f); lam = c["lam"] if "lam" in c.files else c[c.files[0]]
    print(f"  spectrum {f.split('/')[-1]}: k={lam.size} lam min/median/max = {lam.min():.3e} / {np.median(lam):.3e} / {lam.max():.3e}; #lam<1e-6 = {(lam<1e-6).sum()}")
