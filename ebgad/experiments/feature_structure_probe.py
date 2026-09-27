"""What do the reconstruction baselines exploit? Feature-only statistics on the standardized features:
norm, Mahalanobis (empirical covariance), PCA reconstruction error with k components (ANOMALOUS-style),
and the residual to the top-k subspace standardized by the retained noise. DEV AUROC (upper tail) / CAM.
Usage: ... feature_structure_probe.py weibo acm ..."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from run_ebgad import eval_mask
print("%-12s %5s | %-11s | %-11s | %s" % ("dataset", "d", "norm", "Mahalanobis", "PCA-residual k=2 / 4 / 8 / 16 / 32 / 64  (DEV)"))
for name in sys.argv[1:]:
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    x = preprocess_features(data.x, data.edge_index).double().numpy(); n, d = x.shape
    R = lambda z: "%5.1f/%5.1f" % (100 * roc_auc_score(y, z[m]), 100 * roc_auc_score(y, -z[m]))
    norm = (x * x).sum(1)
    xc = x - x.mean(0); U, sv, Wt = np.linalg.svd(xc, full_matrices=False)
    ev = sv ** 2 / (n - 1)
    maha = ((xc @ Wt.T) ** 2 / np.maximum(ev, 1e-8)[None, :]).sum(1)
    out = []
    for k in (2, 4, 8, 16, 32, 64):
        if k >= d: out.append("  -  "); continue
        P = xc @ Wt[:k].T; rec = xc - P @ Wt[:k]; err = (rec * rec).sum(1)
        out.append("%5.1f" % (100 * roc_auc_score(y, err[m])))
    print("%-12s %5d | %s | %s | %s" % (name, d, R(norm), R(maha), " / ".join(out)), flush=True)
