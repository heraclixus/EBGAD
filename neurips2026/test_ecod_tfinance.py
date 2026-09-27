"""Re-run ECOD on T-Finance to confirm 83.5% AUROC."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_utils import load_data
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF

data = load_data("t_finance")
x = data.x.float().numpy()
y = data.y.numpy()
mask = (y >= 0) & (y <= 1)

print("T-Finance: n=%d d=%d anom=%.1f%%" % (x.shape[0], x.shape[1], y[mask].mean()*100))

# ECOD
print("\nRunning ECOD...", flush=True)
ecod = ECOD()
ecod.fit(x)
scores_ecod = ecod.decision_scores_
auc_ecod = roc_auc_score(y[mask], scores_ecod[mask])
print("ECOD AUROC: %.1f%%" % (auc_ecod * 100))

# LOF for comparison
print("\nRunning LOF...", flush=True)
lof = LOF()
lof.fit(x)
scores_lof = lof.decision_scores_
auc_lof = roc_auc_score(y[mask], scores_lof[mask])
print("LOF AUROC: %.1f%%" % (auc_lof * 100))

# Also run on labeled subset only
print("\nOn labeled subset only:")
x_lab = x[mask]
y_lab = y[mask]
ecod2 = ECOD()
ecod2.fit(x_lab)
auc_ecod2 = roc_auc_score(y_lab, ecod2.decision_scores_)
print("ECOD (labeled only): %.1f%%" % (auc_ecod2 * 100))
