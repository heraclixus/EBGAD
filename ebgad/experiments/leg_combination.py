"""Combine the local leg (residual magnitude, exact chi^2_d null) with the edge leg (direction relative to the
neighbors, population-calibrated Fisher z) into one label-free statistic in the declared tail.
Under the model the two are independent per node where the template is the population mean (x_i is
spherical), so Fisher's product of the two p-values is chi^2_4 and Stouffer's sum of the two z's is N(0, 2).
Reads results/ebgad_v4_final/seed0/<ds>_scores.npz (s, y) and results/ebgad_v4_edge/<ds>_edge_leg.npz (corr_std).
"""
import json, sys, numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score, average_precision_score
TAIL = {"elliptic": "cam", "elliptic_plus_plus": "cam", "t_finance": "cam", "yelpchi": "cam", "tolokers": "cam",
        "acm": "dev", "blogcatalog": "dev", "amazon": "dev", "weibo": "dev", "reddit": "dev", "facebook": "dev", "questions": "dev"}
print("%-18s %-4s | %6s %6s | %6s %6s %6s %6s | %6s %6s | %s" % ("dataset", "tail", "local", "edge", "fisher", "stouf", "max", "2s-fis", "noG", "AUPRC", "(fisher AUPRC)"))
for ds in ["elliptic", "elliptic_plus_plus", "t_finance", "acm", "facebook", "blogcatalog", "amazon", "weibo", "reddit", "yelpchi", "tolokers", "questions"]:
    f = np.load(f"results/ebgad_v4_final/seed0/{ds}_scores.npz"); e = np.load(f"results/ebgad_v4_edge/{ds}_edge_leg.npz")
    j = json.load(open(f"results/ebgad_v4_final/seed0/{ds}.json")); d = j["d"]
    s, y = f["s"], f["y"]; m = (y == 0) | (y == 1); yy = y[m]; ce = e["corr_std"]
    tail = TAIL[ds]
    if tail == "dev":
        lp_l = -chi2.logsf(d * s, d); lp_e = -norm.logcdf(ce)          # deviating: large s, small e (less similar)
    else:
        lp_l = -chi2.logcdf(d * s, d); lp_e = -norm.logsf(ce)          # camouflage: small s, large e (more similar)
    lp_l = np.nan_to_num(lp_l, posinf=700); lp_e = np.nan_to_num(lp_e, posinf=700)
    fisher = lp_l + lp_e
    zl = norm.isf(np.exp(-np.minimum(lp_l, 700))); ze = norm.isf(np.exp(-np.minimum(lp_e, 700)))
    stouf = np.nan_to_num(zl + ze, posinf=40, neginf=-40)
    mx = np.maximum(lp_l, lp_e)
    # two-sided per leg, then Fisher (no tail prior at all)
    lp_l2 = -np.log(np.clip(2 * np.minimum(chi2.sf(d * s, d), chi2.cdf(d * s, d)), 1e-300, 1)); lp_e2 = -np.log(np.clip(2 * norm.sf(np.abs(ce)), 1e-300, 1))
    fis2 = lp_l2 + lp_e2
    ng = j["no_graph_" + tail.upper()][0]
    A = lambda z: 100 * roc_auc_score(yy, z[m])
    print("%-18s %-4s | %6.1f %6.1f | %6.1f %6.1f %6.1f %6.1f | %6.1f %6.1f | (%.1f)" % (ds, tail, A(lp_l), A(lp_e), A(fisher), A(stouf), A(mx), A(fis2), ng, 100 * average_precision_score(yy, lp_l[m]), 100 * average_precision_score(yy, fisher[m])))
