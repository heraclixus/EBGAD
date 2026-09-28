"""Sensitivity of Table 1 to two conventions of the protocol (original node order).

dropnull  Laplacian null modes are excluded from the modeled subspace on the full-spectrum datasets
          (runs archived in results/camera_ready_dropnull)
jac       the template bandwidth is chosen with the Jacobian term of the residual map

The same ordered rules and within-family scores are applied to each variant.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_table1 as S  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/camera_ready"
DROP = ROOT + "_dropnull"
FULL = ["weibo", "reddit", "amazon", "blogcatalog", "facebook", "acm"]
out = {"dropnull": {}, "jac": {}}


def record(variant, ds, r, base):
    out[variant][ds] = {"rule": r["rule"], "family": r["family"], "branch": r["branch"], "score": r["score"],
                        "config": r["config"], "auroc": r["selected"]["auroc"], "auprc": r["selected"]["auprc"],
                        "J": r["J*"]["auroc"], "R": r["R"]["auroc"],
                        "protocol_auroc": base["selected"]["auroc"], "protocol_score": base["score"],
                        "protocol_gamma": base["config"]["gamma"]}
    print("%-9s %-19s rule %d %-15s %-32s gamma %g AUROC %.2f (protocol %.2f, gamma %g)" % (
        variant, ds, r["rule"], r["family"], r["score"], r["config"]["gamma"], r["selected"]["auroc"],
        base["selected"]["auroc"], base["config"]["gamma"]), flush=True)


for ds in FULL:
    r, base = S.run_one(DROP, ds, "orig"), S.run_one(ROOT, ds, "orig")
    if r is not None and base is not None:
        record("dropnull", ds, r, base)
for ds in S.DATASETS:
    order = "orig_jac"
    if not os.path.exists(os.path.join(ROOT, "hyp_%s_%s.json" % (ds, order))):
        continue
    prof = S.POOL_PROFILE.get(ds)
    if prof is not None:
        # the pool rule has no bandwidth likelihood: under "jac" it is the pool of the protocol
        pj = os.path.join(ROOT, "h12_%s_%s_%s.json" % (ds, prof, order))
        pd_ = os.path.join(ROOT, "dumps", "h12_%s_%s_%s_scores.npz" % (ds, prof, order))
        src_j = os.path.join(ROOT, "h12_%s_%s_orig.json" % (ds, prof))
        src_d = os.path.join(ROOT, "dumps", "h12_%s_%s_orig_scores.npz" % (ds, prof))
        if not os.path.exists(pj) and os.path.exists(src_j) and os.path.exists(src_d):
            os.symlink(os.path.abspath(src_j), pj)
            os.symlink(os.path.abspath(src_d), pd_)
    r, base = S.run_one(ROOT, ds, order), S.run_one(ROOT, ds, "orig")
    if r is not None and base is not None:
        record("jac", ds, r, base)
json.dump(out, open(os.path.join(ROOT, "studies", "sensitivity.json"), "w"), indent=1, default=str)
print("saved")
