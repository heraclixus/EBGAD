"""The variant that PROTOCOL.md (second amendment) rejects: the bandwidth nearest to gamma_0 for every
family. Reads a run made with --bandwidth-rule center and applies the ordered rules to it."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_table1 as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/camera_ready_center")
    ap.add_argument("--dataset", default="facebook")
    ap.add_argument("--out", default="results/camera_ready/studies/uniform_center_facebook.json")
    a = ap.parse_args()
    ds = a.dataset
    run = json.load(open(os.path.join(a.root, "hyp_%s_orig.json" % ds)))[ds]
    y, mask, names, scores = S.load_dump(os.path.join(a.root, "dumps", "hyp_%s_orig" % ds, "hyp_%s_orig_scores.npz" % ds))
    st = S.graph_stats(ds, run, a.root)
    d = S.diagnostics(run, names, scores)
    rule, family, branch = S.select_family(st, d)
    name = S.score_name(family, branch, d)
    auc = lambda n: S.metrics(y, mask, scores[:, names.index(n)])["auroc"]
    out = {"gamma": run["stationary_fit"]["template_gamma"], "rule": rule, "family": family, "branch": branch,
           "score": name, "auroc": auc(name), "J": auc("stationary/J@inf"), "R": auc("stationary/R@inf"),
           "component_entropy": auc("two_groups/tg_component_entropy")}
    json.dump(out, open(a.out, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
