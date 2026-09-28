"""Selector studies on the regenerated runs (original node order).

cutoffs    one-at-a-time and joint perturbation of every cutoff of the ordered rules
ingredients quality-only, stability-only and combined picks inside each routed bank
stacking   supervised upper bound: cross-validated logistic regression over each score pool
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_table1 as S  # noqa: E402

FACTORS = [0.5, 0.75, 1.25, 1.5]


def load(root, ds, order="orig"):
    jf = os.path.join(root, "hyp_%s_%s.json" % (ds, order))
    df = os.path.join(root, "dumps", "hyp_%s_%s" % (ds, order), "hyp_%s_%s_scores.npz" % (ds, order))
    run = json.load(open(jf))[ds]
    y, mask, names, scores = S.load_dump(df)
    return run, y, mask, names, scores


def pool_dump(root, ds, order="orig"):
    prof = S.POOL_PROFILE.get(ds)
    if prof is None:
        return None
    pd_ = os.path.join(root, "dumps", "h12_%s_%s_%s_scores.npz" % (ds, prof, order))
    return S.load_dump(pd_) if os.path.exists(pd_) else None


def auc_of(root, ds, sel, d, cache, anchors):
    """AUROC of the score a routing would select; None when that pool was not run for the dataset."""
    run, y, mask, names, scores = cache[ds]
    rule, family, branch = sel
    if family == "anchor":
        anc = anchors[ds]
        branch = S.resolve_branch(rule, family, branch, anc)
        return S.metrics(y, mask, anc["J"] if branch == "J*" else anc["R"])["auroc"]
    name = S.score_name(family, branch, d)
    if name == "sel/h14_rule":
        pool = pool_dump(root, ds)
        if pool is None:
            return None
        y2, m2, n2, s2 = pool
        return S.metrics(y2, m2, s2[:, n2.index(name)])["auroc"]
    if name not in names:
        return None
    return S.metrics(y, mask, scores[:, names.index(name)])["auroc"]


def routed(st, d, anc, c=None):
    """Ordered rules with the score type of an anchor read at the anchor's configuration."""
    rule, family, branch = S.select_family(st, d, c) if c is not None else S.select_family(st, d)
    return rule, family, S.resolve_branch(rule, family, branch, anc)


def cutoffs(root):
    cache, stats, diag, base, anchors = {}, {}, {}, {}, {}
    for ds in S.DATASETS:
        cache[ds] = load(root, ds)
        run, y, mask, names, scores = cache[ds]
        stats[ds] = S.graph_stats(ds, run, root)
        diag[ds] = S.diagnostics(run, names, scores)
        anchors[ds] = S.anchor_bank(root, ds, "orig", run, names, scores)
        base[ds] = routed(stats[ds], diag[ds], anchors[ds])
    out = {"constants": list(S.CUTOFFS), "factors": FACTORS, "n_datasets": len(S.DATASETS), "changes": [],
           "base": {ds: list(base[ds]) for ds in S.DATASETS}}
    counts = {f: 0 for f in FACTORS}
    for name in S.CUTOFFS:
        for f in FACTORS:
            c = dict(S.CUTOFFS)
            c[name] = c[name] * f
            for ds in S.DATASETS:
                sel = routed(stats[ds], diag[ds], anchors[ds], c)
                if sel[1:] != base[ds][1:]:
                    counts[f] += 1
                    a0 = auc_of(root, ds, base[ds], diag[ds], cache, anchors)
                    a1 = auc_of(root, ds, sel, diag[ds], cache, anchors)
                    out["changes"].append({"cutoff": name, "factor": f, "dataset": ds,
                                           "from": list(base[ds][1:]), "to": list(sel[1:]),
                                           "auc_from": a0, "auc_to": a1})
    total = len(S.CUTOFFS) * len(S.DATASETS)
    out["per_factor"] = {str(f): {"changed": counts[f], "evaluations": total} for f in FACTORS}
    out["pm25"] = {"changed": counts[0.75] + counts[1.25], "evaluations": 2 * total}
    out["pm50"] = {"changed": counts[0.5] + counts[1.5], "evaluations": 2 * total}
    out["joint"] = {}
    for f in (0.75, 1.25):
        c = {k: v * f for k, v in S.CUTOFFS.items()}
        ch = [ds for ds in S.DATASETS if routed(stats[ds], diag[ds], anchors[ds], c)[1:] != base[ds][1:]]
        out["joint"][str(f)] = ch
    return out


ROUTED_BANK = {"anchor": "stationary", ("profile", "scale_entropy"): "two_groups_path",
               ("profile", "trajectory"): "path_dissipation", ("profile", "sparse_tail"): "dynamic_profile"}


def ingredients(root):
    out = {}
    for ds in S.DATASETS:
        run, y, mask, names, scores = load(root, ds)
        st = S.graph_stats(ds, run, root)
        d = S.diagnostics(run, names, scores)
        rule, family, branch = S.select_family(st, d)
        key = ROUTED_BANK.get(family) or ROUTED_BANK.get((family, branch))
        if key is None:
            continue
        bank = run[key]
        pick = S.score_name(family, branch, d)
        rule_pick = None
        if family == "anchor":
            # the equilibrium bank at the anchor's configuration
            anc = S.anchor_bank(root, ds, "orig", run, names, scores)
            bank = anc["bank"]
            branch = S.resolve_branch(rule, family, branch, anc)
            pick = S.score_name(family, branch, d)
            rule_pick = S.metrics(y, mask, anc["J"] if branch == "J*" else anc["R"])["auroc"]
        sel = bank.get("selectors", {})
        if rule_pick is None:
            rule_pick = S.metrics(y, mask, scores[:, names.index(pick)])["auroc"]
        out[ds] = {
            "bank": key, "n_scores": bank.get("n_scores"),
            "nullks_only": 100 * bank["auc_ks_selected"], "nullks_score": bank["score_ks"],
            "stability_only": 100 * sel["stability"]["auc"], "stability_score": sel["stability"]["score"],
            "tail_stable": 100 * sel["tail_stable"]["auc"], "tail_stable_score": sel["tail_stable"]["score"],
            "corr_stable": 100 * sel["corr_stable"]["auc"], "corr_stable_score": sel["corr_stable"]["score"],
            "rule_pick": rule_pick, "rule_score": pick,
            "oracle": 100 * bank["auc_oracle"], "oracle_score": bank["oracle_score"],
        }
    return out


def stack(y, mask, scores, seed=0):
    X = scores[mask].astype(np.float64)
    yy = y[mask]
    keep = X.std(0) > 0
    X = X[:, keep]
    R = np.apply_along_axis(lambda v: rankdata(v, method="average"), 0, X)
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    oof = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(R, yy):
        clf = LogisticRegression(C=1.0, max_iter=300)
        clf.fit(R[tr], yy[tr])
        oof[te] = clf.decision_function(R[te])
    single = max(roc_auc_score(yy, X[:, j]) for j in range(X.shape[1]))
    return 100 * float(roc_auc_score(yy, oof)), int(X.shape[1]), 100 * float(single)


def stacking(root):
    table = json.load(open(os.path.join(root, "table1_ebgad.json")))
    out = {}
    for ds in S.DATASETS:
        run, y, mask, names, scores = load(root, ds)
        pool = pool_dump(root, ds)
        used = "candidate pool" if (pool is not None and table[ds]["first"]["score"] == "sel/h14_rule") else "bank"
        if used == "candidate pool":
            y, mask, names, scores = pool
        else:
            anc = S.anchor_bank(root, ds, "orig", run, names, scores)
            if anc["source"] == "anchor":  # the anchor of the dataset is not a column of the run
                scores = np.column_stack([scores, anc["J"], anc["R"]])
        auc, ncol, single = stack(y, mask, scores)
        first = S.run_one(root, ds, "orig")
        out[ds] = {"pool": used, "columns": ncol, "selector": first["selected"]["auroc"], "best_single": single,
                   "stacked_cv": auc, "gap": auc - first["selected"]["auroc"]}
        print(ds, out[ds], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["cutoffs", "ingredients", "stacking"])
    ap.add_argument("--root", default="results/camera_ready")
    args = ap.parse_args()
    res = {"cutoffs": cutoffs, "ingredients": ingredients, "stacking": stacking}[args.study](args.root)
    path = os.path.join(args.root, "studies", "selector_%s.json" % args.study)
    json.dump(res, open(path, "w"), indent=1, default=str)
    if args.study == "cutoffs":
        print("pm25", res["pm25"], "pm50", res["pm50"], "joint", res["joint"])
        for c in res["changes"]:
            print("  %-14s x%.2f %-19s %s -> %s  auc %s -> %s" % (
                c["cutoff"], c["factor"], c["dataset"], c["from"], c["to"],
                None if c["auc_from"] is None else round(c["auc_from"], 1),
                None if c["auc_to"] is None else round(c["auc_to"], 1)))
    elif args.study == "ingredients":
        for ds, v in res.items():
            print("%-19s %-16s ks %.1f stab %.1f tail %.1f corr %.1f pick %.1f oracle %.1f" % (
                ds, v["bank"], v["nullks_only"], v["stability_only"], v["tail_stable"], v["corr_stable"],
                v["rule_pick"], v["oracle"]))
    print("saved", path)


if __name__ == "__main__":
    main()
