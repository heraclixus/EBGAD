"""Label-free selection for Table 1, applied to the regenerated runs.

Implements camera_ready/PROTOCOL.md: the ordered family rules of Appendix B
evaluated on unlabeled graph statistics and run diagnostics, then the
within-family score. Labels are read only to evaluate the selected score.

Inputs (per dataset and node order): the JSON and score dump written by
test_dynamic_hypotheses.py (equilibrium and profile families) and, for the
control-energy and control-ratio families, by test_h12_transport_filterbank.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
from test_multiscale_gou_poe import ks_null_deviation  # noqa: E402

DATASETS = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog", "facebook", "acm",
            "elliptic", "elliptic_plus_plus", "dgraph", "t_finance"]
ORDERS = ["orig", "relabel123", "relabel124", "relabel125", "relabel126"]
POOL_PROFILE = {"amazon": "fine", "blogcatalog": "fine", "acm": "h14", "reddit": "fine"}  # reddit: sensitivity study only
# Bandwidth of the equilibrium anchor (PROTOCOL.md, item 4): "center" is the protocol; "likelihood" and
# "average" are the variants of the sensitivity table.
ANCHOR_BANDWIDTH = os.environ.get("EBGAD_ANCHOR_BANDWIDTH", "center")

# Cutoffs of the ordered rules (Appendix B). `scale` multiplies one of them in the sensitivity study.
CUTOFFS = {
    "de_sparse": 1.0, "dx_anchor": 256.0, "kn_trunc": 5e-3,          # rule 1
    "h_zero": 0.05, "dx_hi": 1000.0, "de_dense": 20.0,                # rule 2
    "h_weak": 0.25, "de_mid": 10.0,                                   # rule 3
    "de_very_dense": 100.0,                                           # rule 4
    "traj_ks": 0.50, "traj_bj_lo": 20.0, "traj_bj_hi": 120.0,         # dense profile diagnostic
    "guard_ks": 0.60,                                                 # anchor guard
    "h_hi": 0.50, "dx_lo": 128.0,                                     # rule 5
    "st_bj_lo": 20.0, "st_bj_hi": 100.0, "st_eqks": 0.20, "st_profks": 0.35,  # sparse-tail diagnostic
    "pi_lo": 0.50, "pi_hi": 0.95,                                     # scale-entropy diagnostic
}


def truncation(n: int):
    if n < 20_000:
        return None
    if n >= 1_000_000:
        return 128
    if n >= 500_000:
        return 200
    if n >= 100_000:
        return 300
    return 500


def metrics(y, mask, s):
    yy, ss = y[mask], np.nan_to_num(np.asarray(s, np.float64))[mask]
    p, r, _ = precision_recall_curve(yy, ss)
    return {"auroc": 100 * float(roc_auc_score(yy, ss)), "auprc": 100 * float(sk_auc(r, p)),
            "ap": 100 * float(average_precision_score(yy, ss))}


def load_dump(path):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    return z["y"], z["mask"], names, z["scores"]


def diagnostics(run, names, scores):
    """Unlabeled quantities the rules read."""
    col = {n: i for i, n in enumerate(names)}
    ks_j = ks_null_deviation(np.nan_to_num(scores[:, col["stationary/J@inf"]].astype(np.float64)))
    ks_r = ks_null_deviation(np.nan_to_num(scores[:, col["stationary/R@inf"]].astype(np.float64)))
    rg = run.get("regime_gated_meta", {})
    tg = run.get("two_groups_path", {}).get("selectors", {}) or {}
    return {
        "ks_j": ks_j, "ks_r": ks_r,
        "profile_bj": float(rg.get("sparse", {}).get("bj", 0.0)),
        "profile_ks": float(rg.get("sparse", {}).get("ks", 0.0)),
        "pi": float((run.get("two_groups_path_meta") or {}).get("pi", 0.0)),
        "tg_tail": (tg.get("tail_stable") or {}).get("score"),
        "tg_graph": (tg.get("graph_tail") or {}).get("score"),
        "path_ks_pick": run.get("path_dissipation", {}).get("score_ks"),
    }


def anchor_pick(d):
    return "J*" if d["ks_j"] >= d["ks_r"] else "R"


def anchor_file(root, ds, order):
    jf = os.path.join(root, "anchor", "anchor_%s_%s.json" % (ds, order))
    if not os.path.exists(jf):
        raise FileNotFoundError("no anchor file %s (camera_ready/anchor_bandwidth.py)" % jf)
    z = np.load(jf[:-5] + "_scores.npz", allow_pickle=True)
    return json.load(open(jf)), [str(x) for x in z["names"]], z["scores"]


def anchor_bank(root, ds, order, run, names, scores, rule=None):
    """J*, R and their NullKS for the equilibrium anchor under a bandwidth rule.

    likelihood  the configuration of the run (residual-space likelihood over the bandwidth grid)
    center      the grid bandwidth nearest to the graph-statistic center gamma_0, EB fit of (rho, kappa)
    average     mean percentile rank over the grid bandwidths, each with its own EB fit
    When the rule gives the bandwidth of the run, the scores of the run are the anchor.
    """
    rule = rule or ANCHOR_BANDWIDTH
    col = {n: i for i, n in enumerate(names)}
    fit = run["stationary_fit"]
    a = {"J": scores[:, col["stationary/J@inf"]], "R": scores[:, col["stationary/R@inf"]], "rule": rule,
         "gamma": float(fit["template_gamma"]), "rho": float(fit["rho"]), "kappa": float(fit["kappa"]),
         "source": "run", "bank": run["stationary"]}
    if rule == "center" and "eq_anchor/J@inf" in col and run.get("eq_anchor_fit"):
        af = run["eq_anchor_fit"]  # runs of the current pipeline carry the anchor
        a.update(J=scores[:, col["eq_anchor/J@inf"]], R=scores[:, col["eq_anchor/R@inf"]],
                 gamma=float(af["template_gamma"]), rho=float(af["rho"]), kappa=float(af["kappa"]),
                 bank=run["eq_anchor"])
    elif rule == "center":
        meta, zn, zs = anchor_file(root, ds, order)
        g = float(meta["gamma_nearest"])
        if abs(g - a["gamma"]) > 1e-9:
            row = [r for r in meta["rows"] if abs(r["gamma"] - g) < 1e-9][0]
            a.update(J=zs[:, zn.index("bw/J@g=%g" % g)], R=zs[:, zn.index("bw/R@g=%g" % g)], gamma=g,
                     rho=float(row["rho"]), kappa=float(row["kappa"]), source="anchor", bank=row.get("bank"))
    elif rule == "average":
        meta, zn, zs = anchor_file(root, ds, order)
        pct = lambda v: (rankdata(np.nan_to_num(v.astype(np.float64)), method="average") - 1.0) / max(len(v) - 1, 1)
        gs = [r["gamma"] for r in meta["rows"]]
        a.update(J=np.mean([pct(zs[:, zn.index("bw/J@g=%g" % g)]) for g in gs], axis=0),
                 R=np.mean([pct(zs[:, zn.index("bw/R@g=%g" % g)]) for g in gs], axis=0),
                 gamma=None, rho=None, kappa=None, source="anchor", bank=None,
                 ks_j=float(np.mean([r["ks_J"] for r in meta["rows"]])),
                 ks_r=float(np.mean([r["ks_R"] for r in meta["rows"]])))
    elif rule != "likelihood":
        raise ValueError(rule)
    if "ks_j" not in a:
        a["ks_j"] = ks_null_deviation(np.nan_to_num(np.asarray(a["J"], np.float64)))
        a["ks_r"] = ks_null_deviation(np.nan_to_num(np.asarray(a["R"], np.float64)))
    return a


def resolve_branch(rule, family, branch, anc):
    """The score type of an anchor is read at the anchor's configuration; rule 1 fixes R."""
    if family != "anchor" or rule == 1:
        return branch
    return anchor_pick(anc)


def select_family(st, d, c=CUTOFFS):
    """Ordered rules of Appendix B. Returns (rule number, family, branch)."""
    h, de, dx, n, k = st["h"], st["d_e"], st["D_x"], st["n"], st["k"]
    truncated = k is not None and k / n <= c["kn_trunc"]
    eq_ks = max(d["ks_j"], d["ks_r"])
    guard = anchor_pick(d) == "J*" and d["ks_j"] >= c["guard_ks"]
    dense_ok = d["ks_j"] >= c["traj_ks"] and c["traj_bj_lo"] <= d["profile_bj"] <= c["traj_bj_hi"]
    sparse_ok = (c["st_bj_lo"] <= d["profile_bj"] <= c["st_bj_hi"] and eq_ks < c["st_eqks"]
                 and d["profile_ks"] < c["st_profks"])
    entropy_ok = (c["pi_lo"] <= d["pi"] <= c["pi_hi"] and d["tg_tail"] == "tg_component_entropy"
                  and d["tg_graph"] == "tg_component_entropy")
    if de < c["de_sparse"] and dx <= c["dx_anchor"] and truncated:
        return 1, "anchor", "R"
    if h < c["h_zero"] and dx > c["dx_hi"] and de >= c["de_dense"]:
        return 2, "control_energy", "affinity"
    if c["h_zero"] <= h < c["h_weak"] and dx > c["dx_hi"] and de < c["de_mid"]:
        return 3, "control_ratio", "affinity"
    if de >= c["de_very_dense"]:
        return 4, "profile", "trajectory"
    if c["de_mid"] <= de < c["de_very_dense"] and dense_ok and not guard:
        return 4, "profile", "trajectory"
    if h >= c["h_hi"] and dx <= c["dx_lo"]:
        if sparse_ok:
            return 5, "profile", "sparse_tail"
        if de >= c["de_mid"]:
            return 5, "control_energy", "low_pass"
        return 5, "anchor", anchor_pick(d)
    if c["h_weak"] <= h < c["h_hi"] and c["dx_lo"] < dx <= c["dx_hi"] and de >= c["de_dense"] and entropy_ok:
        return 6, "profile", "scale_entropy"
    return 7, "anchor", anchor_pick(d)


def score_name(family, branch, d):
    if family == "anchor":
        return "stationary/J@inf" if branch == "J*" else "stationary/R@inf"
    if family == "profile":
        return {"scale_entropy": "two_groups/tg_component_entropy",
                "sparse_tail": "profile/dp_ebmix_posterior",
                "trajectory": "path/%s" % d["path_ks_pick"]}[branch]
    return "sel/h14_rule"


def graph_stats(ds, run, root):
    g = run["graph"]
    if ds == "dgraph":
        cache = os.path.join(root, "dgraph_graph_stats.json")
        if os.path.exists(cache):
            st = json.load(open(cache))
        else:
            from data_utils import load_data
            from test_multiscale_gou_poe import compute_graph_stats
            os.environ.pop("EBGAD_RELABEL_SEED", None)
            h, de = compute_graph_stats(load_data("dgraph"))
            st = {"h": float(h), "d_e": float(de)}
            json.dump(st, open(cache, "w"))
        n = int(g["nodes_labeled"])
        return {"h": st["h"], "d_e": st["d_e"], "D_x": int(g["dim"]), "n": n, "k": 128, "pca": None}
    n = int(g["nodes"])
    return {"h": float(g["homophily_cos"]), "d_e": float(g["density"]), "D_x": int(g["dim"]), "n": n,
            "k": truncation(n), "pca": g.get("pca_dim")}


def run_one(root, ds, order, cutoffs=CUTOFFS, want_scores=False, anchor_rule=None):
    jf = os.path.join(root, "hyp_%s_%s.json" % (ds, order))
    df = os.path.join(root, "dumps", "hyp_%s_%s" % (ds, order), "hyp_%s_%s_scores.npz" % (ds, order))
    if not (os.path.exists(jf) and os.path.exists(df)):
        return None
    run = json.load(open(jf))[ds]
    y, mask, names, scores = load_dump(df)
    col = {n: i for i, n in enumerate(names)}
    st = graph_stats(ds, run, root)
    d = diagnostics(run, names, scores)  # the rules read the diagnostics of the EB-fitted run
    anc = anchor_bank(root, ds, order, run, names, scores, anchor_rule)
    rule, family, branch = select_family(st, d, cutoffs)
    branch = resolve_branch(rule, family, branch, anc)
    name = score_name(family, branch, d)
    fit = run["stationary_fit"]
    cfg = {"template": "low-pass", "gamma": fit["template_gamma"], "rho": fit["rho"], "kappa": fit["kappa"],
           "pca": st["pca"], "k": st["k"]}
    if family == "anchor":
        cfg.update(gamma=anc["gamma"], rho=anc["rho"], kappa=anc["kappa"])
    detail = {}
    if name == "sel/h14_rule":
        prof = POOL_PROFILE.get(ds, "fine")
        pj = os.path.join(root, "h12_%s_%s_%s.json" % (ds, prof, order))
        pd_ = os.path.join(root, "dumps", "h12_%s_%s_%s_scores.npz" % (ds, prof, order))
        if not (os.path.exists(pj) and os.path.exists(pd_)):
            return None
        pr = json.load(open(pj))[ds]["selectors"]["h14_rule"]
        y2, mask2, names2, scores2 = load_dump(pd_)
        assert np.array_equal(y2, y)
        vec = scores2[:, names2.index("sel/h14_rule")]
        cfg = {"template": pr["template_type"], "gamma": pr["gamma"], "rho": pr["rho"], "kappa": pr["kappa"],
               "pca": pr.get("pca_dim"), "k": st["k"]}
        detail = {"pool_rule": pr["selector"], "members": pr.get("fused_scores") or [pr["score"]]}
        # equilibrium scores of the same configuration (matched comparator of the ablation)
        prefix = "cand/tmpl=%s,gamma=%g,pca=%s/" % (pr["template_type"], pr["gamma"], pr.get("pca_dim"))
        mj = scores2[:, names2.index(prefix + "J")]
        mr = scores2[:, names2.index(prefix + "R")]
        kj = ks_null_deviation(np.nan_to_num(mj.astype(np.float64)))
        kr = ks_null_deviation(np.nan_to_num(mr.astype(np.float64)))
        matched = dict(metrics(y, mask, mj if kj >= kr else mr), score="J*" if kj >= kr else "R",
                       ks_j=kj, ks_r=kr, J=metrics(y, mask, mj)["auroc"], R=metrics(y, mask, mr)["auroc"])
    elif family == "anchor":
        vec = anc["J"] if branch == "J*" else anc["R"]
        matched = None
    else:
        vec = scores[:, col[name]]
        matched = None
    # the anchor of the dataset, and the equilibrium score at the configuration of the selected score
    eq_pick = "R" if (rule == 1) else anchor_pick(anc)
    anchor_m = dict(metrics(y, mask, anc["J"] if eq_pick == "J*" else anc["R"]), score=eq_pick)
    if matched is None:
        if family == "anchor":
            matched = dict(anchor_m)
        else:
            fit_pick = anchor_pick(d)
            matched = dict(metrics(y, mask, scores[:, col[score_name("anchor", fit_pick, d)]]), score=fit_pick)
    out = {
        "dataset": ds, "order": order, "stats": st, "diagnostics": d, "rule": rule, "family": family,
        "branch": branch, "score": name, "config": cfg, "detail": detail,
        "anchor": {"bandwidth_rule": anc["rule"], "gamma": anc["gamma"], "rho": anc["rho"], "kappa": anc["kappa"],
                   "ks_j": anc["ks_j"], "ks_r": anc["ks_r"], "source": anc["source"], "pick": eq_pick,
                   "gamma_fit": float(fit["template_gamma"])},
        "fit": {"gamma": float(fit["template_gamma"]), "rho": float(fit["rho"]), "kappa": float(fit["kappa"])},
        "selected": metrics(y, mask, vec),
        "equilibrium": anchor_m,
        "equilibrium_matched": matched,
        "J*": metrics(y, mask, anc["J"]),
        "R": metrics(y, mask, anc["R"]),
        "J*_fit": metrics(y, mask, scores[:, col["stationary/J@inf"]]),
        "R_fit": metrics(y, mask, scores[:, col["stationary/R@inf"]]),
        "n_eval": int(mask.sum()), "anomaly_rate": 100 * float(y[mask].mean()),
    }
    if want_scores:
        out["_vec"] = np.asarray(vec, np.float64)
    return out


def relabel_gate(root, ds, seed=123):
    """Spearman correlation of the selected score between the original and a relabeled node order."""
    a = run_one(root, ds, "orig", want_scores=True)
    b = run_one(root, ds, "relabel%d" % seed, want_scores=True)
    if a is None or b is None or a["score"] != b["score"]:
        return None if (a is None or b is None) else {"same_score": False, "orig": a["score"], "relabel": b["score"]}
    perm = np.random.RandomState(seed).permutation(len(a["_vec"]))
    if ds == "dgraph":
        return {"same_score": True, "spearman": None}
    rho = float(spearmanr(a["_vec"][perm], b["_vec"]).correlation)
    return {"same_score": True, "spearman": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/camera_ready")
    ap.add_argument("--out", default="results/camera_ready/table1_ebgad.json")
    ap.add_argument("--anchor-bandwidth", default=None, choices=["center", "likelihood", "average"])
    args = ap.parse_args()
    if args.anchor_bandwidth:
        global ANCHOR_BANDWIDTH
        ANCHOR_BANDWIDTH = args.anchor_bandwidth
    table = {}
    for ds in DATASETS:
        runs = [r for r in (run_one(args.root, ds, o) for o in ORDERS) if r is not None]
        if not runs:
            print("%-20s no runs" % ds)
            continue
        first = runs[0]
        agg = {}
        for key in ("selected", "equilibrium", "equilibrium_matched", "J*", "R", "J*_fit", "R_fit"):
            for m in ("auroc", "auprc", "ap"):
                vals = [r[key][m] for r in runs]
                agg["%s_%s_mean" % (key, m)] = float(np.mean(vals))
                agg["%s_%s_std" % (key, m)] = float(np.std(vals))
                agg["%s_%s_all" % (key, m)] = vals
        families = sorted({(r["rule"], r["family"], r["branch"], r["score"]) for r in runs})
        gate = relabel_gate(args.root, ds)
        table[ds] = {"n_orders": len(runs), "orders": [r["order"] for r in runs], "first": first,
                     "aggregate": agg, "selections": [list(f) for f in families], "gate": gate,
                     "truncated": first["stats"]["k"] is not None}
        print("%-20s rule %d %-15s %-14s %-34s gamma %s AUROC %.2f +- %.2f (n=%d) AUPRC %.2f | eq %.2f | J* %.2f R %.2f | gate %s" % (
            ds, first["rule"], first["family"], first["branch"], first["score"], first["config"]["gamma"],
            agg["selected_auroc_mean"], agg["selected_auroc_std"], len(runs), agg["selected_auprc_mean"],
            agg["equilibrium_matched_auroc_mean"], agg["J*_auroc_mean"], agg["R_auroc_mean"], gate))
        if len(families) > 1:
            print("    selections differ across node orders:", families)
    json.dump(table, open(args.out, "w"), indent=1, default=str)
    print("saved", args.out)


if __name__ == "__main__":
    main()
