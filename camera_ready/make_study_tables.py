"""Generate the appendix study tables from results/camera_ready/studies/*.json.

Usage: make_study_tables.py [output directory]. Writes study_*.tex and results/camera_ready/study_numbers.json.
Missing study files are reported and their tables skipped.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "camera_ready", "tables")
RES = os.path.join(ROOT, "results", "camera_ready")
ST = os.path.join(RES, "studies")

DATASETS = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog", "facebook", "acm",
            "elliptic", "elliptic_plus_plus", "dgraph", "t_finance"]
DISPLAY = {"weibo": "Weibo", "reddit": "Reddit", "amazon": "Amazon", "yelpchi": "YelpChi",
           "blogcatalog": "BlogCatalog", "facebook": "Facebook", "acm": "ACM", "elliptic": "Elliptic",
           "elliptic_plus_plus": "Elliptic++", "dgraph": "DGraph", "t_finance": "T-Finance"}
N = {}
MISSING = []


def load(name):
    p = os.path.join(ST, name)
    if not os.path.exists(p):
        MISSING.append(name)
        return None
    return json.load(open(p))


def write(name, tex):
    open(os.path.join(OUT, name), "w").write(tex)


def bold_max(vals, fmt="%.1f"):
    m = max(round(v, 1) for v in vals)
    return [(r"\textbf{" + fmt % v + "}") if round(v, 1) == m else fmt % v for v in vals]


# ---------------------------------------------------------------------- isolation of the precision family
def isolation():
    fams = ["gou_eb", "gou_fixed", "heat", "poly", "flat"]
    rows, rec = [], {}
    for ds in DATASETS:
        r = load("e2_%s.json" % ds)
        if r is None:
            continue
        r = r[ds] if ds in r else r
        f = r["families"]
        sel = [100 * f[k]["auc_ks_selected"] for k in fams]
        ora = [100 * f[k]["auc_oracle"] for k in fams]
        b = bold_max(sel)
        rows.append("%s & %s \\\\" % (DISPLAY[ds], " & ".join("%s (%.1f)" % (x, o) for x, o in zip(b, ora))))
        rec[ds] = {k: {"selected": round(s, 1), "oracle": round(o, 1)} for k, s, o in zip(fams, sel, ora)}
        rec[ds]["fit"] = {"rho": r["fit"]["rho"], "kappa": r["fit"]["kappa"]}
        best_generic = max(sel[2:])
        rec[ds]["gou_minus_best_generic"] = round(round(sel[0], 1) - round(best_generic, 1), 1)
        rec[ds]["eb_minus_fixed"] = round(round(sel[0], 1) - round(sel[1], 1), 1)
    N["isolation"] = rec
    if not rows:
        return
    write("study_isolation.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Isolation of the spectral precision family. Operator, template, PCA, truncation, residuals and the label-free selection (largest NullKS) are fixed per dataset; only the precision changes. AUROC (\%%) of the selected score, with the best member of the family by AUROC in parentheses (a diagnostic, not a result). \textbf{Bold}: best selected score of the row. GOU-EB: EB-fitted GOU precision with its finite-horizon members; GOU fixed: $\rho=0.5$, $\kappa=1$; heat: $\exp(t\lambda/\lambda_{\max})$, 8 scales; polynomial: $(c+\lambda)^p$, 6 configurations; identity: graph-free residual energy.}
\label{tab:isolation}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l ccccc}
\toprule
Dataset & GOU-EB & GOU fixed & heat & polynomial & identity \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


# ---------------------------------------------------------------------- matched single score
def matched_single():
    rows, rec = [], {}
    t1 = json.load(open(os.path.join(RES, "table1_ebgad.json")))
    for ds in DATASETS:
        r = load("matched_table1_%s.json" % ds)
        if r is None:
            continue
        r = r[ds]
        e, q = r["energy"]["pipeline_grid"], r["ratio"]["pipeline_grid"]
        # family of the comparison: the equilibrium score with the larger NullKS (R where rule 1 fixes it)
        fam, g = ("J*", e) if e["limit"]["nullks"] >= q["limit"]["nullks"] else ("R", q)
        if t1[ds]["first"]["rule"] == 1:
            fam, g = "R", q
        lim, sel, ora = 100 * g["limit"]["auc"], 100 * g["selected_finite"]["auc"], 100 * g["oracle_finite"]["auc"]
        rows.append("%s ($%s$) & %.1f & %.1f & %.1f & $%+.1f$ & $%+.1f$ \\\\" % (
            DISPLAY[ds], "J^\\star" if fam == "J*" else "R", lim, sel, ora,
            round(sel, 1) - round(lim, 1), round(ora, 1) - round(lim, 1)))
        rec[ds] = {"family": fam, "limit": round(lim, 1), "selected": round(sel, 1), "oracle": round(ora, 1),
                   "d_selected": round(round(sel, 1) - round(lim, 1), 1),
                   "d_oracle": round(round(ora, 1) - round(lim, 1), 1),
                   "fit": {"rho": r["fit"]["rho"], "kappa": r["fit"]["kappa"]}, "config": r["config"]}
    N["matched_single"] = rec
    if not rows:
        return
    write("study_matched_single.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Matched single-score comparison at the configuration and EB fit of Table~\ref{tab:ebgad_hyperparams}, shared by both sides. \emph{Equilibrium}: $J^\star$ or $R$, whichever has the larger NullKS ($R$ where rule 1 fixes it). \emph{Finite}: a single hard-endpoint score of the same type ($C_{\Gamma,\infty}$ against $J^\star$, $CR_{\Gamma,\infty}$ against $R$) at one finite horizon $\Gamma\in\{0.02,\dots,50\}$, with no tolerance grid, fusion or aggregation; NullKS picks the horizon, and the best horizon by AUROC is reported as a diagnostic. AUROC (\%%).}
\label{tab:matched_single}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l ccc cc}
\toprule
Dataset (type) & Equilibrium & Finite, selected & Finite, best & $\Delta$ selected & $\Delta$ best \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


# ---------------------------------------------------------------------- selector studies
def cutoffs():
    r = load("selector_cutoffs.json")
    if r is None:
        return
    n_const, n_ds = len(r["constants"]), r["n_datasets"]
    rec = {"constants": n_const, "evaluations_per_factor": n_const * n_ds,
           "pm25_changed": r["pm25"]["changed"], "pm25_total": r["pm25"]["evaluations"],
           "pm50_changed": r["pm50"]["changed"], "pm50_total": r["pm50"]["evaluations"],
           "pm25_unchanged_pct": round(100 * (1 - r["pm25"]["changed"] / r["pm25"]["evaluations"]), 1),
           "pm50_unchanged_pct": round(100 * (1 - r["pm50"]["changed"] / r["pm50"]["evaluations"]), 1),
           "joint_075": r["joint"]["0.75"], "joint_125": r["joint"]["1.25"], "changes": []}
    names = {"de_sparse": "$d_e<1$ (rule 1)", "dx_anchor": "$D_x\\le256$ (rule 1)", "kn_trunc": "$k/n\\le5{\\cdot}10^{-3}$",
             "h_zero": "$h<0.05$", "dx_hi": "$D_x>1000$", "de_dense": "$d_e\\ge20$", "h_weak": "$h<0.25$",
             "de_mid": "$d_e\\ge10$", "de_very_dense": "$d_e\\ge100$", "traj_ks": "dense NullKS $0.50$",
             "traj_bj_lo": "dense BJ $\\ge20$", "traj_bj_hi": "dense BJ $\\le120$", "guard_ks": "guard $0.60$",
             "h_hi": "$h\\ge0.50$", "dx_lo": "$D_x\\le128$", "st_bj_lo": "sparse BJ $\\ge20$",
             "st_bj_hi": "sparse BJ $\\le100$", "st_eqks": "sparse eq.\\ NullKS $0.20$",
             "st_profks": "sparse profile NullKS $0.35$", "pi_lo": "$\\pi\\ge0.50$", "pi_hi": "$\\pi\\le0.95$"}
    fam = {"anchor": "anchor", "control_energy": "energy", "control_ratio": "ratio", "profile": "profile"}
    rows = []
    costs = []
    for c in r["changes"]:
        a0, a1 = c["auc_from"], c["auc_to"]
        cost = None if (a0 is None or a1 is None) else round(a1, 1) - round(a0, 1)
        if cost is not None:
            costs.append(cost)
        rows.append("%s & $\\times%g$ & %s & %s ($%s$) $\\to$ %s ($%s$) & %.1f & %s & %s \\\\" % (
            names[c["cutoff"]], c["factor"], DISPLAY[c["dataset"]],
            fam[c["from"][0]], tag(c["from"]), fam[c["to"][0]], tag(c["to"]), a0,
            "n/a" if a1 is None else "%.1f" % a1, "n/a" if cost is None else "$%+.1f$" % cost))
        rec["changes"].append({"cutoff": c["cutoff"], "factor": c["factor"], "dataset": c["dataset"],
                               "from": c["from"], "to": c["to"], "auc_from": None if a0 is None else round(a0, 1),
                               "auc_to": None if a1 is None else round(a1, 1), "cost": cost})
    if costs:
        rec["cost_min"], rec["cost_max"] = round(min(costs), 1), round(max(costs), 1)
        rec["cost_median"] = round(statistics.median(costs), 1)
    N["cutoffs"] = rec
    write("study_cutoffs.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Cutoff sensitivity of the ordered rules. Each of the %d cutoffs is multiplied by $0.5$, $0.75$, $1.25$ and $1.5$, one at a time, and the rules are re-evaluated on the 11 datasets (%d evaluations per factor). The table lists every evaluation in which the selected family or branch changes, with the AUROC (\%%) before and after (original node order).}
\label{tab:cutoff_sensitivity}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l c l l c c c}
\toprule
Cutoff & Factor & Dataset & Selection & before & after & change \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (n_const, n_const * n_ds, "\n".join(rows)))


def tag(sel):
    return {"J*": "J^\\star", "R": "R", "affinity": "\\text{aff.}", "low_pass": "\\text{low}",
            "scale_entropy": "\\text{entropy}", "sparse_tail": "\\text{sparse}",
            "trajectory": "\\text{traj.}"}[sel[1]]


def ingredients():
    r = load("selector_ingredients.json")
    if r is None:
        return
    rows, rec = [], {}
    bank = {"stationary": "equilibrium", "two_groups_path": "two-groups", "path_dissipation": "path bands",
            "dynamic_profile": "profile"}
    for ds in DATASETS:
        if ds not in r:
            continue
        v = r[ds]
        rows.append("%s & %s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\" % (
            DISPLAY[ds], bank[v["bank"]], v["nullks_only"], v["stability_only"], v["tail_stable"],
            v["corr_stable"], v["rule_pick"], v["oracle"]))
        rec[ds] = {k: round(v[k], 1) for k in ("nullks_only", "stability_only", "tail_stable", "corr_stable",
                                               "rule_pick", "oracle")}
        rec[ds]["gap_to_oracle"] = round(round(v["oracle"], 1) - round(v["rule_pick"], 1), 1)
    N["ingredients"] = rec
    write("study_ingredients.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Ingredients of the within-family rule $A(r)$ on the datasets routed to an equilibrium or profile bank (original node order). AUROC (\%%) of the member picked by quality only (NullKS), by stability only, by the two combined variants (tail-stability, correlation-stability), by the rule used in Table~\ref{tab:main_auroc}, and the best member of the bank by AUROC (a diagnostic). The equilibrium bank contains $J^\star$, $R$ and their rank aggregates; the rule restricts it to $J^\star$ and $R$.}
\label{tab:ar_ablation}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l cccc cc}
\toprule
Dataset & Bank & NullKS & stability & tail-stab. & corr-stab. & rule & best \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def stacking():
    r = load("selector_stacking.json")
    if r is None:
        return
    rows, rec = [], {}
    for ds in DATASETS:
        if ds not in r:
            continue
        v = r[ds]
        gap = round(v["stacked_cv"], 1) - round(v["selector"], 1)
        rows.append("%s & %d & %.1f & %.1f & %.1f & $%+.1f$ \\\\" % (
            DISPLAY[ds], v["columns"], v["selector"], v["best_single"], v["stacked_cv"], gap))
        rec[ds] = {"columns": v["columns"], "selector": round(v["selector"], 1),
                   "best_single": round(v["best_single"], 1), "stacked": round(v["stacked_cv"], 1),
                   "gap": round(gap, 1)}
    N["stacking"] = rec
    write("study_stacking.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Supervised bound on combining the scores. Every score of a dataset's pool is rank-transformed and a logistic regression is fitted on the labels with 5-fold cross-validation; the out-of-fold AUROC (\%%) bounds what a label-driven linear combination of our scores could reach. \emph{Selector}: the label-free pick of Table~\ref{tab:main_auroc} (original node order). \emph{Best single}: best member of the pool by AUROC.}
\label{tab:stacking}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c ccc c}
\toprule
Dataset & Scores & Selector & Best single & Stacked (CV) & Gap \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


# ---------------------------------------------------------------------- robustness
def contamination():
    r = []
    for f in sorted(glob.glob(os.path.join(ST, "parts", "e5_*.json"))):
        r.extend(json.load(open(f)))
    if not r:
        MISSING.append("parts/e5_*.json")
        return
    order = {"weibo": 0, "reddit": 1}
    r.sort(key=lambda x: (order[x["dataset"]], x["extra_frac"]))
    rows, rec = [], {}
    for x in r:
        ds = x["dataset"]
        rows.append("%s & $+%d\\%%$ & %.1f\\%% & (%.3f, %g) & %.1f & %s \\\\" % (
            DISPLAY[ds], round(100 * x["extra_frac"]), 100 * x["total_contamination"], x["rho"], x["kappa"],
            100 * x["auc_original"], "--" if x["auc_injected"] is None else "%.1f" % (100 * x["auc_injected"])))
        rec.setdefault(ds, []).append({"extra": round(100 * x["extra_frac"]), "total": round(100 * x["total_contamination"], 1),
                                       "rho": round(x["rho"], 3), "kappa": x["kappa"],
                                       "auc_original": round(100 * x["auc_original"], 1),
                                       "auc_injected": None if x["auc_injected"] is None else round(100 * x["auc_injected"], 1)})
    N["contamination"] = rec
    write("study_contamination.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Contamination. Contextual anomalies (feature swaps with random nodes) are injected on top of the real anomalies, EB is refitted on the contaminated graph and the score is selected without labels (largest NullKS over the finite-horizon bank). AUROC (\%%) on the original anomalies and on the injected ones (against untouched normal nodes).}
\label{tab:contamination}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c c c c c}
\toprule
Dataset & Injected & Total contamination & Fitted $(\rho,\kappa)$ & Original anomalies & Injected anomalies \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def misspecification():
    r = load("e6_misspec.json")
    if r is None:
        return
    order = ["gaussian", "heavy_tail", "multimodal", "categorical", "categorical_flip"]
    label = {"gaussian": "Gaussian (control)", "heavy_tail": "Student-$t$, 3 d.o.f.", "multimodal": "two-component mixture",
             "categorical": "binarized", "categorical_flip": "binarized, bit-flip anomalies"}
    rows, rec = [], {}
    for v in order:
        xs = sorted([x for x in r if x["variant"] == v], key=lambda x: x["seed"])
        if not xs:
            continue
        sel = [100 * x["auc_sel"] for x in xs]
        ora = [100 * x["auc_oracle"] for x in xs]
        rho = [x["rho"] for x in xs]
        rows.append("%s & %s & %s & %s \\\\" % (label[v], " / ".join("%.1f" % s for s in sel),
                                               " / ".join("%.1f" % s for s in ora), " / ".join("%.2f" % s for s in rho)))
        rec[v] = {"selected": [round(s, 1) for s in sel], "oracle": [round(s, 1) for s in ora],
                  "rho": [round(s, 2) for s in rho]}
    N["misspecification"] = rec
    write("study_misspecification.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Misspecification stress test on stochastic block model graphs ($n=3000$, four blocks, edge probabilities $0.02$ within and $0.002$ between blocks, $D=16$, $5\%%$ injected attribute anomalies, three seeds). The normal signal is a low-pass graph signal plus noise whose distribution is varied; the pipeline, grids and label-free selection are identical in every column. AUROC (\%%) of the selected score and of the best member of the bank (a diagnostic), and the fitted graph trust, per seed.}
\label{tab:misspecification}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c c c}
\toprule
Residual distribution & Selected & Best member & Fitted $\rho$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def neighbors():
    r, e = [], []
    for f in sorted(glob.glob(os.path.join(ST, "parts", "e8_*.json"))):
        r.extend(json.load(open(f)))
    for f in sorted(glob.glob(os.path.join(ST, "parts", "e13_*.json"))):
        e.extend(json.load(open(f)))
    rec = {}
    rows = []
    for pl, name in (("scattered", "scattered"), ("clustered", "connected groups of at most 10")):
        xs = sorted([x for x in r if x["placement"] == pl], key=lambda x: x["seed"])
        if not xs:
            MISSING.append("parts/e8_%s" % pl)
            continue
        sel = [100 * x["auc_injected"] for x in xs]
        ora = [100 * x["auc_injected_oracle"] for x in xs]
        rows.append("%s & %s & %.1f--%.1f & %.1f--%.1f \\\\" % (
            name, " / ".join("%.1f" % v for v in sel), min(sel), max(sel), min(ora), max(ora)))
        rec[pl] = {"selected": [round(v, 1) for v in sel], "oracle": [round(v, 1) for v in ora],
                   "seeds": [x["seed"] for x in xs], "fit": sorted({(round(x["rho"], 3), x["kappa"]) for x in xs})}
    if e:
        e.sort(key=lambda x: x["seed"])
        rec["edges"] = {"min_energy": [round(100 * x["auc_min_node_energy"], 1) for x in e],
                        "cross_std": [round(100 * x["auc_cross_std"], 1) for x in e],
                        "cross_raw": [round(100 * x["auc_cross_raw"], 1) for x in e],
                        "n_pos": [x["n_pos"] for x in e], "seeds": [x["seed"] for x in e]}
        m, c = rec["edges"]["min_energy"], rec["edges"]["cross_std"]
        rows_e = ["smaller endpoint energy & %s & %.1f--%.1f \\\\" % (" / ".join("%.1f" % v for v in m), min(m), max(m)),
                  "standardized cross-term & %s & %.1f--%.1f \\\\" % (" / ".join("%.1f" % v for v in c), min(c), max(c))]
    N["neighbors"] = rec
    if rows:
        write("study_neighbors.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Neighboring anomalies on Reddit. $5\%%$ of the nodes receive a feature swap, either scattered over the graph or as connected groups of at most 10 neighbors; EB is refitted on each contaminated graph and the score is selected without labels. AUROC (\%%) of the injected nodes against untouched normal nodes, per seed.}
\label{tab:neighbors}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c c c}
\toprule
Placement & Selected, seeds 0--4 & Range & Best member, range \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))
    if e:
        write("study_edges.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Pairwise readout on the connected-group runs of Table~\ref{tab:neighbors}. Every undirected edge is scored and edges that join two injected nodes are ranked against edges that join two untouched normal nodes. AUROC (\%%) per seed.}
\label{tab:edges}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c c}
\toprule
Edge score & Seeds 0--4 & Range \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows_e))


def templates():
    rec = {}
    for t in ("affinity", "affinity_high", "affinity_mix"):
        r = load("e10_blogcatalog_%s.json" % t)
        if r is None:
            continue
        r = r["blogcatalog"]
        st = r["stationary"]
        recs = {x["name"]: x for x in st["top_auc"]}
        j, q = recs["J@inf"], recs["R@inf"]
        pick = j if j["ks"] >= q["ks"] else q
        rec[t] = {"selected": round(100 * pick["auc"], 1), "score": pick["name"],
                  "J": round(100 * j["auc"], 1), "R": round(100 * q["auc"], 1),
                  "oracle_bank": round(100 * r["dynamic_combined"]["auc_oracle"], 1),
                  "gamma": r["stationary_fit"]["template_gamma"], "rho": round(r["stationary_fit"]["rho"], 3),
                  "kappa": r["stationary_fit"]["kappa"]}
    N["templates"] = rec


def karate():
    r = load("karate.json")
    if r is not None:
        N["karate"] = r


def ordering():
    a, b = load("ordering_index_ties.json"), load("ordering_average_ties.json")
    rec = {}
    for name, r in (("index_ties", a), ("average_ties", b)):
        if r is None:
            continue
        rec[name] = {ds: {"index_auc": round(r[ds]["index_auc"], 1),
                          "original": {k: round(v, 1) for k, v in r[ds]["original"].items()},
                          "relabeled": {k: round(v, 1) for k, v in r[ds]["relabeled"].items()}}
                     for ds in ("yelpchi", "facebook") if ds in r}
    N["ordering"] = rec
    if a is None or b is None:
        return
    label = {"dp_ebmix_posterior": "profile, mixture posterior", "dp_profile_mean_logp": "profile, mean $-\\log p$",
             "dp_eb_acat": "profile, ACAT", "tg_component_entropy": "profile, component entropy",
             "J@inf": "$J^\\star$", "R@inf": "$R$"}
    rows = []
    for ds in ("yelpchi", "facebook"):
        first = True
        for k in ("dp_ebmix_posterior", "dp_profile_mean_logp", "dp_eb_acat", "tg_component_entropy", "J@inf", "R@inf"):
            if k not in a[ds]["original"] or k not in b[ds]["original"]:
                continue
            rows.append("%s & %s & %.1f & %.1f & %.1f & %.1f & %s \\\\" % (
                DISPLAY[ds] if first else "", label[k], a[ds]["original"][k], a[ds]["relabeled"][k],
                b[ds]["original"][k], b[ds]["relabeled"][k], ("%.1f" % a[ds]["index_auc"]) if first else ""))
            first = False
        if ds == "yelpchi":
            rows.append("\\midrule")
    write("study_ordering.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{The ordering artifact. AUROC (\%%) of the profile scores and of the equilibrium scores on YelpChi and Facebook, in the released node order and after a random relabeling, with the tie-breaking of the submitted version (ties ordered by node index) and with averaged ties. The equilibrium scores involve no rank step. Last column: AUROC of the node index itself.}
\label{tab:ordering}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l cc cc c}
\toprule
& & \multicolumn{2}{c}{ties by node index} & \multicolumn{2}{c}{averaged ties} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Dataset & Score & released & relabeled & released & relabeled & node index \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def gamma_profile():
    """Bandwidth table from the anchor files (camera_ready/anchor_bandwidth.py), original node order."""
    rows, rec = [], {}
    t1 = json.load(open(os.path.join(RES, "table1_ebgad.json")))
    for ds in DATASETS:
        p = os.path.join(RES, "anchor", "anchor_%s_orig.json" % ds)
        if not os.path.exists(p):
            MISSING.append("anchor/anchor_%s_orig.json" % ds)
            continue
        r = json.load(open(p))
        cells = []
        rec[ds] = {"gamma_center": round(r["gamma_center"], 3), "gamma_anchor": r["gamma_nearest"],
                   "gamma_likelihood": r["gamma_likelihood"], "rows": []}
        most_removed = max(r["rows"], key=lambda x: x["removed"])["gamma"]
        rule1 = t1[ds]["first"]["rule"] == 1
        for x in r["rows"]:
            pick = "R" if rule1 else ("J" if x["ks_J"] >= x["ks_R"] else "R")
            auc = x["auc_" + pick]
            txt = "%g: %.1f" % (x["gamma"], auc)
            if abs(x["gamma"] - r["gamma_nearest"]) < 1e-12:
                txt = "\\textbf{%s}" % txt
            if abs(x["gamma"] - r["gamma_likelihood"]) < 1e-12:
                txt += "$^{\\ell}$"
            cells.append("%s (%d)" % (txt, round(100 * max(x["removed"], 0.0))))
            rec[ds]["rows"].append({"gamma": x["gamma"], "auc": round(auc, 1), "score": pick, "ml": x["ml"],
                                    "J": round(x["auc_J"], 1), "R": round(x["auc_R"], 1),
                                    "removed": round(100 * x["removed"], 1),
                                    "rho": round(x["rho"], 3), "kappa": x["kappa"]})
        while len(cells) < 3:
            cells.append("")
        rec[ds]["likelihood_is_most_removed"] = bool(abs(most_removed - r["gamma_likelihood"]) < 1e-12)
        byg = {z["gamma"]: z for z in rec[ds]["rows"]}
        rec[ds]["anchor_auc"] = byg[r["gamma_nearest"]]["auc"]
        rec[ds]["likelihood_auc"] = byg[r["gamma_likelihood"]]["auc"]
        rec[ds]["removed_at_likelihood"] = byg[r["gamma_likelihood"]]["removed"]
        rows.append("%s & %.2f & %s \\\\" % (DISPLAY[ds], r["gamma_center"], " & ".join(cells[:3])))
    N["gamma_profile"] = rec
    # is the NullKS choice of the score type the same in every node order?
    unstable = []
    for ds in rec:
        a = json.load(open(os.path.join(RES, "anchor", "anchor_%s_orig.json" % ds)))["rows"]
        others = [json.load(open(f))["rows"] for f in sorted(glob.glob(os.path.join(RES, "anchor", "anchor_%s_relabel*.json" % ds)))]
        rec[ds]["orders"] = 1 + len(others)
        for i, x in enumerate(a):
            ks = [(x["ks_J"], x["ks_R"])] + [(o[i]["ks_J"], o[i]["ks_R"]) for o in others]
            if len({j >= r for j, r in ks}) > 1:
                unstable.append({"dataset": ds, "gamma": x["gamma"], "ks_R": [round(r, 2) for _, r in ks],
                                 "ks_J": [round(j, 2) for j, _ in ks],
                                 "is_anchor": bool(abs(x["gamma"] - rec[ds]["gamma_anchor"]) < 1e-12)})
    full = [ds for ds in rec if ds in ("weibo", "reddit", "amazon", "blogcatalog", "facebook", "acm")]
    N["gamma_profile_summary"] = {
        "n": len(rec), "pick_unstable": unstable, "likelihood_is_most_removed": sorted(ds for ds in rec if rec[ds]["likelihood_is_most_removed"]),
        "anchor_differs": sorted(ds for ds in rec if rec[ds]["gamma_anchor"] != rec[ds]["gamma_likelihood"]),
        "removed_full_min": min(rec[ds]["removed_at_likelihood"] for ds in full) if full else None,
        "removed_full_max": max(rec[ds]["removed_at_likelihood"] for ds in full) if full else None}
    if rows:
        write("study_gamma_profile.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Template bandwidth. $\gamma_0$: graph-statistic center of the grid. For every bandwidth $\gamma$ of a dataset's grid: AUROC (\%%) of the equilibrium score chosen by NullKS after the EB fit at that bandwidth (low-pass template; $R$ on Elliptic, Elliptic++ and DGraph, where rule 1 fixes it) and, in parentheses, the share (\%%) of the feature energy that the residual map removes (0 when the residual has more energy than the features; the shares are small on truncated spectra, where the map acts on the computed modes only). \textbf{Bold}: the anchor bandwidth $\gamma_a$ (grid value nearest to $\gamma_0$ on the log scale). $^{\ell}$: the bandwidth with the largest residual-space likelihood, used by the finite-horizon families. The AUROC values are diagnostics; no rule reads them. Original node order.}
\label{tab:gamma_profile}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c lll}
\toprule
Dataset & $\gamma_0$ & \multicolumn{3}{l}{$\gamma$: AUROC (removed share), in increasing $\gamma$} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def sensitivity():
    r = load("sensitivity.json")
    if r is None:
        return
    t1 = json.load(open(os.path.join(RES, "table1_ebgad.json")))
    rows, rec = [], {}
    for ds in DATASETS:
        base = t1[ds]["first"]["selected"]["auroc"]
        cells = ["%.1f" % base]
        rec[ds] = {"protocol": round(base, 1)}
        for v in ("likelihood", "average", "dropnull", "jac"):
            x = r.get(v, {}).get(ds)
            if x is None:
                cells.append("--" if (v == "dropnull" and t1[ds]["truncated"]) else "n/a")
                rec[ds][v] = None
            else:
                same = x["score"] == t1[ds]["first"]["score"] and round(x["auroc"], 1) == round(base, 1)
                cells.append("=" if same else "%.1f" % x["auroc"])
                rec[ds][v] = round(x["auroc"], 1)
                rec[ds][v + "_gamma"] = x["config"]["gamma"]
                rec[ds][v + "_family"] = x["family"]
        rows.append("%s & %s \\\\" % (DISPLAY[ds], " & ".join(cells)))
    N["sensitivity"] = rec
    write("study_sensitivity.tex", r"""\begin{table}[!htb]
\centering
\scriptsize
\caption{Sensitivity of Table~\ref{tab:main_auroc} to four conventions (original node order, AUROC \%%; ``='': same score and value; ``--'': not applicable, truncated spectra keep all computed modes; n/a: not run). \emph{Anchor, likelihood}: the anchor uses the bandwidth with the largest residual-space likelihood, as in the submitted Algorithm~\ref{alg:prior_opt}. \emph{Anchor, averaged}: the anchor is the mean percentile rank of the equilibrium score over the three bandwidths of the grid. \emph{Nullspace dropped}: Laplacian null modes are excluded from the modeled subspace on the full-spectrum datasets. \emph{Jacobian}: every bandwidth, including the anchor's, maximizes the likelihood with the term $D\sum_j\log\max(t_j,10^{-12})$ of the residual map $t_j=1-(\gamma^2+\lambda_j)^{-1}$, which excludes bandwidths whose map is negative on a computed mode. The ordered rules and the within-family scores are unchanged.}
\label{tab:sensitivity}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l ccccc}
\toprule
Dataset & Protocol & Anchor, likelihood & Anchor, averaged & Nullspace dropped & Jacobian \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows))


def anchor_variants():
    """The studies of the EB-fitted bank, rerun at the anchor bandwidth (numbers quoted in the text)."""
    rec = {}
    r = []
    for f in sorted(glob.glob(os.path.join(ST, "parts_anchor", "e5_*.json"))):
        r.extend(json.load(open(f)))
    for ds in ("weibo", "reddit"):
        xs = sorted([x for x in r if x["dataset"] == ds], key=lambda x: x["extra_frac"])
        if xs:
            rec.setdefault("contamination", {})[ds] = [
                {"extra": round(100 * x["extra_frac"]), "total": round(100 * x["total_contamination"], 1),
                 "rho": round(x["rho"], 3), "kappa": x["kappa"], "entry": x["entry"],
                 "auc_original": round(100 * x["auc_original"], 1),
                 "auc_injected": None if x["auc_injected"] is None else round(100 * x["auc_injected"], 1)} for x in xs]
    r, e = [], []
    for f in sorted(glob.glob(os.path.join(ST, "parts_anchor", "e8_*.json"))):
        r.extend(json.load(open(f)))
    for f in sorted(glob.glob(os.path.join(ST, "parts_anchor", "e13_*.json"))):
        e.extend(json.load(open(f)))
    for pl in ("scattered", "clustered"):
        xs = sorted([x for x in r if x["placement"] == pl], key=lambda x: x["seed"])
        if xs:
            rec.setdefault("neighbors", {})[pl] = {
                "selected": [round(100 * x["auc_injected"], 1) for x in xs],
                "oracle": [round(100 * x["auc_injected_oracle"], 1) for x in xs],
                "entry": sorted({x["entry"] for x in xs})}
    if e:
        e.sort(key=lambda x: x["seed"])
        rec.setdefault("neighbors", {})["edges"] = {
            "min_energy": [round(100 * x["auc_min_node_energy"], 1) for x in e],
            "cross_std": [round(100 * x["auc_cross_std"], 1) for x in e]}
    k = os.path.join(ST, "karate_anchor.json")
    if os.path.exists(k):
        rec["karate"] = json.load(open(k))
    m = os.path.join(ST, "e6_misspec_anchor.json")
    if os.path.exists(m):
        r = json.load(open(m))
        rec["misspecification"] = {v: [round(100 * x["auc_sel"], 1) for x in sorted(
            [x for x in r if x["variant"] == v], key=lambda x: x["seed"])] for v in sorted({x["variant"] for x in r})}
    # equilibrium score at gamma_0 itself, without the grid (camera_ready/bandwidth_criteria.py)
    t1 = json.load(open(os.path.join(RES, "table1_ebgad.json")))
    direct = {}
    for ds in DATASETS:
        b = os.path.join(ST, "bandwidth", "%s.json" % ds)
        if not os.path.exists(b) or t1[ds]["first"]["family"] != "anchor":
            continue
        c = [x for x in json.load(open(b))["rows"] if x["is_center"]][0]
        pick = "R" if t1[ds]["first"]["rule"] == 1 else ("J" if c["ks_J"] >= c["ks_R"] else "R")
        direct[ds] = {"gamma0": round(c["gamma"], 3), "score": pick, "auc": round(c["auc_" + pick], 1)}
    rec["gamma0_direct"] = direct
    u = os.path.join(ST, "uniform_center_facebook.json")
    if os.path.exists(u):
        rec["uniform_center_facebook"] = json.load(open(u))
    N["anchor_variants"] = rec


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (isolation, matched_single, cutoffs, ingredients, stacking, contamination, misspecification,
              neighbors, templates, karate, ordering, gamma_profile, sensitivity, anchor_variants):
        f()
    json.dump(N, open(os.path.join(RES, "study_numbers.json"), "w"), indent=1, default=str)
    print("missing:", MISSING)
    print("written:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "study_*.tex"))))


if __name__ == "__main__":
    main()
