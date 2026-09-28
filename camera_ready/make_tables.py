"""Generate the tables of the paper that contain EB-GAD numbers.

Sources
  results/camera_ready/table1_ebgad.json   regenerated label-free EB-GAD results (select_table1.py)
  camera_ready/baselines.json              baseline numbers and their sources
  results/camera_ready/timing/timing.jsonl wall-clock times (time_pipeline.sh)

Usage: make_tables.py [output directory]. Writes the tables as LaTeX and
results/camera_ready/paper_numbers.json (every number printed in a table).
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "camera_ready", "tables")
RES = os.path.join(ROOT, "results", "camera_ready")

DATASETS = ["weibo", "reddit", "amazon", "yelpchi", "blogcatalog", "facebook", "acm",
            "elliptic", "elliptic_plus_plus", "dgraph", "t_finance"]
DISPLAY = {"weibo": "Weibo", "reddit": "Reddit", "amazon": "Amazon", "yelpchi": "YelpChi",
           "blogcatalog": "BlogCatalog", "facebook": "Facebook", "acm": "ACM", "elliptic": "Elliptic",
           "elliptic_plus_plus": "Elliptic++", "dgraph": "DGraph", "t_finance": "T-Finance"}
SHORT = {"blogcatalog": "BlogCat.", "elliptic_plus_plus": "Ell.++", "t_finance": "T-Fin."}
METHODS = ["LOF", "DIF", "ANOMALOUS", "DOMINANT", "AnomalyDAE", "CONAD", "CoLA", "DiffGAD", "TAM"]
HEADER = {"ANOMALOUS": "ANOM.", "AnomalyDAE": "AnomDAE"}
# Cells quoted from the TAM benchmark (Qiao and Pang, 2023); all other baseline cells are our runs,
# except the three cells quoted from the baselines' own reports (out of memory on our hardware).
TAM_BENCH = {(m, d) for m in ("ANOMALOUS", "DOMINANT", "CoLA", "TAM")
             for d in ("reddit", "amazon", "yelpchi", "blogcatalog", "facebook", "acm")}
TAM_BENCH -= {("ANOMALOUS", "reddit"), ("DOMINANT", "reddit")}
REPORTED = {("CONAD", "dgraph"), ("DiffGAD", "dgraph"), ("TAM", "t_finance")}


def short(ds):
    return SHORT.get(ds, DISPLAY[ds])


def load_baselines():
    """Baseline numbers and their sources (camera_ready/baselines.json)."""
    b = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines.json")))
    base = {ds: {m: (None if c is None else (c[0], c[1])) for m, c in row.items()} for ds, row in b["auroc"].items()}
    auprc = {m: {ds: (c[0], c[1]) for ds, c in row.items()} for m, row in b["auprc"].items()}
    return base, auprc, b["encoder"], b["runtime"]


def rank_marks(values):
    """values: dict key -> float or None. Returns (best keys, second keys) on values rounded to 0.1."""
    v = {k: round(x, 1) for k, x in values.items() if x is not None}
    top = sorted(set(v.values()), reverse=True)
    best = {k for k, x in v.items() if x == top[0]}
    second = {k for k, x in v.items() if len(top) > 1 and x == top[1]} if len(best) == 1 else set()
    return best, second


def sci(x):
    """x as $m{\\times}10^{e}$ with two decimals."""
    m, e = ("%.2e" % x).split("e")
    return "$%s{\\times}10^{%d}$" % (m, int(e))


def fmt(mean, std=None, mark=""):
    core = "%.1f" % mean
    if mark == "b":
        core = "\\mathbf{%s}" % core
    elif mark == "u":
        core = "\\underline{%s}" % core
    if std is not None:
        return "$%s\\,{\\scriptstyle\\pm\\,%.1f}$" % (core, std)
    return "$%s$" % core


def main():
    os.makedirs(OUT, exist_ok=True)
    t = json.load(open(os.path.join(RES, "table1_ebgad.json")))
    base, auprc, enc_base, runtime_base = load_baselines()
    numbers = {"table1": {}, "auprc": {}, "standing": {}}

    # ------------------------------------------------------------------ Table 1
    lines = []
    standing = {}
    for ds in DATASETS:
        a = t[ds]["aggregate"]
        eb_mean = a["selected_auroc_mean"]
        eb_std = a["selected_auroc_std"] if t[ds]["truncated"] and t[ds]["n_orders"] > 1 else None
        vals = {m: (base[ds][m][0] if base[ds][m] else None) for m in METHODS}
        vals["EB-GAD"] = eb_mean
        best, second = rank_marks(vals)
        order = sorted([x for x in vals.values() if x is not None], reverse=True)
        standing[ds] = 1 + sum(1 for x in order if round(x, 1) > round(eb_mean, 1))
        cells = []
        for m in METHODS:
            c = base[ds][m]
            if c is None:
                cells.append("\\oom")
                continue
            mark = "b" if m in best else ("u" if m in second else "")
            s = fmt(c[0], c[1], mark)
            if (m, ds) in TAM_BENCH:
                s += "$^{*}$"
            if (m, ds) in REPORTED:
                s += "$^{\\dagger}$"
            cells.append(s)
        mark = "b" if "EB-GAD" in best else ("u" if "EB-GAD" in second else "")
        cells.append(fmt(eb_mean, eb_std, mark))
        numbers["table1"][ds] = {"ebgad": round(eb_mean, 1), "ebgad_std": None if eb_std is None else round(eb_std, 1),
                                 "rank": standing[ds]}
        numbers.setdefault("best_baseline", {})[ds] = max(v for k, v in vals.items() if v is not None and k != "EB-GAD")
        numbers.setdefault("truncated", {})[ds] = bool(t[ds]["truncated"])
        lines.append("    %s\n      & %s \\\\" % (DISPLAY[ds], " & ".join(cells)))
    head = " & ".join(HEADER.get(m, m) for m in METHODS)
    tex = """\\begin{table}[t]
  \\centering
  \\caption{%%
    AUROC (\\%%) on 11 GAD benchmarks. \\textbf{Bold}: best; \\underline{underline}: second; \\oom{} = out of memory. Baselines use their native anomaly-score direction (Appendix Table~\\ref{tab:score_orientation_audit}). Cell sources: $^{*}$ published in the TAM benchmark \\cite{qiao2023truncated}; $^{\\dagger}$ published by the baseline's authors (out of memory on our hardware); unmarked baseline cells are our runs with published hyperparameters. For truncated spectra (YelpChi, Elliptic, Elliptic++, DGraph, T-Finance) EB-GAD is the mean $\\pm$ standard deviation over five node orders (Appendix~\\ref{app:regeneration}).
  }
  \\label{tab:main_auroc}
  \\scriptsize
  \\setlength{\\tabcolsep}{2.4pt}
  \\renewcommand{\\arraystretch}{0.9}
  \\resizebox{\\textwidth}{!}{%%
  \\begin{tabular}{l cccccccccc}
    \\toprule
    Dataset
      & %s
      & \\textbf{EB-GAD} \\\\
    \\midrule
%s
    \\bottomrule
  \\end{tabular}
  }
\\end{table}
""" % (head, "\n".join(lines))
    open(os.path.join(OUT, "table_main_auroc.tex"), "w").write(tex)
    numbers["standing"] = standing

    # ------------------------------------------------------------------ AUPRC
    lines = []
    col_marks = {}
    for ds in DATASETS:
        vals = {m: auprc[m][ds][0] for m in METHODS}
        vals["EB-GAD"] = t[ds]["aggregate"]["selected_auprc_mean"]
        col_marks[ds] = rank_marks(vals)
        others = [round(v, 1) for k, v in vals.items() if v is not None and k != "EB-GAD"]
        numbers.setdefault("auprc_rank", {})[ds] = 1 + sum(1 for v in others if v > round(vals["EB-GAD"], 1))
        numbers.setdefault("auprc_best_baseline", {})[ds] = max(others)
    for m in METHODS + ["EB-GAD"]:
        cells = []
        for ds in DATASETS:
            best, second = col_marks[ds]
            mark = "b" if m in best else ("u" if m in second else "")
            if m == "EB-GAD":
                v = t[ds]["aggregate"]["selected_auprc_mean"]
                cells.append(fmt(v, None, mark))
                numbers["auprc"][ds] = round(v, 1)
            else:
                v, flag = auprc[m][ds]
                if v is None:
                    cells.append("\\oom" if flag == "OOM" else "n/a")
                else:
                    cells.append(fmt(v, None, mark) + ("$^{*}$" if flag == "*" else ""))
        name = "\\textbf{EB-GAD}" if m == "EB-GAD" else m
        if m == "EB-GAD":
            lines.append("    \\midrule")
        lines.append("    %s & %s \\\\" % (name, " & ".join(cells)))
    tex = """\\begin{table}[!htb]
  \\centering
  \\caption{%%
    AUPRC (\\%%) for the comparison of Table~\\ref{tab:main_auroc}. \\textbf{Bold}: best; \\underline{underline}: second. $^{*}$: published in the TAM benchmark \\cite{qiao2023truncated}, used exactly where Table~\\ref{tab:main_auroc} follows that benchmark; other baseline cells are our runs. \\oom{} = out of memory; n/a = out of memory and the published report gives AUROC only. EB-GAD cells are the AUPRC of the score selected without labels in Table~\\ref{tab:main_auroc}.
  }
  \\label{tab:auprc}
  \\scriptsize
  \\setlength{\\tabcolsep}{3pt}
  \\resizebox{\\textwidth}{!}{%%
  \\begin{tabular}{l ccccccccccc}
    \\toprule
    Method & %s \\\\
    \\midrule
%s
    \\bottomrule
  \\end{tabular}
  }
\\end{table}
""" % (" & ".join(short(d) for d in DATASETS), "\n".join(lines))
    open(os.path.join(OUT, "table_auprc.tex"), "w").write(tex)

    # ------------------------------------------------------------------ selector configuration
    fam_name = {("anchor", "J*"): "equilibrium anchor ($J^\\star$)", ("anchor", "R"): "equilibrium anchor ($R$)",
                ("control_energy", "low_pass"): "control energy (top-2 $C_{\\Gamma,\\lambda_c}$ fusion)",
                ("control_energy", "affinity"): "control energy (stability-ranked $C_{\\Gamma,\\lambda_c}$ fusion)",
                ("control_ratio", "affinity"): "control ratio (stability-ranked $CR_{\\Gamma,\\lambda_c}$ fusion)",
                ("profile", "scale_entropy"): "profile aggregation (scale-entropy)",
                ("profile", "sparse_tail"): "profile aggregation (sparse-tail)",
                ("profile", "trajectory"): "profile aggregation (trajectory)"}
    lines, lines_stats, lines_eq = [], [], []
    numbers["config"] = {}
    for ds in DATASETS:
        f = t[ds]["first"]
        c, st, d = f["config"], f["stats"], f["diagnostics"]
        tmpl = "affinity" if c["template"] == "affinity" else "low-pass"
        k = "all" if st["k"] is None else str(st["k"])
        pca = "no" if c["pca"] in (None, "None") else str(c["pca"])
        lines.append("    %s & %s & %s & %.3f & %s & %s & %s & %d & %s \\\\" % (
            DISPLAY[ds], tmpl, k, c["rho"], ("%g" % c["kappa"]), ("%g" % c["gamma"]), pca, f["rule"],
            fam_name[(f["family"], f["branch"])]))
        kn = "all" if st["k"] is None else sci(st["k"] / st["n"])
        lines_stats.append("    %s & %.3f & %.2f & %d & %s & %.3f & %.3f & %.1f & %.3f & %.3f \\\\" % (
            DISPLAY[ds], st["h"], st["d_e"], st["D_x"], kn, d["ks_j"], d["ks_r"], d["profile_bj"], d["profile_ks"], d["pi"]))
        numbers["config"][ds] = {"rule": f["rule"], "family": f["family"], "branch": f["branch"], "score": f["score"],
                                 "rho": c["rho"], "kappa": c["kappa"], "gamma": c["gamma"], "pca": c["pca"], "k": st["k"]}
    tex = """\\begin{table*}[!htb]
  \\centering
  \\caption{%%
    Label-free EB-GAD configuration behind each cell of Table~\\ref{tab:main_auroc}.
    \\emph{Template}: low-pass Mat\\'ern or feature-affinity-weighted (\\S\\ref{app:selector_details}).
    $k$: number of retained low-frequency Laplacian eigenpairs (``all'' = full spectrum).
    $\\rho,\\kappa,\\gamma$: EB-fitted graph trust and Mat\\'ern inverse length-scale, and the template bandwidth of the selected score (the anchor bandwidth $\\gamma_a$ for an anchor, the likelihood-selected bandwidth for a profile, the pool rule for control energy and ratio).
    \\emph{PCA}: retained PCA dimensions. \\emph{Rule}: the ordered rule of \\S\\ref{app:selector_details} that fires.
  }
  \\label{tab:ebgad_hyperparams}
  \\scriptsize
  \\setlength{\\tabcolsep}{3pt}
  \\renewcommand{\\arraystretch}{1.05}
  \\resizebox{\\textwidth}{!}{%%
  \\begin{tabular}{l c c c c c c c l}
    \\toprule
    Dataset & Template & $k$ & $\\rho$ & $\\kappa$ & $\\gamma$ & PCA & Rule & Selected family (score) \\\\
    \\midrule
%s
    \\bottomrule
  \\end{tabular}
  }
\\end{table*}
""" % "\n".join(lines)
    open(os.path.join(OUT, "table_selector_config.tex"), "w").write(tex)
    tex = """\\begin{table*}[!htb]
\\centering
\\scriptsize
\\caption{Unlabeled quantities read by the selector. $h$: cosine feature homophily over a fixed sample of edges, $d_e$: edges per node of the graph operator, $D_x$: raw feature dimension, $k/n$: truncation ratio. Diagnostics of the EB-fitted run: NullKS of its two equilibrium scores, Berk--Jones statistic (BJ) and NullKS of the profile score, and the two-groups mixture mass $\\pi$.}
\\label{tab:selector_graph_stats}
\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{l c c c c c c c c c}
\\toprule
Dataset & $h$ & $d_e$ & $D_x$ & $k/n$ & NullKS($J^\\star$) & NullKS($R$) & BJ & NullKS(profile) & $\\pi$ \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table*}
""" % "\n".join(lines_stats)
    open(os.path.join(OUT, "table_selector_stats.tex"), "w").write(tex)

    # ------------------------------------------------------------------ equilibrium scores
    rowJ, rowR, rowG, rowSel = [], [], [], []
    numbers["equilibrium"] = {}
    for ds in DATASETS:
        a = t[ds]["aggregate"]
        f = t[ds]["first"]
        pick = f["equilibrium"]["score"]
        j, r = a["J*_auroc_mean"], a["R_auroc_mean"]
        rowG.append("%g" % f["anchor"]["gamma"])
        rowJ.append(("\\textbf{%.1f}" if pick == "J*" else "%.1f") % j)
        rowR.append(("\\textbf{%.1f}" if pick == "R" else "%.1f") % r)
        rowSel.append("%.1f" % a["selected_auroc_mean"])
        numbers["equilibrium"][ds] = {"J": round(j, 1), "R": round(r, 1), "pick": pick,
                                      "eq": round(a["equilibrium_auroc_mean"], 1),
                                      "selected": round(a["selected_auroc_mean"], 1),
                                      "gamma": f["anchor"]["gamma"], "gamma_fit": f["anchor"]["gamma_fit"],
                                      "J_fit": round(a["J*_fit_auroc_mean"], 1), "R_fit": round(a["R_fit_auroc_mean"], 1)}
    tex = """\\begin{table}[!htb]
\\centering
\\scriptsize
\\caption{Equilibrium anchor: AUROC (\\%%) of $J^\\star$ and $R$ at the anchor bandwidth $\\gamma_a$ (the grid value nearest to $\\gamma_0$) with the EB fit of $(\\rho,\\kappa)$ at that bandwidth. \\textbf{Bold}: the score chosen without labels (larger NullKS; ties go to $J^\\star$; rule 1 of \\S\\ref{app:selector_details} fixes $R$ on sparse truncated graphs). Last row: the score selected in Table~\\ref{tab:main_auroc}.}
\\label{tab:ks_ablation}
\\setlength{\\tabcolsep}{2pt}
\\begin{tabular}{l ccccccccccc}
\\toprule
& %s \\\\
\\midrule
$\\gamma_a$ & %s \\\\
$J^\\star$ & %s \\\\
$R$       & %s \\\\
\\midrule
Table~\\ref{tab:main_auroc} & %s \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""" % (" & ".join(short(d) for d in DATASETS), " & ".join(rowG), " & ".join(rowJ), " & ".join(rowR), " & ".join(rowSel))
    open(os.path.join(OUT, "table_equilibrium.tex"), "w").write(tex)

    # ------------------------------------------------------------------ finite horizon at a matched configuration
    rowE, rowS, rowD, rowF = [], [], [], []
    numbers["matched"] = {}
    fam_short = {"anchor": "anchor", "control_energy": "energy", "control_ratio": "ratio", "profile": "profile"}
    for ds in DATASETS:
        a = t[ds]["aggregate"]
        e, sel = a["equilibrium_matched_auroc_mean"], a["selected_auroc_mean"]
        dlt = round(sel, 1) - round(e, 1)
        rowE.append((r"\textbf{%.1f}" if round(e, 1) > round(sel, 1) else "%.1f") % e)
        rowS.append((r"\textbf{%.1f}" if round(sel, 1) > round(e, 1) else "%.1f") % sel)
        rowD.append("$%s$" % ("0.0" if abs(dlt) < 0.05 else "%+.1f" % dlt))
        rowF.append(fam_short[t[ds]["first"]["family"]])
        numbers["matched"][ds] = {"equilibrium": round(e, 1), "selected": round(sel, 1), "delta": round(dlt, 1),
                                  "family": t[ds]["first"]["family"]}
    tex = r"""\begin{table}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{2.2pt}
  \caption{Contribution of the finite horizon: AUROC (\%%) of the selected score against the equilibrium score ($J^\star$ or $R$, by NullKS) of the \emph{same} configuration (operator, template, bandwidth, PCA, truncation, EB fit). \emph{Family}: the family the selector routes to; for an anchor the two rows coincide.}
  \label{tab:bank_ablation}
  \resizebox{\textwidth}{!}{%%
  \begin{tabular}{l ccccccccccc}
    \toprule
    & %s \\
    \midrule
    Family & %s \\
    Equilibrium & %s \\
    Selected & %s \\
    $\Delta$ & %s \\
    \bottomrule
  \end{tabular}
  }
\end{table}
""" % (" & ".join(short(d) for d in DATASETS), " & ".join(rowF), " & ".join(rowE), " & ".join(rowS), " & ".join(rowD))
    open(os.path.join(OUT, "table_bank_ablation.tex"), "w").write(tex)

    # ------------------------------------------------------------------ trained encoders
    enc = {ds: (v[0], v[1]) for ds, v in enc_base.items()}
    assert set(enc) == set(DATASETS), sorted(enc)
    lines, wins = [], []
    numbers["encoder"] = {}
    for ds in DATASETS:
        eb = round(t[ds]["aggregate"]["selected_auroc_mean"], 1)
        tr, kind = enc[ds]
        gap = round(eb - tr, 1)
        if gap > 0:
            wins.append(ds)
        lines.append("%s & %s & %s (%s) & $%+.1f$ \\\\" % (
            DISPLAY[ds], (r"\textbf{%.1f}" if eb > tr else "%.1f") % eb,
            (r"\textbf{%.1f}" if tr > eb else "%.1f") % tr, kind, gap))
        numbers["encoder"][ds] = {"training_free": eb, "trained": tr, "kind": kind, "gap": gap}
    numbers["encoder_wins"] = wins
    tex = r"""\begin{table}[!htb]
\centering
\caption{Training-free EB-GAD (Table~\ref{tab:main_auroc}) against the better of two trained encoders (MLP, GraphSAGE), chosen per dataset by AUROC, followed by the same score formula. AUROC (\%%); gap = training-free minus trained.}
\label{tab:encoder_comparison}
\scriptsize
\begin{tabular}{lccc}
\toprule
Dataset & Training-free & Best trained encoder & Gap \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines)
    open(os.path.join(OUT, "table_encoder.tex"), "w").write(tex)
    print("training-free ahead of trained encoders on", len(wins), "of 11:", wins)

    # ------------------------------------------------------------------ runtime
    tfile = os.path.join(RES, "timing", "timing.jsonl")
    if os.path.exists(tfile):
        tim = {}
        for line in open(tfile):
            x = json.loads(line)
            assert x["exit"] == 0, x
            tim[x["dataset"]] = x
        cols = ["ANOMALOUS", "DOMINANT", "AnomalyDAE", "CONAD", "CoLA", "DiffGAD", "TAM"]
        lines = []
        numbers["runtime"] = {}
        for ds in DATASETS:
            base_t = [runtime_base[ds][c] for c in cols]
            sec = tim[ds]["seconds"] + (tim["dgraph_eigen"]["seconds"] if ds == "dgraph" else 0.0)
            numbers["runtime"][ds] = {"ebgad_seconds": round(sec), "baselines": dict(zip(cols, base_t)),
                                      "threads": tim[ds]["threads"]}
            lines.append("    %s & %s & %s \\\\" % (DISPLAY[ds], " & ".join(
                "\\oom" if b is None else "{:,.0f}".format(b) for b in base_t), "{:,.0f}".format(sec)))
        tex = r"""\begin{table*}[!htb]
  \centering
  \caption{%%
    Wall-clock time (seconds). Baselines: training and inference on one NVIDIA A100 GPU, as measured for the submitted version. EB-GAD: the complete run behind its cell of Table~\ref{tab:main_auroc} on %d CPU threads, from loading the data to the selected score: eigendecompositions in double precision for every template bandwidth of the grid, EB fits, all score families of the run and the label-free selection. The two columns use different hardware and are not a controlled comparison. \oom{} = out of memory.
  }
  \label{tab:runtime}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.05}
  \begin{tabular}{l rrrrrrr r}
    \toprule
    Dataset & ANOM. & DOMINANT & AnomDAE & CONAD & CoLA & DiffGAD & TAM & EB-GAD (CPU) \\
    \midrule
%s
    \bottomrule
  \end{tabular}
\end{table*}
""" % (tim["facebook"]["threads"], "\n".join(lines))
        open(os.path.join(OUT, "table_runtime.tex"), "w").write(tex)
    else:
        print("timing not available yet")

    # ------------------------------------------------------------------ relabeling gate
    lines = []
    numbers["gate"] = {}
    for ds in DATASETS:
        v = t[ds]
        a = v["aggregate"]
        g = v["gate"] or {}
        rho = g.get("spearman")
        vals = a["selected_auroc_all"]
        lines.append("    %s & %s & %d & %.2f & %.2f & %.2f & %s \\\\" % (
            DISPLAY[ds], "truncated" if v["truncated"] else "full", v["n_orders"], min(vals), max(vals),
            a["selected_auroc_std"], "--" if rho is None else ("%.4f" % rho)))
        numbers["gate"][ds] = {"min": round(min(vals), 2), "max": round(max(vals), 2), "spearman": rho, "n": v["n_orders"]}
    tex = """\\begin{table}[!htb]
\\centering
\\scriptsize
\\caption{Relabeling gate for the scores of Table~\\ref{tab:main_auroc}. The model is equivariant to node relabeling, so the selected score must not depend on the node order. \\emph{Orders}: number of node orders run (the original and random relabelings). AUROC (\\%%): smallest, largest and standard deviation over the orders. Spearman: rank correlation of the selected score between the original order and one relabeling.}
\\label{tab:relabel_gate}
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular}{l c c c c c c}
\\toprule
Dataset & Spectrum & Orders & min & max & std & Spearman \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}
""" % "\n".join(lines)
    open(os.path.join(OUT, "table_relabel_gate.tex"), "w").write(tex)

    json.dump(numbers, open(os.path.join(RES, "paper_numbers.json"), "w"), indent=1)
    print("auprc rank:", numbers["auprc_rank"])
    print("auprc best baseline:", numbers["auprc_best_baseline"])
    print("auprc ebgad:", numbers["auprc"])
    print("standing:", standing)
    print("best:", [d for d, r in standing.items() if r == 1], "second:", [d for d, r in standing.items() if r == 2])
    for ds in DATASETS:
        print("%-20s %.1f  rank %d" % (ds, numbers["table1"][ds]["ebgad"], standing[ds]))


if __name__ == "__main__":
    main()
