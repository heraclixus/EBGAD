#!/usr/bin/env python
"""EB-GAD entry point.

load -> preprocess -> spectrum -> template (per gamma) -> EB fit -> pool of
statistics -> exact per-node p-values -> label-free selection -> report.

Labels are used only for reporting AUROC/AUPRC. Every reported score is
dumped to <out>/<dataset>_scores.npz. ``--permute-check`` reruns the whole
pipeline on a randomly relabeled graph and reports the rank agreement of
every candidate (the gate that would have caught the YelpChi index leak).

Example:
    python run_ebgad.py --dataset weibo --pca 64 --gammas 0.5 --null-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from data_utils import load_data
from scipy.stats import rankdata

from ebgad.fit import PAPER_KAPPAS, fit_horizon, fit_prepared, select_gamma
from ebgad.nulls import member_tails, null_check
from ebgad.prep import permute_data, prepare
from ebgad.scores import (PAPER_HORIZONS, PAPER_LAMS, bank_configs, bank_specs,
                          compute_members, conditional_residual, node_tau2,
                          normalized_conditional_residual, profile_specs)
from ebgad.select import scan_score, select
from ebgad.graphstats import compute_graph_stats, density_adjusted_template_gammas


def parse_list(text, cast=float):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(np.inf if tok.lower() == "inf" else cast(tok))
    return out


def eval_mask(y: np.ndarray) -> np.ndarray:
    return (y == 0) | (y == 1)


def metrics(y, score, mask):
    s = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if s[mask].max() <= s[mask].min():
        return float("nan"), float("nan")
    return (float(roc_auc_score(y[mask], s[mask])) * 100.0,
            float(average_precision_score(y[mask], s[mask])) * 100.0)


UNIT_GAMMAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]


def build_specs(args, q):
    """Statistic pool: J*/R always; dissipation profile and/or the paper's (Gamma, lam_c) bank."""
    specs = bank_specs(q, [(np.inf, np.inf)])          # J*, R
    if args.bank in ("profile", "both"):
        specs += profile_specs(q)
    if args.bank in ("paper", "both"):
        cfgs = [c for c in bank_configs(parse_list(args.horizons), parse_list(args.lams))
                if not (np.isinf(c[0]) and np.isinf(c[1]))]
        specs += bank_specs(q, cfgs)
    return specs


def dedupe_members(members, rng, n_sample=20000, tol=0.999):
    """Group rank-identical statistics; keep the first of each group.

    Preference order: J*, R, P first, then bank order. Returns (kept, groups).
    """
    n = members[0].score.shape[0]
    sub = rng.choice(n, size=min(n, n_sample), replace=False) if n > n_sample else np.arange(n)
    named = [m for m in members if m.name in ("J*", "R", "P", "NP")]
    rest = [m for m in members if m.name not in ("J*", "R", "P", "NP")]
    kept, ranks, groups = [], [], {}
    for m in named + rest:
        r = rankdata(np.nan_to_num(m.score[sub], nan=0.0, posinf=0.0, neginf=0.0))
        r = (r - r.mean()) / max(r.std(), 1e-12)
        dup = None
        for km, kr in zip(kept, ranks):
            if float(np.mean(r * kr)) >= tol:
                dup = km.name
                break
        if dup is None:
            kept.append(m)
            ranks.append(r)
            groups[m.name] = [m.name]
        else:
            groups[dup].append(m.name)
    return kept, groups


def run_once(name, data, args, spectrum_cache, verbose=True):
    """Full label-free pipeline. Returns (result dict, {member name: score}, {name: pvalues})."""
    t0 = time.time()
    unit = args.template.endswith("_unit")
    if args.gammas:
        gammas = parse_list(args.gammas)
    elif unit:
        gammas = list(UNIT_GAMMAS)
    else:
        h, dens = compute_graph_stats(data)
        gammas = density_adjusted_template_gammas(h, dens)
    use_jac = {"auto": unit, "on": True, "off": False}[args.jacobian]
    k_arg = None if args.k in (None, "full") else int(args.k)
    force_full = args.k == "full"
    fits = []
    best_prep, best_key = None, -np.inf
    for g in gammas:
        prep = prepare(name, data, g, template=args.template, graph=args.graph,
                       pca=args.pca, k=k_arg, subspace=args.subspace, seed=args.seed,
                       cache_dir=args.cache_dir, pca_order=args.pca_order,
                       _spectrum_cache=spectrum_cache, force_full=force_full)
        fit = fit_prepared(prep, kappas=parse_list(args.kappas))
        fits.append(fit)
        key = fit.ml_jac if use_jac else fit.ml
        if key > best_key:      # keep only the winning residual field (ACM: ~7 GB per gamma)
            best_key, best_prep = key, prep
        del prep
        if verbose:
            print(f"  gamma={g:g}: rho={fit.rho:.3f} kappa={fit.kappa:g} ml={fit.ml:.1f} "
                  f"ml_jac={fit.ml_jac:.1f}  [k={best_prep.k}, d={best_prep.d}]", flush=True)
    fit = select_gamma(fits, use_jacobian=use_jac)
    prep = best_prep
    q = fit.q

    horizon = None
    if args.fit_horizon:
        horizon = fit_horizon(prep, kappas=parse_list(args.kappas))
        if verbose:
            fb, fi = horizon["finite"], horizon["inf"]
            print(f"  EB horizon: best finite Gamma={fb['Gamma']:g} (rho={fb['rho']:.3f}, kappa={fb['kappa']:g}, "
                  f"ml={fb['ml']:.1f}) vs inf (rho={fi['rho']:.3f}, kappa={fi['kappa']:g}, ml={fi['ml']:.1f}); "
                  f"gain={horizon['gain']:+.1f}", flush=True)

    specs = build_specs(args, q)
    members = compute_members(prep, q, specs, start=args.start, rough=args.rough)
    if not args.no_cond:
        members.append(conditional_residual(prep, fit.rho, fit.kappa))
    if args.np:
        members.append(normalized_conditional_residual(prep, fit.rho, fit.kappa, node_tau2(prep, q, args.rough)))
    n_all = len(members)
    members, groups = dedupe_members(members, np.random.default_rng(args.seed))
    if verbose:
        print(f"  pool: {n_all} statistics, {len(members)} distinct (rank correlation < 0.999)", flush=True)

    scores, pvals = {}, {}
    for m in members:
        scores[m.name] = m.score
        entry = {"stat": m.score, "null_mean": m.null_mean(prep.d)}
        try:
            entry["sf"], entry["cdf"] = member_tails(m, prep.d)
        except NotImplementedError:
            pass
        pvals[m.name] = entry
    selected, cands = select(pvals)
    # Selection-free scan scores over the pool and over each channel.
    energy_names = [n for n in scores if n in ("J*", "P") or n.startswith(("C@", "D@"))]
    ratio_names = [n for n in scores if n in ("R", "NP") or n.startswith(("CR@", "DR@"))]
    for tag, names in (("SCAN", None), ("SCAN-E", energy_names), ("SCAN-R", ratio_names)):
        s = scan_score(cands, names)
        if s is not None:
            scores[tag] = s
    s3 = scan_score(cands, None, top_k=3)
    if s3 is not None:
        scores["SCAN-top3"] = s3
    hier_diag = None
    tg_diag = None
    if args.tg:
        from ebgad.twogroups import two_groups_channel
        from ebgad.select import z_from_tails
        ratio_z, energy_z = {}, {}
        for m in members:
            if m.kind == "ratio":
                try:
                    sf, cdf = member_tails(m, prep.d)
                except NotImplementedError:
                    continue
                ratio_z[m.name] = z_from_tails(sf, cdf)
            elif m.kind in ("energy", "cond"):
                # energy channel (analysis only): log(stat / null mean), calibrated per member
                energy_z[m.name] = np.log(np.maximum(m.score, 1e-300) / np.maximum(m.null_mean(prep.d), 1e-300))
        tg_diag = {}
        for tag, zs in (("TG-R", ratio_z), ("TG-E", energy_z)):
            fit_tg = two_groups_channel(zs)
            if fit_tg is None:
                continue
            scores[tag] = fit_tg["log_bayes_factor"]      # monotone in the posterior at fixed pi
            scores[tag + "-post"] = fit_tg["posterior"]
            tg_diag[tag] = {"pi": fit_tg["pi"], "members": fit_tg["members"], "w": fit_tg["member_weights"],
                            "a": fit_tg["member_a"], "iterations": fit_tg["iterations"]}
            if verbose:
                top = sorted(fit_tg["member_weights"].items(), key=lambda kv: -kv[1])[:3]
                print(f"  {tag}: pi={fit_tg['pi']:.3f} members={len(fit_tg['members'])} top weights "
                      + ", ".join("%s %.2f (a=%.2f)" % (k_, v, fit_tg["member_a"][k_]) for k_, v in top), flush=True)
    surprise_diag = None
    if args.surprise:
        from ebgad.surprise import surprise_scores
        emask = eval_mask(data.y.cpu().numpy().astype(int)) if args.null_population == "eval" else None
        sp = surprise_scores(prep, q, rough=args.rough, n_bands=args.bands, eval_mask=emask)
        scores.update(sp["scores"])
        surprise_diag = sp["diag"]
        if verbose:
            dg = surprise_diag
            print(f"  surprise: bands={dg['n_bands']} d_eff={dg['d_eff']:.1f} of {prep.d}  mu_s={dg['mu_s']:+.2f} sigma_s={dg['sigma_s']:.2f} "
                  f"null on {'evaluated nodes' if emask is not None else 'all nodes'}", flush=True)
    cam_diag = None
    if args.cam or args.hier:
        from ebgad.hier import camouflage_scores
        emask = eval_mask(data.y.cpu().numpy().astype(int)) if args.null_population == "eval" else None
        c = camouflage_scores(prep, q, rough=args.rough, eval_mask=emask)
        scores.update(c["scores"])
        cam_diag = c["diag"]
        if verbose:
            print(f"  camouflage: scale prior mu_s={cam_diag['mu_s']:+.2f} sigma_s={cam_diag['sigma_s']:.2f} on {cam_diag['null_population']} nodes", flush=True)
    if args.hier:
        from ebgad.hier import hierarchical_scores
        h = hierarchical_scores(prep, q, members, rough=args.rough)
        scores.update(h["scores"])
        hier_diag = h["diag"]
        if verbose:
            dg = hier_diag
            print(f"  hier: sigma_s={dg['sigma_s']:.2f} bulk z_mag=({dg['bulk_z_mag'][0]:+.2f},{dg['bulk_z_mag'][1]:.2f}) "
                  f"bulk z_J*=({dg['bulk_z_energy_Jstar'][0]:+.2f},{dg['bulk_z_energy_Jstar'][1]:.2f}) "
                  f"bulk z_R=({dg['bulk_z_ratio_R'][0]:+.2f},{dg['bulk_z_ratio_R'][1]:.2f}) "
                  f"pi: mag {dg['pi_mag']:.3f} E {dg['pi_E_best']:.3f} R {dg['pi_R_best']:.3f} -> channel {dg['channel']}", flush=True)
    res = {
        "dataset": name, "n": prep.n, "d": prep.d, "k": prep.k,
        "gamma": fit.gamma, "rho": fit.rho, "kappa": fit.kappa, "ml": fit.ml, "ml_jac": fit.ml_jac,
        "gammas": [float(g) for g in gammas], "jacobian": use_jac,
        "gamma_profile": [f.as_dict() for f in fits],
        "template": args.template, "graph": args.graph, "pca": args.pca,
        "subspace": args.subspace, "start": args.start, "rough": args.rough, "bank": args.bank,
        "horizon_fit": horizon,
        "hier": hier_diag,
        "two_groups": tg_diag,
        "surprise": surprise_diag,
        "camouflage": cam_diag,
        "pool_groups": groups,
        "selected": selected,
        "candidates": {n: c.summary() for n, c in cands.items()},
        "seconds": time.time() - t0,
    }
    return res, scores, pvals, prep, q, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gammas", default="", help="comma list; default = paper's h/d_e grid")
    ap.add_argument("--kappas", default=",".join("%g" % k for k in PAPER_KAPPAS))
    ap.add_argument("--horizons", default=",".join("inf" if np.isinf(h) else "%g" % h for h in PAPER_HORIZONS))
    ap.add_argument("--lams", default=",".join("inf" if np.isinf(l) else "%g" % l for l in PAPER_LAMS))
    ap.add_argument("--template", default="low", choices=["low", "affinity", "low_unit", "affinity_unit"],
                    help="low/affinity = paper's unnormalized Matern template; *_unit = unit-DC-gain template")
    ap.add_argument("--graph", default="original", choices=["original", "affinity"])
    ap.add_argument("--pca", type=int, default=None)
    ap.add_argument("--pca-order", default="code", choices=["code", "paper"])
    ap.add_argument("--k", default=None, help="truncation k; default = paper auto rule; 'full' forces the full spectrum "
                    "(block-diagonal per connected component when components are small, e.g. YelpChi's 7,308 components)")
    ap.add_argument("--subspace", default="drop_nullspace", choices=["legacy", "drop_nullspace"])
    ap.add_argument("--start", default="template", choices=["template", "origin"])
    ap.add_argument("--rough", default="isotropic", choices=["isotropic", "drop"])
    ap.add_argument("--jacobian", default="auto", choices=["auto", "on", "off"],
                    help="select gamma by the x-likelihood (with log|det T|); auto = on for *_unit templates")
    ap.add_argument("--fit-horizon", action="store_true", help="joint EB over (rho, kappa, Gamma); reports the gain over Gamma = inf")
    ap.add_argument("--labeled-subgraph", action="store_true",
                    help="restrict to labeled nodes (y in {0,1}) and edges among them before everything (DGraph paper protocol; the default is the full graph)")
    ap.add_argument("--hier", action="store_true", help="hierarchical node-scale prior: MAG, HSCAN-E/R, H-SUM, H-CHAN scores (ebgad/hier.py)")
    ap.add_argument("--tg", action="store_true", help="two-groups EB aggregation over the shape channel (TG-R) and, for analysis, the energy channel (TG-E)")
    ap.add_argument("--surprise", action="store_true", help="v3 score: marginal surprise of the relaxation profile under the hierarchical GOU (ebgad/surprise.py)")
    ap.add_argument("--cam", action="store_true", help="camouflage surprise CAM (lower tail of the template displacement), plus DEV (upper) and TWO-SIDED")
    ap.add_argument("--bands", type=int, default=4, help="number of relaxation-rate bands for the surprise score")
    ap.add_argument("--null-population", default="eval", choices=["eval", "all"],
                    help="fit the hierarchical null on the evaluated (labeled) nodes or on all nodes (label values are never used)")
    ap.add_argument("--bank", default="profile", choices=["profile", "paper", "both"],
                    help="statistic pool besides J*, R, P: dissipation-rate profile (default), the paper's (Gamma, lam_c) bank, or both")
    ap.add_argument("--no-cond", action="store_true", help="exclude the conditional residual P")
    ap.add_argument("--np", action="store_true", help="add NP, the exact full-graph t = 0 shape member (opt-in: it lowered the dev-pair shape scan, 2026-09-18)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-check", type=int, default=0, metavar="B",
                    help="Monte Carlo self-check of the closed-form nulls: B field draws on a 1000-node subset")
    ap.add_argument("--permute-check", action="store_true")
    ap.add_argument("--cache-dir", default="cache/ebgad")
    ap.add_argument("--out", default="results/ebgad_v2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    name = args.dataset
    data = load_data(name)
    if args.labeled_subgraph:
        from data_utils import extract_labeled_subgraph
        n_full = data.num_nodes
        data, _mask = extract_labeled_subgraph(data)
        name = name + "_labeled"
        print(f"[{args.dataset}] labeled subgraph: {data.num_nodes} of {n_full} nodes, {data.edge_index.shape[1]} directed edges", flush=True)
    y = data.y.cpu().numpy().astype(int)
    mask = eval_mask(y)
    print(f"[{name}] n={data.num_nodes} d={data.x.shape[1]} anomalies={int((y==1).sum())} "
          f"(evaluated on {int(mask.sum())} labeled nodes)", flush=True)

    spectrum_cache = {}
    res, scores, pvals, prep, q, fit = run_once(name, data, args, spectrum_cache)

    rows = []
    for n_, s in scores.items():
        auc, ap_ = metrics(y, s, mask)
        c = res["candidates"].get(n_, {})
        rows.append((n_, auc, ap_, c.get("pi1", float("nan")), c.get("mode", float("nan")),
                     c.get("pi1_gauss", float("nan")), c.get("sigma0", float("nan")),
                     c.get("delta0_z", float("nan")), c.get("sigma0_z", float("nan"))))
    rows_sorted = sorted(rows, key=lambda r: -(r[3] if np.isfinite(r[3]) else -1))
    print(f"\n[{name}] fit ({'x-likelihood' if res['jacobian'] else 'residual profile'}): gamma={fit.gamma:g} "
          f"rho={fit.rho:.3f} kappa={fit.kappa:g}  selected = {res['selected']}   ({res['seconds']:.1f}s)")
    print("%-14s %7s %7s %8s %7s %8s %7s %8s %8s" % ("statistic", "AUROC", "AUPRC", "pi1_mir", "mode_w", "pi1_gau", "s0_w", "d0_z", "s0_z"))
    for r in rows_sorted[:12]:
        print("%-14s %7.2f %7.2f %8.4f %7.3f %8.4f %7.3f %8.2f %8.2f" % r)
    derived = ("SCAN", "MAG", "HSCAN", "H-", "TG-", "SURPRISE", "S-", "CAM", "DEV", "TWO-SIDED")
    scan_rows = [r for r in rows if r[0].startswith(derived)]
    rows = [r for r in rows if not r[0].startswith(derived)]
    oracle = max(rows, key=lambda r: (r[1] if np.isfinite(r[1]) else -1))
    sel = next((r for r in rows if r[0] == res["selected"]), None)
    eq = {r[0]: r[1] for r in rows if r[0] in ("J*", "R", "P", "NP")}
    print(f"\nselected {res['selected']}: AUROC {sel[1]:.2f} AUPRC {sel[2]:.2f} | oracle {oracle[0]} {oracle[1]:.2f} "
          f"| J* {eq.get('J*', float('nan')):.2f} R {eq.get('R', float('nan')):.2f} P {eq.get('P', float('nan')):.2f} NP {eq.get('NP', float('nan')):.2f}")
    print("selection-free: " + "  ".join("%s %.2f/%.2f" % (r[0], r[1], r[2]) for r in scan_rows) + "   (AUROC/AUPRC)")
    res["metrics"] = {r[0]: {"auroc": r[1], "auprc": r[2]} for r in rows}
    res["oracle"] = {"name": oracle[0], "auroc": oracle[1]}

    if args.null_check:
        check_cond = (not args.no_cond) and (not prep.truncated)
        print(f"\n[null-check] B={args.null_check} simulated fields from the fitted prior; KS(p, U[0,1]) per statistic:")
        if not check_cond and not args.no_cond:
            print("  (P skipped: the simulation draws from the truncated-spectrum model, P's null is the full-graph model)")
        ks = null_check(prep, q, build_specs(args, q),
                        B=args.null_check, seed=args.seed + 1, start=args.start, rough=args.rough,
                        include_cond=check_cond, rho=fit.rho, kappa=fit.kappa)
        worst = sorted(ks.items(), key=lambda kv: -kv[1]["ks"])
        for k_, v in worst[:6]:
            print("  %-14s KS=%.4f  mean(stat/null mean)=%.4f" % (k_, v["ks"], v["mean_ratio"]))
        n_sub = min(prep.n, 1000)
        print("  median KS=%.4f  (%d nodes x %d draws = %d p-values per statistic; iid 1%% level %.4f)" % (
            float(np.median([v["ks"] for v in ks.values()])), n_sub, args.null_check,
            n_sub * args.null_check, 1.63 / np.sqrt(n_sub * args.null_check)))
        res["null_check"] = ks

    if args.permute_check:
        rng = np.random.default_rng(args.seed + 12345)
        perm = rng.permutation(data.num_nodes)
        data_p = permute_data(data, perm)
        res_p, scores_p, _, _, _, _ = run_once(name + "_perm", data_p, args, {}, verbose=False)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        agree = {}
        for n_, s in scores.items():
            if n_ in scores_p:
                s_back = scores_p[n_][inv]
                agree[n_] = float(spearmanr(s, s_back).correlation)
        worst = min(agree.items(), key=lambda kv: kv[1])
        print(f"\n[permute-check] selected under permutation: {res_p['selected']} (unpermuted: {res['selected']}); "
              f"min Spearman over candidates = {worst[1]:.4f} ({worst[0]}); "
              f"selected-score Spearman = {agree.get(res['selected'], float('nan')):.4f}")
        res["permute_check"] = {"selected_perm": res_p["selected"], "spearman": agree}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    np.savez_compressed(os.path.join(args.out, f"{name}_scores.npz"), y=y, **{k_: v for k_, v in scores.items()})
    print(f"\nwrote {args.out}/{name}.json and {name}_scores.npz")


if __name__ == "__main__":
    main()
