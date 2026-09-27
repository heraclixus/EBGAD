"""Bank vs infinite-horizon collapse .

Direction is reversed relative to test_matched_single_score.py: the anchor is
the pipeline's actual finite-horizon bank output per dataset (the score that
produces the Table 1/Table 2 Bank cell), and the counterfactual collapses ONLY
the horizon axis to Gamma = infinity while keeping everything else fixed:
same configuration, same EB fit, same score types, same endpoint-tolerance
axis (lambda_c grid), and the same label-free aggregation machinery
(two-groups scale mixture / ebmix / ACAT / min-p / NullKS selection), applied
to the pool that survives the limit:

  eq pool = {J*, R} + {C@(inf,lam), CR@(inf,lam) : lam in ERA_LAMS}

C@(inf,lam) is the soft-endpoint control energy at infinite horizon
(endpoint covariance saturates: Sigma -> 1/q, weight w = 1/(1/lam + 1/q)).
Computed with the same soc control_energy_anomaly convention via tau=1e6
(saturation checked against the analytic weight form).

Bank side per dataset (from the E1 provenance table):
  amazon    single soft C@(tau=0.5, lam=500), h13 fine legacy config -> 78.06
  acm       single soft CR@(tau=0.25, lam=0.3), h13 drop config     -> ~86.0
  yelpchi   dp_ebmix_posterior over the h7 dynamic-profile machinery -> 95.32
  facebook  tg_component_entropy over the path p-value bank          -> 91.40
  t_finance NullKS-selected path band (fresh: PD@10:50 84.76,
            oracle PDR@10:50 86.81)

Second batch (Table 2 Bank row cells NOT covered by Table 1 provenance;
traced 2026-07-31 from results/forced_bank_ablation/forced_bank_legacy_*.json
and results/h13_transport_refine/h13_refine_drop_nullspace_blogcatalog.json):
  weibo     dynamic_combined pool, corr_stable-selected rank_mean   -> 94.21
  reddit    dynamic_combined pool, graph_tail-selected
            mix_resp_entropy                                        -> 56.74
  blogcatalog single soft C@(tau=8, lam=0.3), h13 drop affinity
            gamma=0.15 pca=48 rho=1 kappa=0.5 (oracle cell; audited
            local rerun previously drifted to ~75)                  -> 78.66
  elliptic  dynamic_combined pool ORACLE (label-selected); winning
            member is rank-identical to R@inf, i.e. equilibrium     -> 73.81
  elliptic_plus_plus  same as elliptic                              -> 72.82
The published Weibo cell is a label-free selector pick (corr_stable); the
Reddit cell is the graph_tail pick which coincides with the pool oracle; the
Elliptic/Ell++ cells match ONLY the oracle (every label-free selector on
those pools lands at 44-50).

The collapse side reports, generously, the NullKS-selected single, the
oracle single, every aggregate (tg_*, ebmix flat + grouped, acat/minp/
mean_logp, profile top-2), and the max over all of them ("collapse_best").
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

np.random.seed(0)

from data_utils import load_data
from soc.soc_anomaly import ce_ratio_anomaly, control_energy_anomaly
from test_dynamic_score_bank import (
    energy_pvalues,
    evaluate_score_bank,
    horizon_pvalue_bank,
    horizon_score_bank,
    leverage_terms,
    node_unsupervised_references,
    prepare_model_entries,
    pvalue_aggregate_scores,
    q_for_fit,
    ratio_pvalues,
    safe_auc,
    sanitize_pvalues,
)
from test_dynamic_hypotheses import (
    dynamic_profile_selector_score_bank,
    fit_two_groups_scale_mixture,
    latent_time_mixture_score_bank,
    make_profile_records,
    path_dissipation_pvalue_bank,
    path_dissipation_score_bank,
    profile_from_records,
    robust_path_aggregate_score_bank,
    two_groups_path_score_bank,
)
from test_multiscale_gou_poe import format_horizon, ks_null_deviation

FINITE_HORIZONS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
ERA_TAUS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0]
ERA_LAMS = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]
TAU_INF = 1.0e6  # numerically saturates Sigma_{0:T} -> 1/q for lam <= 500

# Producing-run configurations per bank-routed dataset (result JSONs under results/).
CONFIGS = {
    "amazon": dict(load="amazon", graph="original", tmpl="low", gamma=2.0,
                   tk=None, pca=None, sub="legacy",
                   rho=0.5947412764140091, kappa=1.0,
                   bank_kind="single_soft", bank_score=("C", 0.5, 500.0),
                   published=78.1, repro=78.06),
    "acm": dict(load="acm", graph="original", tmpl="affinity", gamma=0.2,
                tk=500, pca=256, sub="drop_nullspace",
                rho=1.0, kappa=0.03,
                bank_kind="single_soft", bank_score=("CR", 0.25, 0.3),
                published=86.0, repro=None),
    "yelpchi": dict(load="yelpchi", graph="original", tmpl="low", gamma=1.0,
                    tk=500, pca=None, sub="drop_nullspace",
                    rho=1.0, kappa=20.0,
                    bank_kind="profile_ebmix", published=95.3, repro=95.32),
    "facebook": dict(load="facebook", graph="original", tmpl="low", gamma=0.7,
                     tk=None, pca=64, sub="drop_nullspace",
                     rho=0.7267152364486923, kappa=3.0,
                     bank_kind="two_groups", published=91.4, repro=91.40),
    "t_finance": dict(load="t_finance", graph="original", tmpl="low",
                      gamma=0.2, tk=500, pca=None, sub="legacy",
                      rho=0.999, kappa=0.0,
                      bank_kind="path_ks", published=86.8,
                      repro=(84.76, 86.81)),
    # --- second batch: remaining Table 2 Bank cells (DGraph excluded) ---
    "weibo": dict(load="weibo", graph="original", tmpl="low", gamma=0.7,
                  tk=None, pca=64, sub="legacy",
                  rho=0.6969347137436389, kappa=3.0,
                  bank_kind="combined_pool", anchor_selector="corr_stable",
                  published=94.2, repro=94.21),
    "reddit": dict(load="reddit", graph="original", tmpl="low", gamma=0.7,
                   tk=None, pca=None, sub="legacy",
                   rho=0.9090750357787287, kappa=3.0,
                   bank_kind="combined_pool", anchor_selector="graph_tail",
                   published=56.7, repro=56.74),
    "blogcatalog": dict(load="blogcatalog", graph="original", tmpl="affinity",
                        gamma=0.15, tk=None, pca=48, sub="drop_nullspace",
                        rho=1.0, kappa=0.5,
                        bank_kind="single_soft", bank_score=("C", 8.0, 0.3),
                        published=78.6, repro=78.66),
    "elliptic": dict(load="elliptic", graph="original", tmpl="low", gamma=1.0,
                     tk=None, pca=64, sub="legacy",
                     rho=1.0, kappa=20.0,
                     bank_kind="combined_pool", anchor_selector="oracle",
                     published=73.8, repro=73.81),
    "elliptic_plus_plus": dict(load="elliptic_plus_plus", graph="original",
                               tmpl="low", gamma=1.0,
                               tk=None, pca=64, sub="legacy",
                               rho=1.0, kappa=20.0,
                               bank_kind="combined_pool",
                               anchor_selector="oracle",
                               published=72.8, repro=72.82),
}


def soft_scores(entry, q, pairs):
    """C/CR at (tau, lam) via the soc convention (same as the h12/h13 runs)."""
    import torch
    delta = torch.from_numpy(np.asarray(entry["delta"], dtype=np.float32))
    zeros = torch.zeros_like(delta)
    V = torch.from_numpy(np.asarray(entry["V"], dtype=np.float32))
    lam_q = torch.from_numpy(np.asarray(q, dtype=np.float32))
    out = {}
    with torch.no_grad():
        for tau, lp in pairs:
            c = control_energy_anomaly(delta, zeros, V, lam_q, 1.0, float(tau),
                                       lam_penalty=float(lp))
            cr = ce_ratio_anomaly(delta, zeros, V, lam_q, 1.0, float(tau),
                                  lam_penalty=float(lp))
            label = "inf" if tau >= TAU_INF else "%g" % tau
            out["C@%s,%g" % (label, lp)] = c.cpu().numpy().astype(np.float32)
            out["CR@%s,%g" % (label, lp)] = cr.cpu().numpy().astype(np.float32)
    return out


def eq_pool_banks(entry, q):
    """Gamma = infinity pool: {J*, R} + soft C/CR at (inf, lam) with p-values.

    p-values use the same plug-in construction as the horizon bank: diagonal
    weight w = 1/(1/lam + 1/q) (the exact Gamma->inf limit of the soft-endpoint
    precision) through leverage_terms + energy/ratio_pvalues.
    """
    scores = dict(horizon_score_bank(entry, q, [np.inf]))
    pvalues = dict(horizon_pvalue_bank(entry, q, [np.inf], scores))

    soft = soft_scores(entry, q, [(TAU_INF, lam) for lam in ERA_LAMS])
    q_safe = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    for lam in ERA_LAMS:
        w = 1.0 / (1.0 / float(lam) + 1.0 / q_safe)
        num_scale, den_scale, cross_scale = leverage_terms(entry, q, w)
        c_name = "C@inf,%g" % lam
        cr_name = "CR@inf,%g" % lam
        scores[c_name] = soft[c_name]
        scores[cr_name] = soft[cr_name]
        pvalues[c_name] = energy_pvalues(soft[c_name], num_scale, entry["d"])
        pvalues[cr_name] = ratio_pvalues(soft[cr_name], num_scale, den_scale,
                                         cross_scale, entry["d"])
    return scores, pvalues


def saturation_check(entry, q):
    """Verify tau=1e6 reproduces the analytic Gamma=inf weight form."""
    import torch
    delta_hat = np.asarray(entry["V"], dtype=np.float64).T @ np.asarray(
        entry["delta"], dtype=np.float64)
    energy_modes = np.sum(delta_hat ** 2, axis=1)
    q_safe = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    worst = 0.0
    for lam in (ERA_LAMS[0], ERA_LAMS[-1]):
        w = 1.0 / (1.0 / float(lam) + 1.0 / q_safe)
        analytic_total = float(np.sum(w * energy_modes))
        delta = torch.from_numpy(np.asarray(entry["delta"], dtype=np.float32))
        zeros = torch.zeros_like(delta)
        V = torch.from_numpy(np.asarray(entry["V"], dtype=np.float32))
        lam_q = torch.from_numpy(np.asarray(q, dtype=np.float32))
        with torch.no_grad():
            c = control_energy_anomaly(delta, zeros, V, lam_q, 1.0, TAU_INF,
                                       lam_penalty=float(lam))
        func_total = float(np.sum(c.cpu().numpy()))
        if analytic_total > 0:
            worst = max(worst, abs(func_total - analytic_total)
                        / max(analytic_total, 1e-12))
    return worst


def rate(scores, pvalue_bank, y, mask):
    y_eval = y[mask]
    out = {}
    for name, score in scores.items():
        s = np.nan_to_num(np.asarray(score, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmax(s) <= np.nanmin(s):
            continue
        out[name] = {
            "auc": float(safe_auc(y_eval, s[mask])),
            "nullks": float(ks_null_deviation(s[mask])),
        }
    return out


def collapse_report(entry, q, y, mask, references=None, matched_selector=None):
    """All infinite-horizon machinery outputs: singles + every aggregate."""
    eq_scores, eq_pvals = eq_pool_banks(entry, q)
    singles = rate(eq_scores, eq_pvals, y, mask)

    agg_scores = {}

    # two-groups scale mixture (root call convention: Facebook's producer)
    try:
        tg_scores, _ = two_groups_path_score_bank(
            eq_pvals, pi_init=0.05, pi_max=0.20, weight_alpha=1.05)
        for k, v in tg_scores.items():
            agg_scores["tg/%s" % k] = v
    except Exception as exc:
        agg_scores["tg/error"] = None
        print("    two-groups failed: %r" % exc, flush=True)

    # ebmix over the flat pool (dynamic_profile constants)
    try:
        _, posterior, _, _, _, _ = fit_two_groups_scale_mixture(
            eq_pvals, pi_init=0.05, pi_max=0.20, weight_alpha=1.10,
            a_min=0.05, a_max=0.95)
        agg_scores["ebmix_flat_posterior"] = posterior.astype(np.float32)
    except Exception as exc:
        print("    ebmix flat failed: %r" % exc, flush=True)

    # ebmix over grouped profiles (energy family / ratio family), mirroring
    # the dynamic_profile two-stage structure with score-type groups standing
    # in for the horizon-band profiles that no longer exist at Gamma=inf.
    try:
        groups = {
            "energy": [n for n in eq_scores if n.startswith(("J@", "C@"))],
            "ratio": [n for n in eq_scores if n.startswith(("R@", "CR@"))],
        }
        prof_pvals = {}
        for gname, names in groups.items():
            records = make_profile_records(names, eq_scores, eq_pvals)
            if not records:
                continue
            score, pvals, _, _ = profile_from_records(records)
            agg_scores["profile_top2/%s" % gname] = score.astype(np.float32)
            prof_pvals["dp_%s" % gname] = pvals
        if prof_pvals:
            _, posterior, _, _, _, _ = fit_two_groups_scale_mixture(
                prof_pvals, pi_init=0.05, pi_max=0.20, weight_alpha=1.10,
                a_min=0.05, a_max=0.95)
            agg_scores["ebmix_grouped_posterior"] = posterior.astype(np.float32)
    except Exception as exc:
        print("    ebmix grouped failed: %r" % exc, flush=True)

    # ACAT / min-p (raw + median-calibrated), mean_logp
    try:
        pa_scores, _ = pvalue_aggregate_scores(eq_pvals)
        for k, v in pa_scores.items():
            agg_scores["pool/%s" % k] = v
        pmat = np.vstack([sanitize_pvalues(eq_pvals[n]) for n in sorted(eq_pvals)]).T
        agg_scores["pool/mean_logp"] = np.mean(
            -np.log(np.maximum(pmat, 1e-300)), axis=1).astype(np.float32)
    except Exception as exc:
        print("    pool aggregates failed: %r" % exc, flush=True)

    aggregates = rate({k: v for k, v in agg_scores.items() if v is not None},
                      {}, y, mask)

    sel_name = max(singles, key=lambda n: singles[n]["nullks"])
    ora_name = max(singles, key=lambda n: singles[n]["auc"])
    everything = dict(singles)
    everything.update(aggregates)
    best_name = max(everything, key=lambda n: everything[n]["auc"])
    out = {
        "singles": singles,
        "aggregates": aggregates,
        "selected_single": {"name": sel_name, **singles[sel_name]},
        "oracle_single": {"name": ora_name, **singles[ora_name]},
        "collapse_best": {"name": best_name, **everything[best_name]},
    }

    # Matched-selector collapse for combined_pool datasets: run the SAME
    # evaluate_score_bank machinery (rank fusions + selector diagnostics) on
    # the Gamma=inf pool, so the bank anchor's selector (corr_stable /
    # graph_tail / oracle) has a like-for-like counterpart.
    if references is not None:
        pool = dict(eq_scores)
        pool.update({k: v for k, v in agg_scores.items() if v is not None})
        summary = evaluate_score_bank(
            y, mask, pool, references=references, pvalue_bank=eq_pvals)
        sel_summary = {
            k: {"score": v["score"], "auc": float(v["auc"])}
            for k, v in summary.get("selectors", {}).items()
        }
        out["matched_machinery"] = {
            "selectors": sel_summary,
            "ks_selected": {"name": summary.get("score_ks"),
                            "auc": float(summary.get("auc_ks_selected"))},
            "oracle": {"name": summary.get("oracle_score"),
                       "auc": float(summary.get("auc_oracle"))},
        }
        if matched_selector == "oracle":
            out["matched_collapse"] = dict(out["matched_machinery"]["oracle"])
        elif matched_selector in sel_summary:
            pick = sel_summary[matched_selector]
            out["matched_collapse"] = {"name": pick["score"],
                                       "auc": pick["auc"]}
    return out


def bank_report(entry, q, y, mask, cfg, references=None):
    kind = cfg["bank_kind"]
    y_eval = y[mask]

    if kind == "single_soft":
        stype, tau, lam = cfg["bank_score"]
        pairs = [(t, l) for t in ERA_TAUS for l in ERA_LAMS]
        scores = soft_scores(entry, q, pairs)
        rated = rate(scores, {}, y, mask)
        fam = {n: r for n, r in rated.items() if n.startswith(stype + "@")
               and not n.startswith(stype + "@inf")}
        anchor_name = "%s@%g,%g" % (stype, tau, lam)
        sel_name = max(fam, key=lambda n: fam[n]["nullks"])
        ora_name = max(fam, key=lambda n: fam[n]["auc"])
        return {
            "kind": kind,
            "anchor": {"name": anchor_name, **rated.get(anchor_name, {})},
            "selected": {"name": sel_name, **fam[sel_name]},
            "oracle": {"name": ora_name, **fam[ora_name]},
            "matched_collapse_member": "%s@inf,%g" % (stype, lam),
        }

    # trajectory machinery banks (yelpchi / facebook / t_finance)
    path_scores, _ = path_dissipation_score_bank(entry, q, FINITE_HORIZONS)
    path_pvalues = path_dissipation_pvalue_bank(
        entry, q, FINITE_HORIZONS, path_scores)

    if kind == "two_groups":
        tg_scores, _ = two_groups_path_score_bank(
            path_pvalues, pi_init=0.05, pi_max=0.20, weight_alpha=1.05)
        rated = rate(tg_scores, {}, y, mask)
        anchor = rated.get("tg_component_entropy", {})
        return {"kind": kind,
                "anchor": {"name": "tg_component_entropy", **anchor},
                "all_tg": rated}

    if kind == "path_ks":
        rated = rate(path_scores, {}, y, mask)
        sel_name = max(rated, key=lambda n: rated[n]["nullks"])
        ora_name = max(rated, key=lambda n: rated[n]["auc"])
        return {"kind": kind,
                "anchor": {"name": sel_name, **rated[sel_name]},
                "oracle": {"name": ora_name, **rated[ora_name]},
                "named": {n: rated[n] for n in ("PD@10:50", "PDR@10:50")
                          if n in rated}}

    if kind == "profile_ebmix":
        horizons = FINITE_HORIZONS + [np.inf]
        stationary_scores = horizon_score_bank(entry, q, [np.inf])
        tg_scores, _ = two_groups_path_score_bank(
            path_pvalues, pi_init=0.05, pi_max=0.20, weight_alpha=1.05)
        mixture_scores, _ = latent_time_mixture_score_bank(
            entry, q, horizons, dirichlet_alpha=1.05)
        profile_scores, _, _ = dynamic_profile_selector_score_bank(
            path_scores, path_pvalues, mixture_scores, tg_scores,
            stationary_scores)
        rated = rate(profile_scores, {}, y, mask)
        anchor = rated.get("dp_ebmix_posterior", {})
        top = sorted(rated.items(), key=lambda kv: -kv[1]["auc"])[:5]
        return {"kind": kind,
                "anchor": {"name": "dp_ebmix_posterior", **anchor},
                "top5": {n: r for n, r in top}}

    if kind == "combined_pool":
        # exact mirror of the forced-bank dynamic_combined pool
        # (test_dynamic_hypotheses.run_dataset with --finite_only_bank):
        # path + robust_path + mixture + two_groups + profile scores,
        # evaluated with the same selector machinery.
        robust_scores, _ = robust_path_aggregate_score_bank(
            path_scores, path_pvalues, min_bands=3, max_bands=8)
        tg_scores, _ = two_groups_path_score_bank(
            path_pvalues, pi_init=0.05, pi_max=0.20, weight_alpha=1.05)
        mixture_scores, _ = latent_time_mixture_score_bank(
            entry, q, FINITE_HORIZONS, dirichlet_alpha=1.05)
        stationary_scores = horizon_score_bank(entry, q, [np.inf])
        profile_scores, profile_pvalues, _ = dynamic_profile_selector_score_bank(
            path_scores, path_pvalues, mixture_scores, tg_scores,
            stationary_scores)
        combined = {}
        combined.update(path_scores)
        combined.update(robust_scores)
        combined.update(mixture_scores)
        combined.update(tg_scores)
        combined.update(profile_scores)
        combined_pvalues = dict(path_pvalues)
        combined_pvalues.update(profile_pvalues)
        summary = evaluate_score_bank(
            y, mask, combined, references=references,
            pvalue_bank=combined_pvalues)
        sel = cfg["anchor_selector"]
        sel_summary = {
            k: {"score": v["score"], "auc": float(v["auc"])}
            for k, v in summary.get("selectors", {}).items()
        }
        if sel == "oracle":
            anchor = {"name": summary.get("oracle_score"),
                      "auc": float(summary.get("auc_oracle"))}
        else:
            pick = sel_summary[sel]
            anchor = {"name": pick["score"], "auc": pick["auc"]}
        return {"kind": kind,
                "anchor": anchor,
                "anchor_selector": sel,
                "oracle": {"name": summary.get("oracle_score"),
                           "auc": float(summary.get("auc_oracle"))},
                "ks_selected": {"name": summary.get("score_ks"),
                                "auc": float(summary.get("auc_ks_selected"))},
                "selectors": sel_summary}

    raise ValueError(kind)


def run_dataset(ds, device):
    cfg = CONFIGS[ds]
    load_name = ("YelpChi" if cfg["load"] == "yelpchi" else
                 "Facebook" if cfg["load"] == "facebook" else cfg["load"])
    print("\n=== %s | tmpl=%s gamma=%g tk=%s pca=%s sub=%s rho=%g kappa=%g ==="
          % (ds, cfg["tmpl"], cfg["gamma"], cfg["tk"], cfg["pca"], cfg["sub"],
             cfg["rho"], cfg["kappa"]), flush=True)
    data = load_data(load_name)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)

    entries = prepare_model_entries(
        data,
        graph_types=[cfg["graph"]],
        template_gammas=[cfg["gamma"]],
        pca_dim=cfg["pca"],
        truncated_k=cfg["tk"],
        modeled_subspace=cfg["sub"],
        template_type=cfg["tmpl"],
        device=device,
    )
    entry = entries[0]
    fit = {"rho": float(cfg["rho"]), "kappa": float(cfg["kappa"]),
           "ml": float("nan"), "entry_key": entry["key"]}
    q = q_for_fit(entry, fit)

    sat = saturation_check(entry, q)
    print("  tau=1e6 saturation rel-err vs analytic: %.2e" % sat, flush=True)

    references = None
    if cfg["bank_kind"] == "combined_pool":
        references = node_unsupervised_references(data, entry["delta"])

    print("  bank side (%s)" % cfg["bank_kind"], flush=True)
    bank = bank_report(entry, q, y, mask, cfg, references=references)
    print("    anchor %s = %.2f  [published %.1f, repro %s]" % (
        bank["anchor"].get("name"), 100 * bank["anchor"].get("auc", float("nan")),
        cfg["published"], cfg["repro"]), flush=True)
    if cfg["bank_kind"] == "combined_pool":
        print("    pool oracle %s = %.2f | NullKS pick %s = %.2f" % (
            bank["oracle"]["name"], 100 * bank["oracle"]["auc"],
            bank["ks_selected"]["name"], 100 * bank["ks_selected"]["auc"]),
            flush=True)

    print("  collapse side (Gamma = inf pool + same machinery)", flush=True)
    collapse = collapse_report(
        entry, q, y, mask, references=references,
        matched_selector=cfg.get("anchor_selector"))
    print("    selected single %s = %.2f | oracle single %s = %.2f | "
          "best-of-everything %s = %.2f" % (
              collapse["selected_single"]["name"],
              100 * collapse["selected_single"]["auc"],
              collapse["oracle_single"]["name"],
              100 * collapse["oracle_single"]["auc"],
              collapse["collapse_best"]["name"],
              100 * collapse["collapse_best"]["auc"]), flush=True)

    if cfg["bank_kind"] == "single_soft":
        mc = bank["matched_collapse_member"]
        rec = collapse["singles"].get(mc)
        if rec:
            print("    matched one-parameter collapse %s = %.2f (bank %s = %.2f)"
                  % (mc, 100 * rec["auc"], bank["anchor"]["name"],
                     100 * bank["anchor"].get("auc", float("nan"))), flush=True)
    if "matched_collapse" in collapse:
        print("    matched-selector collapse (%s) %s = %.2f (bank anchor %.2f)"
              % (cfg.get("anchor_selector"),
                 collapse["matched_collapse"].get("name"),
                 100 * collapse["matched_collapse"]["auc"],
                 100 * bank["anchor"].get("auc", float("nan"))), flush=True)

    return {
        "dataset": ds,
        "config": {k: cfg.get(k) for k in ("load", "graph", "tmpl", "gamma",
                                           "tk", "pca", "sub", "rho", "kappa",
                                           "bank_kind", "bank_score",
                                           "anchor_selector", "published",
                                           "repro")},
        "saturation_rel_err": sat,
        "bank": bank,
        "collapse": collapse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(CONFIGS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out",
                        default="results/horizon_study/bank_horizon_collapse.json")
    args = parser.parse_args()

    results = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        try:
            results[ds] = run_dataset(ds, args.device)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[ds] = {"error": str(exc)[:500]}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print("\nSaved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
