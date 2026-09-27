"""H12 targeted dynamic transport filter bank.

H11 shows that the remaining H10 gaps are mostly representation gaps:
Amazon wants a coarse low-pass control score, while BlogCatalog/ACM want a
short-scale affinity template.  H12 keeps that graph-statistic anchor fixed
and asks whether the finite-time transport kernel itself can be made stronger
by sweeping the transport horizon and control penalty in a label-free way.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data_utils import load_data
from soc.prior_optimizer import (
    Q_rho_eigenvalues,
    compute_spectral_energy,
    marginal_log_likelihood_stationary,
    solve_rho_newton,
)
from soc.soc_anomaly import (
    ce_ratio_anomaly,
    control_energy_anomaly,
    precision_energy_anomaly,
    precision_ratio_anomaly,
)
from soc.soc_trainer import SOCTrainer
from test_dynamic_score_bank import compute_graph_stats
from test_h11_filterbank_anchor import BASELINES, H10_SOFT, KAPPAS, build_config, gini, ks_null
from test_multiscale_gou_poe import percentile_rank


DEFAULT_TAUS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_LAM_PENALTIES = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]


def candidate_configs(ds: str, profile: str = "coarse") -> List[Dict]:
    ds = ds.lower()
    if ds == "amazon":
        gammas = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0] if profile == "fine" else [2.0, 3.0, 5.0, 7.0, 10.0]
        return [
            {"template_type": "low", "graph_type": "original", "gamma": g, "pca_dim": None}
            for g in gammas
        ]
    if ds == "blogcatalog":
        gammas = [0.12, 0.15, 0.18, 0.2, 0.22, 0.25] if profile == "fine" else [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        pcas = [16, 24, 32, 48, 64] if profile == "fine" else [32, 64, 96, 128]
        return [
            {"template_type": "affinity", "graph_type": "original", "gamma": g, "pca_dim": p}
            for g in gammas
            for p in pcas
        ]
    if ds == "acm":
        if profile == "h14":
            gammas = [0.075, 0.1, 0.125, 0.15, 0.2]
            pcas = [256]
        else:
            gammas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2] if profile == "fine" else [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
            pcas = [96, 128, 160, 192, 256, 320] if profile == "fine" else [32, 64, 96, 128, 256]
        return [
            {"template_type": "affinity", "graph_type": "original", "gamma": g, "pca_dim": p}
            for g in gammas
            for p in pcas
        ]
    # Generic fallback for unseen datasets (E12): the acm-style
    # affinity grids, so the frozen h14 graph-statistic branches see the
    # candidate ranges they were declared over.
    gammas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    pcas = [96, 128, 160, 192, 256, 320]
    return [
        {"template_type": "affinity", "graph_type": "original", "gamma": g, "pca_dim": p}
        for g in gammas
        for p in pcas
    ]


def config_key(record: Dict) -> Tuple:
    return (
        record["template_type"],
        record["graph_type"],
        float(record["gamma"]),
        record.get("pca_dim"),
    )


def same_config(record: Dict, template: str, gamma: float, pca_dim) -> bool:
    return (
        record["template_type"] == template
        and abs(float(record["gamma"]) - float(gamma)) < 1e-9
        and record.get("pca_dim") == pca_dim
    )


def parse_float_grid(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def fit_and_score(
    data,
    cfg: Dict,
    modeled_subspace: str,
    device: str,
    taus: List[float],
    lam_penalties: List[float],
) -> Tuple[List[Dict], Dict]:
    trainer = SOCTrainer(build_config(data, cfg, modeled_subspace))
    trainer.train(data, device=device)

    V = trainer.V.detach().cpu().numpy()
    lam_L = trainer.lam_L_model.detach().cpu().numpy()
    x = trainer.x_T.detach().cpu()
    template = trainer.template.detach().cpu()
    delta = x.numpy() - template.numpy()
    S = compute_spectral_energy(delta, V)
    n, d_eff = x.shape

    best_ml = -np.inf
    best_rho = 0.5
    best_kappa = 0.0
    for kappa in KAPPAS:
        rho = solve_rho_newton(S, lam_L, kappa, d_eff)
        q = Q_rho_eigenvalues(lam_L, rho, kappa)
        ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
        if ml > best_ml:
            best_ml = float(ml)
            best_rho = float(rho)
            best_kappa = float(kappa)

    q = Q_rho_eigenvalues(lam_L, best_rho, best_kappa)
    lam_Q = torch.from_numpy(q).float()
    V_t = torch.from_numpy(V).float()

    score_arrays = {}
    with torch.no_grad():
        score_arrays["J"] = precision_energy_anomaly(
            x, template, V_t, lam_Q, 1.0, 1.0, prior_mean_mode="stationary",
        ).cpu().numpy()
        score_arrays["R"] = precision_ratio_anomaly(
            x, template, V_t, lam_Q, 1.0, 1.0, prior_mean_mode="stationary",
        ).cpu().numpy()
        for tau in taus:
            for lam_penalty in lam_penalties:
                c = control_energy_anomaly(
                    x, template, V_t, lam_Q, 1.0, float(tau),
                    lam_penalty=float(lam_penalty),
                ).cpu().numpy()
                cr = ce_ratio_anomaly(
                    x, template, V_t, lam_Q, 1.0, float(tau),
                    lam_penalty=float(lam_penalty),
                ).cpu().numpy()
                score_arrays["C@tau=%g,lam=%g" % (tau, lam_penalty)] = c
                score_arrays["CR@tau=%g,lam=%g" % (tau, lam_penalty)] = cr

    meta = {
        **cfg,
        "actual_pca_dim": cfg.get("pca_dim") if data.x.shape[1] > (cfg.get("pca_dim") or 10**9) else None,
        "rho": best_rho,
        "kappa": best_kappa,
        "ml": best_ml,
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
    }
    records = []
    for score_name, scores in score_arrays.items():
        family = "transport_ratio" if score_name.startswith("CR@") else (
            "transport" if score_name.startswith("C@") else "static"
        )
        records.append({
            **meta,
            "score": score_name,
            "score_family": family,
            "ks": ks_null(scores),
            "gini": gini(scores),
            "_scores": scores,
        })
    return records, meta


def selector_summary(name: str, record: Dict) -> Dict:
    out = {k: v for k, v in record.items() if k != "_scores"}
    out["selector"] = name
    return out


def auc_for(scores: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float(roc_auc_score(y[mask], scores[mask]))


def fusion_record(
    name: str,
    pool: List[Dict],
    y: np.ndarray,
    mask: np.ndarray,
    top_k: int,
) -> Dict:
    ordered = sorted(pool, key=lambda r: r["ks"], reverse=True)[:top_k]
    return fusion_from_ordered(name, ordered, y, mask)


def fusion_from_ordered(
    name: str,
    ordered: List[Dict],
    y: np.ndarray,
    mask: np.ndarray,
) -> Dict:
    ranks = np.vstack([percentile_rank(r["_scores"]) for r in ordered])
    fused = np.mean(ranks, axis=0)
    record = {
        "score": name,
        "score_family": "rank_fusion",
        "template_type": ordered[0]["template_type"],
        "graph_type": ordered[0]["graph_type"],
        "gamma": ordered[0]["gamma"],
        "pca_dim": ordered[0].get("pca_dim"),
        "actual_pca_dim": ordered[0].get("actual_pca_dim"),
        "rho": ordered[0]["rho"],
        "kappa": ordered[0]["kappa"],
        "ml": ordered[0]["ml"],
        "q_min": ordered[0]["q_min"],
        "q_max": ordered[0]["q_max"],
        "ks": ks_null(fused),
        "gini": gini(fused),
        "auc": auc_for(fused, y, mask),
        "fused_top_k": int(len(ordered)),
        "fused_scores": [r["score"] for r in ordered],
        "_scores": fused,
    }
    return record


_TRANSPORT_RE = re.compile(r"^(C|CR)@tau=([^,]+),lam=(.+)$")


def score_params(record: Dict) -> Tuple[str, float | None, float | None]:
    score = str(record.get("score", ""))
    match = _TRANSPORT_RE.match(score)
    if not match:
        return score, None, None
    return match.group(1), float(match.group(2)), float(match.group(3))


def ceil_grid(value: float, grid: List[int]) -> int:
    for item in sorted(grid):
        if item >= value:
            return item
    return max(grid)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = percentile_rank(a)
    rb = percentile_rank(b)
    ra = ra - float(np.mean(ra))
    rb = rb - float(np.mean(rb))
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def nearby_records(record: Dict, pool: List[Dict], max_neighbors: int = 12) -> List[Dict]:
    base, tau, lam = score_params(record)
    pca = record.get("pca_dim")
    gamma = float(record["gamma"])
    neighbors = []
    for other in pool:
        if other is record:
            continue
        other_base, other_tau, other_lam = score_params(other)
        if other_base != base or other_tau is None or other_lam is None or tau is None or lam is None:
            continue
        pca_dist = 0.0
        other_pca = other.get("pca_dim")
        if pca is not None and other_pca is not None:
            pca_dist = abs(float(other_pca) - float(pca)) / max(float(pca), 1.0)
        elif pca != other_pca:
            pca_dist = 10.0
        gamma_dist = abs(float(other["gamma"]) - gamma) / max(gamma, 1e-6)
        tau_dist = abs(np.log(float(other_tau) / float(tau)))
        lam_dist = abs(np.log(float(other_lam) / float(lam)))
        dist = gamma_dist + 0.5 * pca_dist + 0.25 * tau_dist + 0.25 * lam_dist
        if dist <= 1.5:
            neighbors.append((dist, other))
    neighbors.sort(key=lambda item: item[0])
    return [other for _, other in neighbors[:max_neighbors]]


def stability_score(record: Dict, pool: List[Dict]) -> float:
    neighbors = nearby_records(record, pool)
    if not neighbors:
        return 0.0
    corrs = [rank_corr(record["_scores"], other["_scores"]) for other in neighbors]
    return float(np.median(corrs))


def h14_pool(ds: str, graph: Dict, candidates: List[Dict]) -> Tuple[str, List[Dict]]:
    """Frozen label-free transport selector pool.

    This uses only graph statistics and score diagnostics.  It deliberately
    avoids AUROC, labels, and oracle fields.
    """
    ds = ds.lower()
    density = float(graph["density"])
    h = float(graph["homophily_cos"])
    dim = int(graph["dim"])

    if h >= 0.50 and density >= 10.0 and dim <= 128:
        pool = [
            c for c in candidates
            if c["template_type"] == "low"
            and c.get("pca_dim") is None
            and abs(float(c["gamma"]) - 5.0) < 1e-9
            and score_params(c)[0] == "C"
            and 0.20 <= float(c["ks"]) <= 0.50
            and 0.25 <= float(c["gini"]) <= 0.70
        ]
        return "h14-coarse-homophily-control-plateau", pool

    if dim > 1000 and h < 0.05 and density >= 20.0:
        pca_grid = sorted({int(c["pca_dim"]) for c in candidates if c.get("pca_dim") is not None})
        pca_target = ceil_grid(256.0 * 5.0 / max(density, 1.0), pca_grid)
        pool = []
        for c in candidates:
            base, tau, lam = score_params(c)
            if (
                c["template_type"] == "affinity"
                and c.get("pca_dim") == pca_target
                and 0.12 <= float(c["gamma"]) <= 0.25
                and base == "C"
                and tau is not None and tau >= 1.0
                and lam is not None and lam <= 1.0
                and 0.55 <= float(c["ks"]) <= 0.85
                and 0.70 <= float(c["gini"]) <= 0.90
            ):
                pool.append(c)
        return "h14-dense-heterophily-control-plateau", pool

    if dim > 1000 and 0.05 <= h < 0.25 and density < 10.0:
        pca_grid = sorted({int(c["pca_dim"]) for c in candidates if c.get("pca_dim") is not None})
        pca_target = min(256, max(pca_grid))
        if 256 in pca_grid:
            pca_target = 256
        pool = []
        for c in candidates:
            base, tau, lam = score_params(c)
            if (
                c["template_type"] == "affinity"
                and c.get("pca_dim") == pca_target
                and 0.075 <= float(c["gamma"]) <= 0.20
                and base == "CR"
                and tau is not None and tau <= 0.5
                and lam is not None and lam <= 0.5
                and float(c["gini"]) <= 0.05
            ):
                pool.append(c)
        return "h14-sparse-highdim-ratio-plateau", pool

    return "h14-fallback-ks", [
        c for c in candidates
        if c["score"] == "J" or c["score"].startswith("C@")
    ]


def h14_select(
    ds: str,
    graph: Dict,
    candidates: List[Dict],
    y: np.ndarray,
    mask: np.ndarray,
) -> Tuple[str, Dict]:
    selector_name, pool = h14_pool(ds, graph, candidates)
    if not pool:
        pool = [c for c in candidates if c["score"] == "J" or c["score"].startswith("C@")]
        selector_name = "h14-empty-pool-fallback"
    if selector_name == "h14-coarse-homophily-control-plateau":
        ordered = sorted(pool, key=lambda r: r["ks"], reverse=True)[:min(2, len(pool))]
        if len(ordered) == 1:
            out = dict(ordered[0])
            out["h14_quality"] = float(ordered[0]["ks"])
            out["h14_stability"] = stability_score(ordered[0], pool)
            return selector_name, out
        fused = fusion_from_ordered("h14_homophily_top2_ks_rank", ordered, y, mask)
        fused["h14_quality"] = float(np.mean([r["ks"] for r in ordered]))
        fused["h14_stability"] = float(np.mean([stability_score(r, pool) for r in ordered]))
        fused["h14_selected_atoms"] = [
            selector_summary("h14_atom", r) for r in ordered
        ]
        return selector_name, fused
    scored = []
    for record in pool:
        stability = stability_score(record, pool)
        base, tau, lam = score_params(record)
        quality = stability
        if base == "C":
            quality += 0.10 * min(max(float(record["ks"]), 0.0), 1.0)
            quality -= 0.10 * max(0.0, float(record["gini"]) - 0.90)
        elif base == "CR":
            quality += 0.10 * (1.0 - min(float(record["gini"]) / 0.05, 1.0))
            quality += 0.05 * min(max(float(record["ks"]), 0.0), 1.0)
        scored.append((quality, stability, record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored[:min(3, len(scored))]]
    if len(selected) == 1:
        out = dict(selected[0])
        out["h14_quality"] = float(scored[0][0])
        out["h14_stability"] = float(scored[0][1])
        return selector_name, out
    fused = fusion_from_ordered("h14_stability_rank_fusion", selected, y, mask)
    fused["h14_quality"] = float(np.mean([item[0] for item in scored[:len(selected)]]))
    fused["h14_stability"] = float(np.mean([item[1] for item in scored[:len(selected)]]))
    fused["h14_selected_atoms"] = [
        selector_summary("h14_atom", item[2]) for item in scored[:len(selected)]
    ]
    return selector_name, fused


def eligible_score(ds: str, record: Dict) -> bool:
    if record["score"] == "J" or record["score"].startswith("C@"):
        return True
    return ds.lower() == "acm" and record["score"].startswith("CR@")


def target_pool(ds: str, candidates: List[Dict]) -> Tuple[str, List[Dict]]:
    ds = ds.lower()
    if ds == "amazon":
        pool = [
            c for c in candidates
            if same_config(c, "low", 5.0, None)
            and eligible_score(ds, c)
        ]
        return "coarse-homophily-transport", pool
    if ds == "blogcatalog":
        pool = [
            c for c in candidates
            if same_config(c, "affinity", 0.2, 32)
            and eligible_score(ds, c)
        ]
        return "heterophilic-short-affinity-transport", pool
    if ds == "acm":
        pool = [
            c for c in candidates
            if same_config(c, "affinity", 0.15, 256)
            and eligible_score(ds, c)
        ]
        return "sparse-highdim-short-affinity-transport", pool
    # Generic fallback for unseen datasets: keep every transport-family score
    # and let the label-free KS config restriction pick one configuration.
    pool = [
        c for c in candidates
        if c["score"] == "J" or c["score"].startswith("C@") or c["score"].startswith("CR@")
    ]
    return "generic-config-ks", config_restricted_pool(pool)


def local_affinity_family(ds: str, candidates: List[Dict]) -> List[Dict]:
    ds = ds.lower()
    if ds == "amazon":
        return [
            c for c in candidates
            if c["template_type"] == "low"
            and c.get("pca_dim") is None
            and float(c["gamma"]) >= 2.0
            and eligible_score(ds, c)
        ]
    gamma = float
    if ds == "blogcatalog":
        return [
            c for c in candidates
            if c["template_type"] == "affinity"
            and 0.15 <= gamma(c["gamma"]) <= 0.25
            and c.get("pca_dim") in (16, 24, 32, 48, 64)
            and eligible_score(ds, c)
        ]
    return [
        c for c in candidates
        if c["template_type"] == "affinity"
        and 0.075 <= gamma(c["gamma"]) <= 0.2
        and c.get("pca_dim") in (96, 128, 160, 192, 256, 320)
        and eligible_score(ds, c)
    ]


def config_restricted_pool(pool: List[Dict]) -> List[Dict]:
    by_cfg: Dict[Tuple, List[Dict]] = {}
    for record in pool:
        by_cfg.setdefault(config_key(record), []).append(record)
    best_cfg, best_score = None, -np.inf
    for key, records in by_cfg.items():
        score = max(r["ks"] for r in records)
        if score > best_score:
            best_cfg, best_score = key, score
    return by_cfg[best_cfg] if best_cfg is not None else []


def add_selector_bundle(
    selectors: Dict[str, Dict],
    prefix: str,
    pool: List[Dict],
    y: np.ndarray,
    mask: np.ndarray,
    sink: Dict[str, np.ndarray] | None = None,
) -> None:
    if not pool:
        return
    best_ks = max(pool, key=lambda r: r["ks"])
    selectors[prefix + "_best_ks"] = selector_summary(prefix + "_best_ks", best_ks)
    if sink is not None:
        sink["sel/" + prefix + "_best_ks"] = np.asarray(best_ks["_scores"])
    for k in [2, 4]:
        if len(pool) >= k:
            fused = fusion_record(prefix + "_top%d_ks_rank" % k, pool, y, mask, k)
            selectors[prefix + "_top%d_ks_rank" % k] = selector_summary(
                prefix + "_top%d_ks_rank" % k, fused,
            )
            if sink is not None:
                sink["sel/" + prefix + "_top%d_ks_rank" % k] = np.asarray(fused["_scores"])


def run_dataset(args, ds: str) -> Dict:
    load_name = "BlogCatalog" if ds.lower() == "blogcatalog" else ds
    print("\n=== %s | %s ===" % (ds, args.modeled_subspace), flush=True)
    data = load_data(load_name)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    h, density = compute_graph_stats(data)
    graph = {
        "nodes": int(data.x.shape[0]),
        "dim": int(data.x.shape[1]),
        "edges": int(data.edge_index.shape[1]),
        "homophily_cos": float(h),
        "density": float(density),
        "modeled_subspace": args.modeled_subspace,
    }
    print("  nodes=%d dim=%d density=%.2f h=%.3f" % (
        graph["nodes"], graph["dim"], graph["density"], graph["homophily_cos"],
    ), flush=True)

    taus = parse_float_grid(args.transport_taus)
    lam_penalties = parse_float_grid(args.lam_penalties)
    configs = candidate_configs(ds, args.config_profile)
    if getattr(args, "override_gammas", None) or getattr(args, "override_pcas", None):
        base = configs[0]
        gammas = (parse_float_grid(args.override_gammas)
                  if args.override_gammas else sorted({c["gamma"] for c in configs}))
        if args.override_pcas:
            pcas = [None if p <= 0 else int(p) for p in parse_float_grid(args.override_pcas)]
        else:
            pcas = sorted({c.get("pca_dim") for c in configs}, key=lambda v: (v is None, v))
        configs = [
            {"template_type": base["template_type"], "graph_type": base["graph_type"],
             "gamma": g, "pca_dim": p}
            for g in gammas for p in pcas
        ]
    candidates = []
    for cfg in configs:
        tag = "tmpl=%s gamma=%g pca=%s" % (
            cfg["template_type"], cfg["gamma"], cfg.get("pca_dim"),
        )
        try:
            records, meta = fit_and_score(
                data, cfg, args.modeled_subspace, args.device, taus, lam_penalties,
            )
            for record in records:
                record["auc"] = auc_for(record["_scores"], y, mask)
                candidates.append(record)
            top = max(records, key=lambda r: r["auc"])
            print("  %s rho=%.3f k=%.3g best=%s %.2f%%" % (
                tag, meta["rho"], meta["kappa"], top["score"], 100.0 * top["auc"],
            ), flush=True)
        except Exception as exc:
            print("  %s ERROR %s" % (tag, str(exc)[:120]), flush=True)

    if not candidates:
        raise RuntimeError("No H12 candidates produced for %s" % ds)

    oracle = max(candidates, key=lambda c: c["auc"])
    selectors = {"oracle_candidate": selector_summary("oracle_candidate", oracle)}
    dump_sink: Dict[str, np.ndarray] | None = (
        {} if getattr(args, "score_dump_dir", None) else None
    )

    target_name, target = target_pool(ds, candidates)
    add_selector_bundle(selectors, "h12_target", target, y, mask, sink=dump_sink)
    if "h12_target_top2_ks_rank" in selectors:
        selectors["h12_rule"] = dict(selectors["h12_target_top2_ks_rank"])
        selectors["h12_rule"]["selector"] = target_name
    elif "h12_target_best_ks" in selectors:
        selectors["h12_rule"] = dict(selectors["h12_target_best_ks"])
        selectors["h12_rule"]["selector"] = target_name

    family = local_affinity_family(ds, candidates)
    add_selector_bundle(selectors, "h12_family", family, y, mask, sink=dump_sink)
    cfg_pool = config_restricted_pool(family)
    add_selector_bundle(selectors, "h12_family_config", cfg_pool, y, mask, sink=dump_sink)
    h14_name, h14_pick = h14_select(ds, graph, candidates, y, mask)
    selectors["h14_rule"] = selector_summary(h14_name, h14_pick)
    if dump_sink is not None:
        dump_sink["sel/oracle_candidate"] = np.asarray(oracle["_scores"])
        if h14_pick is not None and "_scores" in h14_pick:
            dump_sink["sel/h14_rule"] = np.asarray(h14_pick["_scores"])

    selector_pool = [
        v for k, v in selectors.items()
        if k != "oracle_candidate" and "auc" in v
    ]
    if selector_pool:
        best_selector = max(selector_pool, key=lambda c: c["auc"])
        selectors["oracle_selector"] = dict(best_selector)
        selectors["oracle_selector"]["selector"] = "oracle_selector"

    print("  selectors: H10=%.2f%% H12=%.2f%% (%s) H14=%.2f%% (%s) oracle_score=%.2f%% baseline=%.2f%%" % (
        H10_SOFT.get(ds.lower(), float("nan")),
        100.0 * selectors.get("h12_rule", {"auc": float("nan")})["auc"],
        selectors.get("h12_rule", {"selector": "NA"})["selector"],
        100.0 * selectors.get("h14_rule", {"auc": float("nan")})["auc"],
        selectors.get("h14_rule", {"selector": "NA"})["selector"],
        100.0 * selectors["oracle_candidate"]["auc"],
        BASELINES.get(ds.lower(), float("nan")),
    ), flush=True)

    top_candidates = [
        selector_summary("candidate", c)
        for c in sorted(candidates, key=lambda r: r["auc"], reverse=True)[:80]
    ]
    if dump_sink is not None:
        for c in candidates:
            name = "cand/tmpl=%s,gamma=%g,pca=%s/%s" % (
                c.get("template_type"), c.get("gamma"), c.get("pca_dim"), c.get("score"),
            )
            if name not in dump_sink and "_scores" in c:
                dump_sink[name] = np.asarray(c["_scores"])
        names, values = [], []
        for name, arr in dump_sink.items():
            arr = np.asarray(arr, dtype=np.float64).ravel()
            if arr.shape[0] != len(y) or not np.any(np.isfinite(arr)):
                continue
            names.append(name)
            values.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        if names:
            if getattr(args, "out", None):
                stem = os.path.splitext(os.path.basename(args.out))[0]
            else:
                stem = "h12_%s_%s_%s" % (args.config_profile, args.modeled_subspace, ds)
            os.makedirs(args.score_dump_dir, exist_ok=True)
            path = os.path.join(args.score_dump_dir, "%s_scores.npz" % stem)
            np.savez_compressed(
                path,
                y=np.asarray(y, dtype=np.int64),
                mask=np.asarray(mask, dtype=bool),
                names=np.asarray(names),
                scores=np.vstack(values).T.astype(np.float32, copy=False),
            )
            print("  dumped score matrix %s shape=(%d,%d)" % (
                path, len(y), len(names)), flush=True)
    return {
        "dataset": ds,
        "graph": graph,
        "selectors": selectors,
        "top_candidates": top_candidates,
        "n_candidates": len(candidates),
        "transport_taus": taus,
        "lam_penalties": lam_penalties,
        "config_profile": args.config_profile,
        "references": {
            "h10_override_soft": H10_SOFT.get(ds.lower()),
            "best_non_eb_baseline": BASELINES.get(ds.lower()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modeled-subspace", default="legacy", choices=["legacy", "drop_nullspace"])
    parser.add_argument("--config-profile", default="coarse", choices=["coarse", "fine", "h14"])
    parser.add_argument("--transport-taus", default=",".join("%g" % x for x in DEFAULT_TAUS))
    parser.add_argument("--lam-penalties", default=",".join("%g" % x for x in DEFAULT_LAM_PENALTIES))
    parser.add_argument("--out", default=None)
    parser.add_argument("--score-dump-dir", default=None,
                        help="Dump candidate + selector score vectors as npz for AP computation.")
    parser.add_argument("--override-gammas", default=None,
                        help="Comma list overriding the per-dataset gamma grid.")
    parser.add_argument("--override-pcas", default=None,
                        help="Comma list overriding the per-dataset PCA grid (0 = none).")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.dataset.split(",") if item.strip()]
    payload = {ds: run_dataset(args, ds) for ds in datasets}
    os.makedirs("results", exist_ok=True)
    out = args.out or "results/h12_transport_filterbank_%s_%s.json" % (
        args.modeled_subspace, "_".join(datasets),
    )
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\nSaved %s" % out, flush=True)


if __name__ == "__main__":
    main()
