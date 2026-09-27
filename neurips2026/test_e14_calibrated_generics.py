"""E14: can EB calibration rescue the generic spectral families? (+ AP for E2)

The E2 isolation ablation showed the GOU bank beating the heat/polynomial
banks by 10-14pp on Elliptic, Elliptic++ and T-Finance under identical
label-free selection, and attributed the gap to calibration: the GOU family
has a plug-in chi-squared null that makes NullKS informative. The natural
counter-question: give the generic families the
same calibration machinery and see if the gap survives.

This script EB-fits each generic family with the same marginal likelihood
used for (rho, kappa): for every candidate config (heat scale t, polynomial
(c, p)), a global precision scale nu is profiled in closed form
(nu* = d*k / (S . p)) and the same PoE marginal log-likelihood is maximized
over the same template-entry grid the GOU fit scans. Two calibrated variants
are then scored with the identical label-free selector:

  cal_strict    bank = {J, R} at the ML-fitted config: the fit commits to a
                model, NullKS only picks the score type (exactly how the GOU
                fit commits to (rho, kappa) before the bank is built)
  cal_dynamics  the fitted generic precision nu*q is treated as the
                stationary precision of its canonical GOU process, and the
                paper's full finite-horizon + transport bank is built from
                it; only the stationary precision SHAPE now differs from
                GOU-EB

Both the raw and nu-profiled ML values are reported for every family so the
fit quality of each null model is directly comparable.

Also emits AUPRC (AP) for every family's label-free pick and the per-bank
AP oracle: the E2 isolation table in average precision.
"""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import average_precision_score

from data_utils import load_data
from test_dynamic_score_bank import (
    aggregate_rank_scores,
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    horizon_score_bank,
    node_unsupervised_references,
    prepare_model_entries,
    q_for_fit,
)
from test_e2_isolation_banks import (
    DS_CONFIG,
    ERA_HORIZONS,
    ERA_KAPPAS,
    HEAT_TS,
    POLY_GRID,
    family_banks,
    generic_bank,
    transport_scores,
)
from test_multiscale_gou_poe import (
    optimize_poe_weights,
    poe_marginal_log_likelihood,
)

TRIO = ["elliptic", "elliptic_plus_plus", "t_finance"]


def heat_configs(lam: np.ndarray) -> dict[str, np.ndarray]:
    lam_scale = float(np.max(lam)) or 1.0
    return {"t=%g" % t: np.exp(t * lam / lam_scale) for t in HEAT_TS}


def poly_configs(lam: np.ndarray) -> dict[str, np.ndarray]:
    return {"c=%g,p=%d" % (c, p): np.power(c + lam, p) for c, p in POLY_GRID}


def profiled_nu_ml(S: np.ndarray, p: np.ndarray, d: int) -> tuple[float, float]:
    """Closed-form global-scale profile of the PoE marginal log-likelihood."""
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-12)
    k = len(p)
    sp = float(np.dot(S, p))
    if sp <= 0:
        return -np.inf, 1.0
    nu = (d * k) / sp
    return poe_marginal_log_likelihood(S, nu * p, d), float(nu)


def fit_generic_family(
    entries, family: str,
) -> dict:
    """EB-fit a generic family: best (entry, config) by nu-profiled ML.

    Mirrors fit_best_entry: the same entry grid, the same ML objective; the
    family's config grid plays the role of the (rho, kappa) grids.
    """
    best = {"ml": -np.inf}
    ml_poe_best = -np.inf
    for entry in entries:
        lam = np.asarray(entry["lam_L"], dtype=np.float64)
        cfgs = heat_configs(lam) if family == "heat" else poly_configs(lam)
        S, d = entry["S"], entry["d"]
        for name, p in cfgs.items():
            ml, nu = profiled_nu_ml(S, p, d)
            if ml > best["ml"]:
                best = {"ml": float(ml), "nu": nu, "config": name,
                        "entry_key": entry["key"], "q": nu * np.maximum(p, 1e-12)}
        # PoE simplex weights over the whole config grid (diagnostic only)
        tau = np.vstack([np.maximum(cfgs[n], 1e-12) for n in sorted(cfgs)])
        try:
            w, _ = optimize_poe_weights(S, tau, d)
            ml_poe, _ = profiled_nu_ml(S, w @ tau, d)
        except Exception:
            ml_poe = float("nan")
        if np.isfinite(ml_poe) and ml_poe > ml_poe_best:
            ml_poe_best = float(ml_poe)
    best["ml_poe"] = ml_poe_best
    return best


def bank_candidates(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reproduce evaluate_score_bank's classic candidate pool (bank + rank aggregates)."""
    cand = dict(bank)
    cand.update(aggregate_rank_scores(bank))
    return cand


def ap_metrics(y, mask, bank, summary) -> dict:
    cand = bank_candidates(bank)
    y_eval = np.asarray(y)[mask]
    out = {}
    sel = summary.get("score_ks")
    if sel in cand:
        s = np.nan_to_num(np.asarray(cand[sel], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        out["ap_ks_selected"] = float(average_precision_score(y_eval, s[mask]))
    best_ap, best_name = -1.0, None
    for name, s in cand.items():
        s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmax(s) <= np.nanmin(s):
            continue
        ap = float(average_precision_score(y_eval, s[mask]))
        if ap > best_ap:
            best_ap, best_name = ap, name
    out["ap_oracle"] = best_ap
    out["ap_oracle_score"] = best_name
    return out


def eval_bank(y, mask, bank, references, label: str) -> dict:
    summary = evaluate_score_bank(y, mask, bank, references=references)
    row = {
        "n_scores": summary.get("n_scores"),
        "auc_ks_selected": summary.get("auc_ks_selected"),
        "score_ks": summary.get("score_ks"),
        "auc_oracle": summary.get("auc_oracle"),
        "oracle_score": summary.get("oracle_score"),
    }
    row.update(ap_metrics(y, mask, bank, summary))
    print("  %-22s sel(KS)=%5.1f%% (%s)  AP(sel)=%5.1f%%  oracle=%5.1f%% (%s)  AP(oracle)=%5.1f%% (%s)" % (
        label,
        100.0 * (row["auc_ks_selected"] or 0), row["score_ks"],
        100.0 * (row.get("ap_ks_selected") or 0),
        100.0 * (row["auc_oracle"] or 0), row["oracle_score"],
        100.0 * (row.get("ap_oracle") or 0), row.get("ap_oracle_score"),
    ), flush=True)
    return row


def run_dataset(ds: str, device: str) -> dict:
    cfg = DS_CONFIG[ds]
    print("\n=== %s | %s ===" % (ds, cfg["sub"]), flush=True)
    data = load_data(ds)
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    pca_dim = cfg.get("pca", 64 if data.x.shape[1] > 100 else None)

    entries = prepare_model_entries(
        data,
        graph_types=["original"],
        template_gammas=cfg["gammas"],
        pca_dim=pca_dim,
        truncated_k=cfg["tk"],
        modeled_subspace=cfg["sub"],
        template_type=cfg["tmpl"],
        device=device,
    )
    fit = fit_best_entry(entries, ERA_HORIZONS, ERA_KAPPAS, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    references = node_unsupervised_references(data, entry["delta"])

    # nu-profiled ML of the GOU fit's PoE-mixed precision, for a like-for-like
    # ML comparison with the generic fits
    ml_gou_nu, _ = profiled_nu_ml(entry["S"], fit["precision"], entry["d"])
    print("  gou fit: %s rho=%.3f kappa=%g ml=%.0f ml_nu=%.0f" % (
        fit["entry_key"], fit["rho"], fit["kappa"], fit["ml"], ml_gou_nu), flush=True)

    out = {
        "dataset": ds,
        "fit": {k: fit[k] for k in ("entry_key", "rho", "kappa", "ml")},
        "ml_gou_nu_profiled": float(ml_gou_nu),
        "families": {},
        "calibrated": {},
    }

    # 1) the five uncalibrated E2 families, now with AP (E4/AP addendum)
    for family, bank in family_banks(entry, fit).items():
        out["families"][family] = eval_bank(y, mask, bank, references, family)

    # 2) calibrated generic families
    for family in ("heat", "poly"):
        gfit = fit_generic_family(entries, family)
        gentry = entry_for_fit(entries, gfit)
        grefs = (references if gentry["key"] == entry["key"]
                 else node_unsupervised_references(data, gentry["delta"]))
        print("  %s calibrated fit: %s config=%s ml=%.0f ml_poe=%.0f nu=%.3g%s" % (
            family, gfit["entry_key"], gfit["config"], gfit["ml"],
            gfit.get("ml_poe", float("nan")), gfit["nu"],
            "" if gentry["key"] == entry["key"] else "  [own entry]"), flush=True)
        q_cal = gfit["q"]

        strict = generic_bank(gentry, {gfit["config"]: q_cal})
        dyn = horizon_score_bank(gentry, q_cal, ERA_HORIZONS)
        dyn.update(transport_scores(gentry, q_cal))

        out["calibrated"][family] = {
            "fit": {k: gfit[k] for k in ("entry_key", "config", "ml", "ml_poe", "nu")},
            "cal_strict": eval_bank(y, mask, strict, grefs, "%s cal_strict" % family),
            "cal_dynamics": eval_bank(y, mask, dyn, grefs, "%s cal_dynamics" % family),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(TRIO))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/e14/e14_calibrated.json")
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
