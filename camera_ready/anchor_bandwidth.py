"""Equilibrium anchor at every bandwidth of a dataset's grid.

For each bandwidth the template and residual are built exactly as in the pipeline (one SOCTrainer per
bandwidth, as in prepare_model_entries), (rho, kappa) are fitted by the equilibrium likelihood and the
two equilibrium scores are written to a score dump. The JSON holds the unlabeled quantities that a
bandwidth rule may read (likelihood, NullKS, share of the feature energy that the residual map
removes) and the AUROC of each score. select_table1.py builds the anchor from these files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
from data_utils import load_data  # noqa: E402
from soc.prior_optimizer import compute_spectral_energy  # noqa: E402
from soc.soc_trainer import SOCTrainer  # noqa: E402
from test_dynamic_score_bank import (  # noqa: E402
    build_trainer_config, compute_graph_stats, density_adjusted_template_gammas, evaluate_score_bank,
    fit_best_entry, horizon_score_bank, node_unsupervised_references, q_for_fit, safe_auc,
)
from test_multiscale_gou_poe import (  # noqa: E402
    ks_null_deviation, nearest_template_gamma, template_gamma_center,
)

KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
NAMES = {"yelpchi": "YelpChi", "facebook": "Facebook"}
DGRAPH_GRID = [0.5, 1.0, 2.0]  # grid of the cached DGraph pipeline


def truncation(n):
    return None if n < 20_000 else (128 if n >= 1_000_000 else 200 if n >= 500_000 else 300 if n >= 100_000 else 500)


def entries_generic(data, gammas, pca, k):
    out = []
    for g in gammas:
        print("    bandwidth %g" % g, flush=True)
        trainer = SOCTrainer(build_trainer_config(data, gamma=g, pca_dim=pca, truncated_k=k,
                                                  modeled_subspace="legacy", graph_type="original",
                                                  template_type="low"))
        trainer.train(data, device="cpu")
        V = trainer.V.detach().cpu().numpy()
        lam = trainer.lam_L_model.detach().cpu().numpy()
        x = trainer.x_T.detach().cpu().numpy()
        delta = x - trainer.template.detach().cpu().numpy()
        n, d = x.shape
        out.append(({"key": "original/gamma=%g" % g, "graph_type": "original", "template_gamma": float(g), "V": V,
                     "lam_L": lam, "delta": delta, "S": compute_spectral_energy(delta, V), "n": int(n), "d": int(d)},
                    float(np.sum(x.astype(np.float64) ** 2))))
    return out


def entries_dgraph(cache_dir, gammas):
    import torch
    from test_dynamic_hypotheses_dgraph_cached import build_entries, normalize_features
    data = load_data("dgraph")
    y = data.y.detach().cpu().numpy()
    idx = np.where((y >= 0) & (y <= 1))[0]
    x = normalize_features(data.x.detach().cpu().float().numpy(), "zscore")[idx]
    cache = torch.load(os.path.join(cache_dir, "labeled_original_k128.pt"), weights_only=False, map_location="cpu")
    V = cache["V"].detach().cpu().numpy().astype(np.float32, copy=False)
    lam = cache["lam_L"].detach().cpu().numpy().astype(np.float64, copy=False)
    total = float(np.sum(x.astype(np.float64) ** 2))
    return [(e, total) for e in build_entries(x, V, lam, gammas, "labeled_original_k128")], data, y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", default="results/camera_ready/anchor")
    ap.add_argument("--dgraph-cache-dir", default=None)
    args = ap.parse_args()
    ds = args.dataset
    seed = os.environ.get("EBGAD_RELABEL_SEED")
    tag = "relabel%s" % seed if seed else "orig"
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "anchor_%s_%s" % (ds, tag))

    if ds == "dgraph":
        gammas = list(DGRAPH_GRID)
        cache_dir = args.dgraph_cache_dir or ("cache/dgraph_eigen_relabel%s" % seed if seed else "cache/dgraph_eigen")
        ents, data, y = entries_dgraph(cache_dir, gammas)
        full = None
        os.environ.pop("EBGAD_RELABEL_SEED", None)
        h, de = compute_graph_stats(load_data("dgraph"))
    else:
        data = load_data(NAMES.get(ds, ds))
        y = data.y.detach().cpu().numpy()
        h, de = compute_graph_stats(data)
        gammas = sorted(set(float(g) for g in density_adjusted_template_gammas(h, de))
                        | {nearest_template_gamma(h, de)})
        n = data.x.shape[0]
        ents = entries_generic(data, gammas, 64 if data.x.shape[1] > 100 else None, truncation(n))
        full = data
    mask = (y >= 0) & (y <= 1)

    names, cols, rows = [], [], []
    for e, total in ents:
        g = e["template_gamma"]
        fit = fit_best_entry([e], [np.inf], KAPPAS, False)
        bank = horizon_score_bank(e, q_for_fit(e, fit), [np.inf])
        J, R = bank["J@inf"], bank["R@inf"]
        Jd, Rd = np.nan_to_num(J.astype(np.float64)), np.nan_to_num(R.astype(np.float64))
        # the equilibrium bank of the pipeline at this bandwidth (J*, R and their rank aggregates)
        refs = node_unsupervised_references(full, e["delta"]) if full is not None else None
        summary = evaluate_score_bank(y, mask, {"J@inf": J, "R@inf": R}, references=refs)
        names += ["bw/J@g=%g" % g, "bw/R@g=%g" % g]
        cols += [J.astype(np.float32), R.astype(np.float32)]
        rows.append({"gamma": g, "ml": float(fit["ml"]), "rho": float(fit["rho"]), "kappa": float(fit["kappa"]),
                     "removed": 1.0 - float(np.sum(e["delta"].astype(np.float64) ** 2)) / total,
                     "ks_J": float(ks_null_deviation(Jd)), "ks_R": float(ks_null_deviation(Rd)),
                     "auc_J": 100 * safe_auc(y[mask], Jd[mask]), "auc_R": 100 * safe_auc(y[mask], Rd[mask])})
        print(ds, tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        rows[-1]["bank"] = summary
    np.savez_compressed(stem + "_scores.npz", names=np.array(names), scores=np.stack(cols, axis=1), y=y, mask=mask)
    json.dump({"dataset": ds, "order": tag, "h": float(h), "d_e": float(de),
               "gamma_center": template_gamma_center(h, de), "gamma_nearest": nearest_template_gamma(h, de),
               "gamma_likelihood": max(rows, key=lambda r: r["ml"])["gamma"], "grid": gammas, "rows": rows},
              open(stem + ".json", "w"), indent=1, default=float)
    print("Saved", stem + ".json")


if __name__ == "__main__":
    main()
