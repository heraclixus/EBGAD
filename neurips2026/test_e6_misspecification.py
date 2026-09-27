"""E6: misspecification stress test.

Synthetic SBM graphs where the residual distribution violates the Gaussian
model in controlled ways: heavy tails (Student-t, df=3), multimodal normal
population (two-component mean mixture), and categorical (binarized) features.
Anomalies are additive latent-noise attribute anomalies at 5% (bit flips for
the categorical_flip variant). Same pipeline, same era grids, same label-free
selection for every variant; only the data distribution moves.
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
import torch
from types import SimpleNamespace

from bridge.synthetic import _stochastic_block_model
from bridge.graph_ops import compute_laplacian
from test_dynamic_score_bank import (
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    horizon_score_bank,
    node_unsupervised_references,
    prepare_model_entries,
    q_for_fit,
)

ERA_HORIZONS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, np.inf]
ERA_KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5,
              1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]


def make_data(variant: str, n: int, d: int, frac_anom: float, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    edge_index = _stochastic_block_model(n)
    L = compute_laplacian(edge_index, n, variant="sym")
    lam, V = torch.linalg.eigh(L)
    # smooth normal template: low-pass filtered white noise
    g = 1.0 / (1.0 + 5.0 * lam)
    m = V @ (g[:, None] * (V.T @ torch.randn(n, d)))
    m = m / m.std()

    sigma = 0.5
    if variant == "gaussian":
        z = m + sigma * torch.randn(n, d)
    elif variant == "heavy_tail":
        t = torch.distributions.StudentT(df=3.0).sample((n, d))
        z = m + sigma * t / np.sqrt(3.0)
    elif variant == "multimodal":
        modes = torch.tensor(rng.integers(0, 2, size=n), dtype=torch.float32)
        shift = torch.randn(1, d)
        z = m + (2 * modes - 1)[:, None] * shift + sigma * torch.randn(n, d)
    elif variant in ("categorical", "categorical_flip"):
        z = m + sigma * torch.randn(n, d)
    else:
        raise ValueError(variant)

    y = np.zeros(n, dtype=np.int64)
    n_anom = int(round(frac_anom * n))
    anom = rng.choice(n, size=n_anom, replace=False)
    y[anom] = 1
    z = z.clone()
    if variant == "categorical_flip":
        # anomalies = random bit flips on binarized features (detectable signal)
        x = (z > 0).float()
        flips = torch.tensor(rng.random((n_anom, d)) < 0.3, dtype=torch.bool)
        xa = x[torch.tensor(anom)]
        xa[flips] = 1.0 - xa[flips]
        x[torch.tensor(anom)] = xa
    else:
        # benchmark-standard injection: additive unstructured noise on the latent
        z[anom] += 2.0 * torch.randn(n_anom, d)
        x = (z > 0).float() if variant == "categorical" else z
    return SimpleNamespace(x=x, edge_index=edge_index,
                           y=torch.tensor(y), num_nodes=n)


def run(variant: str, n: int, d: int, frac: float, seed: int) -> dict:
    data = make_data(variant, n, d, frac, seed)
    y = data.y.numpy()
    mask = np.ones(len(y), bool)
    entries = prepare_model_entries(
        data, graph_types=["original"], template_gammas=[0.5, 1.0, 2.0],
        pca_dim=None, truncated_k=None, modeled_subspace="drop_nullspace",
        template_type="low", device="cpu",
    )
    fit = fit_best_entry(entries, ERA_HORIZONS, ERA_KAPPAS, use_gamma_jacobian=False)
    entry = entry_for_fit(entries, fit)
    q = q_for_fit(entry, fit)
    bank = horizon_score_bank(entry, q, ERA_HORIZONS)
    refs = node_unsupervised_references(data, entry["delta"])
    s = evaluate_score_bank(y, mask, bank, references=refs)
    return {"variant": variant, "seed": seed, "rho": fit["rho"], "kappa": fit["kappa"],
            "auc_sel": s.get("auc_ks_selected"), "score": s.get("score_ks"),
            "auc_oracle": s.get("auc_oracle")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants",
                    default="gaussian,heavy_tail,multimodal,categorical,categorical_flip")
    ap.add_argument("--out", default="results/e6/e6_misspec.json")
    args = ap.parse_args()

    rows = []
    for variant in args.variants.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = run(variant, args.n, args.d, args.frac, seed)
            rows.append(r)
            print("%-12s seed=%d rho=%.3f kappa=%-6g sel=%.1f%% (%s) oracle=%.1f%%" % (
                variant, seed, r["rho"], r["kappa"], 100 * (r["auc_sel"] or 0),
                r["score"], 100 * (r["auc_oracle"] or 0)), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
    print("Saved %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
