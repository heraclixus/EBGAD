"""Small trained-encoder ablation for T-Finance and DGraph.

The earlier encoder sweep used broad grids and graph-structure reconstruction,
which is unnecessarily expensive on the two largest benchmarks.  This runner
keeps the encoder deliberately small, trains it with feature-only
reconstruction, reuses one graph eigenspace, and then applies the closed-form
EB-GAD score bank to the latent features.
"""

from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from bridge.encoder import EncoderConfig, train_autoencoder
from data_utils import extract_labeled_subgraph, load_data
from soc.prior_optimizer import compute_spectral_energy
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from test_dynamic_score_bank import (
    KAPPAS,
    canonical_horizons,
    entry_for_fit,
    evaluate_score_bank,
    fit_best_entry,
    format_horizon,
    horizon_score_bank,
    parse_grid,
    q_for_fit,
)
from test_dynamic_hypotheses import (
    dynamic_profile_selector_score_bank,
    latent_time_mixture_score_bank,
    path_dissipation_pvalue_bank,
    path_dissipation_score_bank,
    robust_path_aggregate_score_bank,
    two_groups_path_score_bank,
)


def parse_int_grid(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_names(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def dataset_load_name(dataset: str) -> str:
    if dataset == "t_finance":
        return "t_finance"
    if dataset == "dgraph":
        return "dgraph"
    raise ValueError("Unsupported dataset %r" % dataset)


def load_eval_data(dataset: str):
    data = load_data(dataset_load_name(dataset))
    if dataset == "dgraph":
        data, _ = extract_labeled_subgraph(data)
    return data


def compute_template(x: np.ndarray, V: np.ndarray, lam_L: np.ndarray, gamma: float) -> np.ndarray:
    smoother = 1.0 / np.maximum(float(gamma) ** 2 + lam_L, 1e-12)
    return V @ (smoother[:, None] * (V.T @ x))


def zscore(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x).astype(np.float32)


def load_or_compute_eigenspace(data, dataset: str, args):
    if dataset == "dgraph":
        cache_path = os.path.join(args.cache_dir, args.cache_file)
        print("Loading DGraph eigenspace cache %s" % cache_path, flush=True)
        cache = torch.load(cache_path, weights_only=False, map_location="cpu")
        V = cache["V"].detach().cpu().numpy().astype(np.float32, copy=False)
        lam_L = cache["lam_L"].detach().cpu().numpy().astype(np.float64, copy=False)
        if V.shape[0] != data.x.shape[0]:
            raise ValueError(
                "DGraph cache rows %d do not match labeled nodes %d"
                % (V.shape[0], data.x.shape[0])
            )
        return V, lam_L, {
            "source": "cache",
            "cache_dir": args.cache_dir,
            "cache_file": args.cache_file,
            "k": int(V.shape[1]),
        }

    print("Computing %s eigenspace once (k=%d)" % (dataset, args.truncated_k), flush=True)
    cfg = SOCTrainerConfig(
        kappa=1.0,
        nu=1.0,
        rho=0.5,
        lam_penalty=50.0,
        alpha=2.0,
        T=1.0,
        normalize_mode="zscore",
        prior_mean_mode="stationary",
        laplacian_variant="sym",
        d_hidden=16,
        n_hidden_layers=1,
        epochs=0,
        normalize_features=True,
        template_type="low",
        graph_type="original",
        gamma=0.5,
        verbose=False,
        truncated_k=args.truncated_k,
        modeled_subspace=args.modeled_subspace,
    )
    trainer = SOCTrainer(cfg)
    trainer.train(data, device=args.device)
    V = trainer.V.detach().cpu().numpy().astype(np.float32, copy=False)
    lam_L = trainer.lam_L_model.detach().cpu().numpy().astype(np.float64, copy=False)
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return V, lam_L, {"source": "computed", "k": int(V.shape[1])}


def build_entries(
    z: np.ndarray,
    V: np.ndarray,
    lam_L: np.ndarray,
    template_gammas: Iterable[float],
) -> List[Dict]:
    entries = []
    n, d = z.shape
    for gamma in template_gammas:
        template = compute_template(z, V, lam_L, gamma)
        delta = z - template
        S = compute_spectral_energy(delta, V)
        entries.append({
            "key": "original/gamma=%g" % gamma,
            "graph_type": "original",
            "template_gamma": float(gamma),
            "V": V,
            "lam_L": lam_L,
            "delta": delta,
            "S": S,
            "n": int(n),
            "d": int(d),
        })
    return entries


def best_auc(summary: Dict) -> float:
    return float(summary.get("auc_oracle", summary.get("best_auc", 0.0)))


def summarize_scores(entries: List[Dict], horizons: List[float], kappas: List[float], y: np.ndarray, mask: np.ndarray, args) -> Dict:
    stationary_fit = fit_best_entry(entries, [np.inf], kappas, use_gamma_jacobian=False)
    stationary_entry = entry_for_fit(entries, stationary_fit)
    stationary_q = q_for_fit(stationary_entry, stationary_fit)
    stationary_scores = horizon_score_bank(stationary_entry, stationary_q, [np.inf])
    stationary_summary = evaluate_score_bank(y, mask, stationary_scores, references=None)

    bank_scores = horizon_score_bank(stationary_entry, stationary_q, horizons)
    bank_summary = evaluate_score_bank(y, mask, bank_scores, references=None)

    finite_horizons = [h for h in horizons if np.isfinite(h)]
    combined_scores = dict(bank_scores)
    combined_pvalues = {}
    extra = {}
    if finite_horizons:
        path_scores, path_meta = path_dissipation_score_bank(
            stationary_entry,
            stationary_q,
            finite_horizons,
            n_quad=args.path_quad,
            tail_factor=args.path_tail_factor,
            tail_cap=args.path_tail_cap,
        )
        path_pvalues = path_dissipation_pvalue_bank(
            stationary_entry,
            stationary_q,
            finite_horizons,
            path_scores,
            tail_factor=args.path_tail_factor,
            tail_cap=args.path_tail_cap,
        )
        robust_scores, robust_meta = robust_path_aggregate_score_bank(
            path_scores,
            path_pvalues,
            min_bands=args.robust_min_bands,
            max_bands=args.robust_max_bands,
        )
        two_group_scores, two_group_meta = two_groups_path_score_bank(
            path_pvalues,
            pi_init=args.tg_pi_init,
            pi_max=args.tg_pi_max,
            weight_alpha=args.tg_weight_alpha,
        )
        mixture_scores, mixture_meta = latent_time_mixture_score_bank(
            stationary_entry,
            stationary_q,
            horizons,
            dirichlet_alpha=args.mixture_alpha,
        )
        profile_scores, profile_pvalues, profile_meta = dynamic_profile_selector_score_bank(
            path_scores,
            path_pvalues,
            mixture_scores,
            two_group_scores,
            stationary_scores,
        )
        combined_scores.update(path_scores)
        combined_scores.update(robust_scores)
        combined_scores.update(two_group_scores)
        combined_scores.update(mixture_scores)
        combined_scores.update(profile_scores)
        combined_pvalues.update(path_pvalues)
        combined_pvalues.update(profile_pvalues)
        extra = {
            "path": evaluate_score_bank(y, mask, path_scores, references=None, pvalue_bank=path_pvalues),
            "robust_path": evaluate_score_bank(y, mask, robust_scores, references=None),
            "two_groups": evaluate_score_bank(y, mask, two_group_scores, references=None),
            "latent_mixture": evaluate_score_bank(y, mask, mixture_scores, references=None),
            "dynamic_profile": evaluate_score_bank(
                y, mask, profile_scores, references=None, pvalue_bank=profile_pvalues,
            ),
            "path_meta": path_meta,
            "robust_meta": robust_meta,
            "two_group_meta": two_group_meta,
            "mixture_meta": mixture_meta,
            "profile_meta": profile_meta,
        }

    combined_summary = evaluate_score_bank(
        y, mask, combined_scores, references=None, pvalue_bank=combined_pvalues or None,
    )

    return {
        "stationary_fit": {
            "rho": float(stationary_fit["rho"]),
            "kappa": float(stationary_fit["kappa"]),
            "ml": float(stationary_fit["ml"]),
            "entry_key": stationary_fit["entry_key"],
        },
        "stationary": stationary_summary,
        "horizon_bank": bank_summary,
        "dynamic_combined": combined_summary,
        **extra,
    }


def train_and_encode(data, enc_cfg: Dict, device: str) -> np.ndarray:
    cfg = EncoderConfig(
        encoder_type=enc_cfg["encoder_type"],
        hid_dim=int(enc_cfg["hid_dim"]),
        num_layers=int(enc_cfg["num_layers"]),
        dropout=float(enc_cfg["dropout"]),
        lr=float(enc_cfg["lr"]),
        epochs=int(enc_cfg["epochs"]),
        alpha=1.0,
        weight_decay=float(enc_cfg["weight_decay"]),
        verbose=True,
        log_every=max(1, int(enc_cfg["epochs"]) // 2),
    )
    model = train_autoencoder(data, cfg, device=device)
    x = data.x.to(device).float()
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        z = model.encode(x, edge_index).detach().cpu().numpy().astype(np.float32, copy=False)
    del model, x, edge_index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return zscore(z)


def encoder_grid(args) -> List[Dict]:
    configs = []
    encoders = parse_names(args.encoders)
    for enc_type in encoders:
        for hid_dim in parse_int_grid(args.hidden_dims):
            for num_layers in parse_int_grid(args.layers):
                for epochs in parse_int_grid(args.epochs_grid):
                    configs.append({
                        "encoder_type": enc_type,
                        "hid_dim": int(hid_dim),
                        "num_layers": int(num_layers),
                        "epochs": int(epochs),
                        "dropout": float(args.dropout),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                    })
    return configs


def run(args) -> Dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_eval_data(args.dataset)
    n, d = data.x.shape
    y = data.y.detach().cpu().numpy()
    mask = (y >= 0) & (y <= 1)
    print(
        "%s: n=%d d=%d edges=%d anomalies=%.2f%%"
        % (args.dataset, n, d, data.edge_index.shape[1], 100.0 * float(np.mean(y[mask] == 1))),
        flush=True,
    )

    V, lam_L, eig_meta = load_or_compute_eigenspace(data, args.dataset, args)
    print("Eigenspace V=%s lam=%s" % (tuple(V.shape), tuple(lam_L.shape)), flush=True)

    configs = encoder_grid(args)
    configs = [cfg for i, cfg in enumerate(configs) if i % args.n_chunks == args.chunk]
    print("Chunk %d/%d has %d encoder configs" % (args.chunk, args.n_chunks, len(configs)), flush=True)

    template_gammas = parse_grid(args.template_gammas)
    horizons = canonical_horizons(parse_grid(args.horizons), include_inf=True)
    kappas = parse_grid(args.kappas)

    result = {
        "dataset": args.dataset,
        "chunk": int(args.chunk),
        "n_chunks": int(args.n_chunks),
        "graph": {
            "n": int(n),
            "d_raw": int(d),
            "edges": int(data.edge_index.shape[1]),
            "eigenspace": eig_meta,
        },
        "grid": {
            "template_gammas": [float(x) for x in template_gammas],
            "horizons": [format_horizon(x) for x in horizons],
            "kappas": [float(x) for x in kappas],
            "encoder_configs_total": len(encoder_grid(args)),
        },
        "runs": [],
        "best": {"auc": 0.0, "config": None, "summary_key": None},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        "%s_encoder_small_chunk%d.json" % (args.dataset, args.chunk),
    )

    for idx, enc_cfg in enumerate(configs, 1):
        print("\n=== [%d/%d] %s ===" % (idx, len(configs), enc_cfg), flush=True)
        try:
            z = train_and_encode(data, enc_cfg, args.device)
            entries = build_entries(z, V, lam_L, template_gammas)
            summaries = summarize_scores(entries, horizons, kappas, y, mask, args)
            run_record = {
                "encoder": enc_cfg,
                "latent_dim": int(z.shape[1]),
                "summaries": summaries,
            }
            result["runs"].append(run_record)

            for key in ("stationary", "horizon_bank", "dynamic_combined", "path", "latent_mixture", "dynamic_profile"):
                if key in summaries:
                    auc = best_auc(summaries[key])
                    if auc > result["best"]["auc"]:
                        result["best"] = {
                            "auc": auc,
                            "config": enc_cfg,
                            "summary_key": key,
                            "summary": summaries[key],
                        }
                        print("  NEW BEST %.2f%% via %s" % (100.0 * auc, key), flush=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  ERROR: %s" % str(exc)[:240], flush=True)
            result["runs"].append({"encoder": enc_cfg, "error": str(exc)})
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved %s" % out_path, flush=True)
    print("Best %.2f%%" % (100.0 * result["best"]["auc"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["t_finance", "dgraph"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--encoders", default="mlp,sage")
    parser.add_argument("--hidden-dims", default="8,16")
    parser.add_argument("--layers", default="2")
    parser.add_argument("--epochs-grid", default="20,50")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--template-gammas", default="0.5")
    parser.add_argument("--horizons", default="0.1,1,10,inf")
    parser.add_argument("--kappas", default=",".join("%g" % k for k in KAPPAS))
    parser.add_argument("--modeled-subspace", default="legacy")
    parser.add_argument("--truncated-k", type=int, default=500)
    parser.add_argument("--cache-dir", default="cache/dgraph_eigen")
    parser.add_argument("--cache-file", default="labeled_original_k256.pt")
    parser.add_argument("--path-quad", type=int, default=2)
    parser.add_argument("--path-tail-factor", type=float, default=8.0)
    parser.add_argument("--path-tail-cap", type=float, default=100.0)
    parser.add_argument("--mixture-alpha", type=float, default=1.05)
    parser.add_argument("--tg-pi-init", type=float, default=0.05)
    parser.add_argument("--tg-pi-max", type=float, default=0.20)
    parser.add_argument("--tg-weight-alpha", type=float, default=1.05)
    parser.add_argument("--robust-min-bands", type=int, default=2)
    parser.add_argument("--robust-max-bands", type=int, default=6)
    parser.add_argument("--out-dir", default="results/large_encoder_small_20260501")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
