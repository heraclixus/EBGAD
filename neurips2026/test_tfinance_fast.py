"""T-Finance fast optimizer: sparsify dense graph + run all variants.

T-Finance has 42M edges on 39K nodes (avg degree ~1000). This script:
1. Sparsifies to top-k neighbors per node (k=50 by default)
2. Runs both prior_mean_mode = "zero" and "stationary"
3. Sweeps encoders: none, MLP, PCA, SAGE (skip GCN/GAT — too slow on dense graphs)
4. Saves per-variant results
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, copy
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, coordinate_descent_stationary,
    Q_rho_eigenvalues, compute_kappa_bounds,
)
from data_utils import load_data


def sparsify_topk(data, k=50):
    """Keep only top-k edges per node by edge weight (or random if unweighted)."""
    edge_index = data.edge_index
    n = data.x.shape[0]

    # Build adjacency list with cosine similarity weights
    src, dst = edge_index[0], edge_index[1]
    x_norm = data.x / data.x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    print("  Computing edge weights for sparsification...", flush=True)
    # Compute similarities in batches to avoid OOM
    sims = torch.zeros(src.shape[0])
    batch_size = 1000000
    for start in range(0, len(src), batch_size):
        end = min(start + batch_size, len(src))
        s, d = src[start:end], dst[start:end]
        sims[start:end] = (x_norm[s] * x_norm[d]).sum(dim=1)

    # For each node, keep top-k neighbors by similarity
    print("  Selecting top-%d neighbors per node..." % k, flush=True)
    keep_mask = torch.zeros(len(src), dtype=torch.bool)

    for node in range(n):
        node_mask = (src == node)
        node_indices = node_mask.nonzero(as_tuple=True)[0]
        if len(node_indices) <= k:
            keep_mask[node_indices] = True
        else:
            node_sims = sims[node_indices]
            topk_idx = node_sims.topk(k).indices
            keep_mask[node_indices[topk_idx]] = True

    new_edge_index = edge_index[:, keep_mask]

    data_sparse = copy.copy(data)
    data_sparse.edge_index = new_edge_index
    print("  Sparsified: %d -> %d edges (%.1f%%)" % (
        edge_index.shape[1], new_edge_index.shape[1],
        100.0 * new_edge_index.shape[1] / edge_index.shape[1]), flush=True)
    return data_sparse


def run_optimizer(data, ds, prior_mean_mode, encoder_configs, device="cuda"):
    """Run optimizer with given prior mode and encoder configs."""
    n = data.x.shape[0]
    d = data.x.shape[1]

    discrete_choices = []
    for enc in encoder_configs:
        for t in ["zero", "low", "high", "affinity"]:
            for g in ["original", "affinity"]:
                for gamma in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                    for norm in ["zscore", "minmax"]:
                        choice = {"template_type": t, "graph_type": g,
                                  "gamma": gamma, "normalize_mode": norm}
                        if enc is not None:
                            choice.update(enc)
                            choice["use_encoder"] = True
                        else:
                            choice["use_encoder"] = False
                        choices = choice
                        discrete_choices.append(choices)

    print("  %d discrete choices, prior_mean=%s" % (len(discrete_choices), prior_mean_mode), flush=True)

    best_ce = 0; best_cr = 0
    best_ce_cfg = {}; best_cr_cfg = {}

    for i, disc in enumerate(discrete_choices):
        trainer_kwargs = {
            "kappa": 1.0, "nu": 1.0, "rho": 0.5,
            "lam_penalty": 50.0, "alpha": 2.0, "T": 1.0,
            "template_type": disc["template_type"],
            "graph_type": disc["graph_type"],
            "gamma": disc["gamma"],
            "normalize_mode": disc["normalize_mode"],
            "prior_mean_mode": prior_mean_mode,
            "laplacian_variant": "sym",
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 0, "normalize_features": True,
        }
        if disc.get("use_encoder"):
            for k in ["use_encoder", "encoder_type", "encoder_hid_dim",
                       "encoder_num_layers", "encoder_dropout", "encoder_lr",
                       "encoder_epochs", "encoder_alpha"]:
                if k in disc:
                    trainer_kwargs[k] = disc[k]

        cfg = SOCTrainerConfig(**trainer_kwargs)
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=device)
        except Exception as e:
            continue

        x_normed = trainer.normalize(data.x.to(device).float())
        template = trainer.template
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
        delta = (x_normed - template).cpu().numpy()
        S = compute_spectral_energy(delta, V)

        best_opt = None; best_opt_ml = -np.inf
        for rho_init in [0.0, 0.5, 1.0]:
            for kappa_init in [0.1, 1.0, 5.0]:
                for pctile in [25, 50, 75]:
                    try:
                        if prior_mean_mode == "stationary":
                            opt = coordinate_descent_stationary(
                                S, lam_L, n, d,
                                rho_init=rho_init, kappa_init=kappa_init,
                                n_iters=10,
                                use_data_driven_bounds=True,
                                bound_percentile=pctile,
                            )
                        else:
                            opt = coordinate_descent(
                                S, lam_L, n, d,
                                rho_init=rho_init, kappa_init=kappa_init,
                                n_iters=10,
                                use_data_driven_bounds=True,
                                bound_percentile=pctile,
                            )
                        if opt["marginal_likelihood"] > best_opt_ml:
                            best_opt_ml = opt["marginal_likelihood"]
                            best_opt = opt
                    except:
                        pass

                # Unbounded fallback
                try:
                    if prior_mean_mode == "stationary":
                        opt = coordinate_descent_stationary(
                            S, lam_L, n, d,
                            rho_init=rho_init, kappa_init=kappa_init,
                            n_iters=10, use_data_driven_bounds=False,
                        )
                    else:
                        opt = coordinate_descent(
                            S, lam_L, n, d,
                            rho_init=rho_init, kappa_init=kappa_init,
                            n_iters=10, use_data_driven_bounds=False,
                        )
                    if opt["marginal_likelihood"] > best_opt_ml:
                        best_opt_ml = opt["marginal_likelihood"]
                        best_opt = opt
                except:
                    pass

        if best_opt is None:
            continue

        torch.manual_seed(0); np.random.seed(0)
        full_cfg = {
            "rho": best_opt["rho"], "kappa": best_opt["kappa"],
            "gamma": disc["gamma"], "alpha": 2.0, "T": 1.0,
            "lam_penalty": best_opt.get("lam_penalty", 50.0),
            "template_type": disc["template_type"],
            "graph_type": disc["graph_type"],
            "normalize_mode": disc["normalize_mode"],
            "prior_mean_mode": prior_mean_mode,
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 1, "parameterization": "score",
            "time_weighting": "importance",
            "laplacian_variant": "sym",
        }
        if disc.get("use_encoder"):
            for k in ["use_encoder", "encoder_type", "encoder_hid_dim",
                       "encoder_num_layers", "encoder_dropout", "encoder_lr",
                       "encoder_epochs", "encoder_alpha"]:
                if k in disc:
                    full_cfg[k] = disc[k]

        try:
            for method in ["precision_energy", "precision_ratio"]:
                eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                r = evaluate_single_trial(data, eval_cfg, device=device)
                if method == "precision_energy" and r.auc > best_ce:
                    best_ce = r.auc
                    best_ce_cfg = {**full_cfg, "score_method": "precision_energy"}
                if method == "precision_ratio" and r.auc > best_cr:
                    best_cr = r.auc
                    best_cr_cfg = {**full_cfg, "score_method": "precision_ratio"}
        except:
            pass

        if (i + 1) % 50 == 0:
            enc_tag = disc.get("encoder_type", "none") if disc.get("use_encoder") else "none"
            print("  [%d/%d enc=%s] best_ce=%.1f%% best_cr=%.1f%%" %
                  (i + 1, len(discrete_choices), enc_tag,
                   best_ce * 100, best_cr * 100), flush=True)

    best_overall = max(best_ce, best_cr)
    best_cfg = best_ce_cfg if best_ce >= best_cr else best_cr_cfg
    enc_tag = best_cfg.get("encoder_type", "none") if best_cfg.get("use_encoder") else "none"
    print("  BEST [%s]: %.1f%% (CE=%.1f%%, CR=%.1f%%) enc=%s" % (
        prior_mean_mode, best_overall * 100, best_ce * 100, best_cr * 100, enc_tag))
    return {"best_auc": best_overall, "best_ce": best_ce, "best_cr": best_cr,
            "best_config": best_cfg, "prior_mean_mode": prior_mean_mode}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--prior-mode", type=str, default="both",
                        choices=["zero", "stationary", "both"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading T-Finance ===", flush=True)
    raw_data = load_data("t_finance")
    print("  Raw: %d nodes, %d features, %d edges" % (
        raw_data.x.shape[0], raw_data.x.shape[1], raw_data.edge_index.shape[1]), flush=True)

    print("\n=== Sparsifying to top-%d ===" % args.topk, flush=True)
    data = sparsify_topk(raw_data, k=args.topk)

    # Encoder configs (skip GCN/GAT — too slow)
    encoder_configs = [
        None,  # no encoder
        {"use_encoder": True, "encoder_type": "mlp", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 8,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ]

    modes = ["zero", "stationary"] if args.prior_mode == "both" else [args.prior_mode]
    all_results = {}

    for mode in modes:
        out_file = os.path.join(args.output_dir, "eb_gad_tfinance_topk%d_%s.json" % (args.topk, mode))
        if os.path.exists(out_file):
            print("\n=== SKIP %s (exists) ===" % out_file, flush=True)
            with open(out_file) as f:
                all_results[mode] = json.load(f)
            continue

        print("\n=== T-Finance prior_mean=%s ===" % mode, flush=True)
        result = run_optimizer(data, "t_finance", mode, encoder_configs, device=args.device)
        result["dataset"] = "t_finance"
        result["topk"] = args.topk

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("  Saved to %s" % out_file)
        all_results[mode] = result

    print("\n=== SUMMARY ===")
    for mode, r in all_results.items():
        cfg = r["best_config"]
        enc = cfg.get("encoder_type", "none") if cfg.get("use_encoder") else "none"
        print("  %-12s best=%.1f%%  CE=%.1f%%  CR=%.1f%%  enc=%s" % (
            mode, r["best_auc"]*100, r["best_ce"]*100, r["best_cr"]*100, enc))


if __name__ == "__main__":
    main()
