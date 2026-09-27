"""Prior optimizer v6: PCA, GAT, GraphSAGE encoders.

Complements v5 (which covers no-encoder, MLP, GCN). v6 adds:
- PCA: deterministic, no training, orthogonal latent features
- GAT: attention-weighted message passing
- GraphSAGE: sampling-based, scales to large graphs

Same optimizer pipeline as v5: encoder → spectral energies → coordinate descent.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, coordinate_descent, Q_rho_eigenvalues,
    compute_kappa_bounds,
)
from data_utils import load_data


def build_discrete_choices(use_large_datasets=False):
    """Build discrete choice grid for new encoder types."""
    choices = []

    encoder_options = [
        # PCA variants (deterministic, fast)
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 16,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 32,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        {"use_encoder": True, "encoder_type": "pca", "encoder_hid_dim": 64,
         "encoder_num_layers": 1, "encoder_dropout": 0.0, "encoder_lr": 0.0,
         "encoder_epochs": 0, "encoder_alpha": 1.0, "encoder_weight_decay": 0.0},
        # GAT variants
        {"use_encoder": True, "encoder_type": "gat", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "gat", "encoder_hid_dim": 32,
         "encoder_num_layers": 4, "encoder_dropout": 0.3, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "gat", "encoder_hid_dim": 64,
         "encoder_num_layers": 4, "encoder_dropout": 0.3, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.5, "encoder_weight_decay": 0.01},
    ]

    # GraphSAGE: add for all datasets (scales well)
    encoder_options.extend([
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 32,
         "encoder_num_layers": 2, "encoder_dropout": 0.1, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
        {"use_encoder": True, "encoder_type": "sage", "encoder_hid_dim": 64,
         "encoder_num_layers": 4, "encoder_dropout": 0.3, "encoder_lr": 0.005,
         "encoder_epochs": 300, "encoder_alpha": 0.8, "encoder_weight_decay": 0.01},
    ])

    if use_large_datasets:
        # Drop GAT for large datasets (OOM risk), keep PCA and SAGE
        encoder_options = [e for e in encoder_options if e["encoder_type"] != "gat"]

    templates = ["zero", "low", "high", "affinity"]
    graphs = ["original", "affinity"]
    gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    norms = ["zscore", "minmax"]

    for enc in encoder_options:
        for t in templates:
            for g in graphs:
                for gamma in gammas:
                    for norm in norms:
                        choice = {"template_type": t, "graph_type": g,
                                  "gamma": gamma, "normalize_mode": norm}
                        choice.update(enc)
                        choices.append(choice)

    return choices


def run_dataset(ds, device="cuda"):
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    n = data.x.shape[0]
    d = data.x.shape[1]

    use_large = ds.lower() in ("dgraph", "t_finance")
    discrete_choices = build_discrete_choices(use_large_datasets=use_large)

    enc_types = set(c["encoder_type"] for c in discrete_choices)
    print("  %d discrete choices, encoder types: %s" % (
        len(discrete_choices), ", ".join(sorted(enc_types))), flush=True)

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
            "laplacian_variant": "sym",
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 0, "normalize_features": True,
            "use_encoder": True,
            "encoder_type": disc["encoder_type"],
            "encoder_hid_dim": disc["encoder_hid_dim"],
            "encoder_num_layers": disc["encoder_num_layers"],
            "encoder_dropout": disc["encoder_dropout"],
            "encoder_lr": disc["encoder_lr"],
            "encoder_epochs": disc["encoder_epochs"],
            "encoder_alpha": disc["encoder_alpha"],
        }

        cfg = SOCTrainerConfig(**trainer_kwargs)
        trainer = SOCTrainer(cfg)
        try:
            trainer._setup(data, device=device)
        except Exception as e:
            if (i + 1) % 100 == 0:
                print("    [%d/%d] setup failed: %s" % (i+1, len(discrete_choices), e), flush=True)
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

                try:
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
            "gamma": disc["gamma"], "alpha": best_opt["alpha_T"], "T": 1.0,
            "lam_penalty": best_opt["lam_penalty"],
            "template_type": disc["template_type"],
            "graph_type": disc["graph_type"],
            "normalize_mode": disc["normalize_mode"],
            "d_hidden": 64, "n_hidden_layers": 2,
            "epochs": 1, "parameterization": "score",
            "time_weighting": "importance",
            "laplacian_variant": "sym",
            "use_encoder": True,
            "encoder_type": disc["encoder_type"],
            "encoder_hid_dim": disc["encoder_hid_dim"],
            "encoder_num_layers": disc["encoder_num_layers"],
            "encoder_dropout": disc["encoder_dropout"],
            "encoder_lr": disc["encoder_lr"],
            "encoder_epochs": disc["encoder_epochs"],
            "encoder_alpha": disc["encoder_alpha"],
        }

        try:
            for method in ["precision_energy", "precision_ratio"]:
                eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                r = evaluate_single_trial(data, eval_cfg, device=device)
                if method == "precision_energy" and r.auc > best_ce:
                    best_ce = r.auc
                    best_ce_cfg = {**full_cfg, "score_method": "precision_energy",
                                   "kappa_bounds": list(best_opt.get("kappa_bounds", [0.01, 50.0]))}
                if method == "precision_ratio" and r.auc > best_cr:
                    best_cr = r.auc
                    best_cr_cfg = {**full_cfg, "score_method": "precision_ratio",
                                   "kappa_bounds": list(best_opt.get("kappa_bounds", [0.01, 50.0]))}
        except:
            pass

        if (i + 1) % 50 == 0:
            print("  [%d/%d enc=%s] best_ce=%.1f%% best_cr=%.1f%%" %
                  (i + 1, len(discrete_choices), disc["encoder_type"],
                   best_ce * 100, best_cr * 100), flush=True)

    best_overall = max(best_ce, best_cr)
    best_cfg = best_ce_cfg if best_ce >= best_cr else best_cr_cfg
    print("  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%) enc=%s" % (
        best_overall * 100, best_ce * 100, best_cr * 100,
        best_cfg.get("encoder_type", "?")))
    return {"dataset": ds, "best_auc": best_overall,
            "best_ce": best_ce, "best_cr": best_cr,
            "best_config": best_cfg}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="enron,weibo,reddit,yelpchi,facebook,acm,amazon,blogcatalog,t_finance,elliptic,elliptic_plus_plus,dgraph")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_v6_configs_%s.json" % ds)
        if os.path.exists(out_file):
            print("=== %s === SKIP (already exists: %s)" % (ds, out_file))
            continue

        print("=== %s ===" % ds)
        result = run_dataset(ds, device=args.device)

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("  Saved to %s" % out_file)

    print("\n=== SUMMARY ===")
    for ds in args.datasets.split(","):
        ds = ds.strip()
        out_file = os.path.join(args.output_dir, "eb_gad_v6_configs_%s.json" % ds)
        if os.path.exists(out_file):
            with open(out_file) as f:
                r = json.load(f)
            cfg = r["best_config"]
            print("%-18s  best=%.1f%%  CE=%.1f%%  CR=%.1f%%  enc=%s" % (
                ds, r["best_auc"]*100, r["best_ce"]*100, r["best_cr"]*100,
                cfg.get("encoder_type", "?")))
        else:
            print("%-18s  not ready" % ds)


if __name__ == "__main__":
    main()
