"""Empirical Bayes test v2: includes encoder options in prior specification."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse, json, sys, os, itertools
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig
from soc.soc_trainer import SOCTrainer
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc

# Compact grids including encoder
GRIDS = {
    "weibo": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [1.0, 5.0],
        "gamma": [0.2, 0.5],
        "template_type": ["zero", "low", "high"],
        "graph_type": ["original"],
        "use_encoder": [False, True],
        "encoder_type": ["gcn"],
        "encoder_hid_dim": [32],
        "encoder_epochs": [300],
        "laplacian_variant": ["rw", "sym"],
    },
    "facebook": {
        "rho": [0.0, 0.25, 0.5, 1.0],
        "kappa": [1.0, 5.0, 10.0, 20.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low", "affinity"],
        "graph_type": ["original"],
        "use_encoder": [False, True],
        "encoder_type": ["mlp"],
        "encoder_hid_dim": [32],
        "encoder_epochs": [300],
        "recon_features": [False, True],
    },
    "reddit": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.1, 1.0, 5.0],
        "gamma": [0.5, 1.0],
        "template_type": ["zero", "low", "affinity"],
        "graph_type": ["original", "affinity"],
    },
    "yelpchi": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [1.0, 5.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low"],
        "graph_type": ["original"],
    },
    "acm": {
        "rho": [0.0, 0.5, 1.0],
        "kappa": [0.01, 0.1, 1.0],
        "gamma": [0.1, 0.2, 0.5],
        "template_type": ["zero", "low", "affinity"],
        "graph_type": ["original", "affinity"],
        "use_encoder": [False, True],
        "encoder_type": ["mlp"],
        "encoder_hid_dim": [32],
        "encoder_epochs": [300],
        "normalize_mode": ["zscore", "minmax"],
    },
    "t_finance": {
        "rho": [0.0, 0.5, 0.75, 1.0],
        "kappa": [0.1, 1.0, 5.0],
        "gamma": [0.2, 0.5, 1.0],
        "template_type": ["zero", "low", "affinity"],
        "graph_type": ["original"],
    },
}

FIXED = dict(
    d_hidden=64, n_hidden_layers=2, lr=0.001, epochs=200, patience=50,
    parameterization="score", time_weighting="importance",
    lam_penalty=50, alpha=2.0, laplacian_variant="sym",
)


def grid_product(grid):
    keys = sorted(grid.keys())
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def filter_invalid(prior):
    """Remove invalid combos (e.g., encoder_type without use_encoder)."""
    if not prior.get("use_encoder", False):
        for k in ["encoder_type", "encoder_hid_dim", "encoder_epochs", "recon_features"]:
            prior.pop(k, None)
    return prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ds = args.dataset.lower()
    if ds not in GRIDS:
        print("Available: %s" % list(GRIDS.keys()))
        sys.exit(1)

    data = load_data("YelpChi" if ds == "yelpchi" else ds)

    # Deduplicate configs after filtering
    seen = set()
    configs = []
    for prior in grid_product(GRIDS[ds]):
        prior = filter_invalid(prior)
        key = json.dumps(prior, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(prior)

    print("Dataset: %s | %d unique configs" % (ds, len(configs)))

    out_path = args.output or "results/eb_v2_%s.jsonl" % ds
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "w")

    for i, prior in enumerate(configs):
        torch.manual_seed(0)
        np.random.seed(0)

        # Merge FIXED with prior, prior overrides
        params = {**FIXED, **prior}
        # Remove keys not in FIXED that conflict
        if "laplacian_variant" in prior:
            params["laplacian_variant"] = prior["laplacian_variant"]

        try:
            cfg = SOCGADConfig(dataset=ds, score_method="control_energy", score_K=1, **params)
            trainer_cfg = cfg.to_trainer_config()
            trainer = SOCTrainer(trainer_cfg)
            trainer.train(data, device=args.device)
            best_loss = float(trainer.best_loss) if hasattr(trainer, "best_loss") else 1e30
            if np.isnan(best_loss) or np.isinf(best_loss):
                best_loss = 1e30

            x_test = data.x.to(args.device).float()
            ce = compute_anomaly_scores(trainer, x_test, K=1, method="control_energy")
            cr = compute_anomaly_scores(trainer, x_test, K=1, method="ce_ratio")

            y_ce, s_ce = mask_labeled(data.y, ce.cpu().detach())
            y_cr, s_cr = mask_labeled(data.y, cr.cpu().detach())
            auc_ce = eval_roc_auc(y_ce, s_ce) if y_ce is not None else 0.0
            auc_cr = eval_roc_auc(y_cr, s_cr) if y_cr is not None else 0.0
        except Exception as e:
            print("  ERROR config %d: %s" % (i, e))
            best_loss, auc_ce, auc_cr = 1e30, 0.0, 0.0

        rec = {"dataset": ds, "prior": prior, "best_loss": best_loss,
               "auc_ce": float(auc_ce), "auc_ce_ratio": float(auc_cr)}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print("[%d/%d] loss=%.4f  ce=%.3f  cr=%.3f" % (i+1, len(configs), best_loss, auc_ce, auc_cr))

    fout.close()

    # Correlation analysis
    records = [json.loads(l) for l in open(out_path)]
    valid = [(r["best_loss"], r["auc_ce"], r["auc_ce_ratio"]) for r in records
             if r["best_loss"] < 1e20]
    print("\n%d/%d valid records" % (len(valid), len(records)))
    if len(valid) >= 10:
        from scipy.stats import spearmanr
        vl, vce, vcr = zip(*valid)
        corr_ce, p_ce = spearmanr(vl, vce)
        corr_cr, p_cr = spearmanr(vl, vcr)
        print("Spearman(loss, CE):  r=%.3f  p=%.6f" % (corr_ce, p_ce))
        print("Spearman(loss, CR):  r=%.3f  p=%.6f" % (corr_cr, p_cr))
        bi = int(np.argmin(vl))
        bce = int(np.argmax(vce))
        print("Best-loss: loss=%.4f ce=%.3f cr=%.3f" % (vl[bi], vce[bi], vcr[bi]))
        print("Best-CE:   loss=%.4f ce=%.3f" % (vl[bce], vce[bce]))
    print("Saved to %s" % out_path)


if __name__ == "__main__":
    main()
