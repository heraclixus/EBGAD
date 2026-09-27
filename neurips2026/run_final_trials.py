"""Run 10-trial evaluation on the best config for each dataset.

Loads the best config from all sweep results, runs 10 seeded trials,
reports mean ± std.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, glob
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data


def load_best_config(ds):
    """Load the best config across all sweep versions."""
    best_auc = 0
    best_cfg = None
    best_src = ""

    for v in ["v4", "v5", "v6", "v7", "v8"]:
        f = "results/eb_gad_%s_configs_%s.json" % (v, ds)
        try:
            r = json.load(open(f))
            if r["best_auc"] > best_auc:
                best_auc = r["best_auc"]
                best_cfg = r["best_config"]
                best_src = v
        except:
            pass

    for tag in ["targeted", "bandpass", "lbfgsb", "nokappa"]:
        f = "results/eb_gad_%s_%s.json" % (tag, ds)
        try:
            r = json.load(open(f))
            if r["best_auc"] > best_auc:
                best_auc = r["best_auc"]
                best_cfg = r["best_config"]
                best_src = tag
        except:
            pass

    for f in glob.glob("results/eb_gad_dgraph_*.json"):
        if ds == "dgraph":
            try:
                r = json.load(open(f))
                if r["best_auc"] > best_auc:
                    best_auc = r["best_auc"]
                    best_cfg = r["best_config"]
                    best_src = os.path.basename(f)
            except:
                pass

    return best_auc, best_cfg, best_src


def run_trials(ds, cfg, n_trials=10, device="cuda"):
    """Run n_trials seeded evaluations and return AUCs."""
    data = load_data("YelpChi" if ds == "yelpchi" else ds)
    cfg_clean = {k: v for k, v in cfg.items()
                 if k not in ("kappa_bounds", "optimizer", "rho_newton", "ml")}
    # Force epochs=0 to avoid score network training (we use closed-form scoring)
    cfg_clean["epochs"] = 0

    aucs = []
    for seed in range(n_trials):
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            eval_cfg = SOCGADConfig(dataset=ds, score_K=1, **cfg_clean)
            r = evaluate_single_trial(data, eval_cfg, device=device)
            aucs.append(r.auc)
            print("    seed %d: %.1f%%" % (seed, r.auc * 100), flush=True)
        except Exception as e:
            print("    seed %d: ERROR %s" % (seed, str(e)[:50]), flush=True)

    return np.array(aucs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="enron,weibo,reddit,amazon,yelpchi,blogcatalog,facebook,acm,elliptic,elliptic_plus_plus,dgraph,t_finance")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/final_trials.json")
    args = parser.parse_args()

    all_results = {}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print("=== %s ===" % ds, flush=True)

        best_auc, best_cfg, best_src = load_best_config(ds)
        if best_cfg is None:
            print("  No config found, skipping")
            continue

        enc = best_cfg.get("encoder_type", "none") if best_cfg.get("use_encoder") else "none"
        print("  Config from %s: %.1f%% enc=%s score=%s" % (
            best_src, best_auc * 100, enc, best_cfg.get("score_method")), flush=True)

        aucs = run_trials(ds, best_cfg, args.n_trials, args.device)

        if len(aucs) > 0:
            mean = float(np.mean(aucs))
            std = float(np.std(aucs))
            print("  Result: %.1f ± %.1f%% (%d trials)" % (mean*100, std*100, len(aucs)), flush=True)
            all_results[ds] = {
                "mean": mean, "std": std, "n_trials": len(aucs),
                "aucs": aucs.tolist(), "config": best_cfg, "source": best_src,
            }

            # Save incrementally
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    print("\n=== FINAL RESULTS ===")
    for ds, r in all_results.items():
        print("  %-18s %.1f ± %.1f%%" % (ds, r["mean"]*100, r["std"]*100))

    print("\nSaved to %s" % args.output)


if __name__ == "__main__":
    main()
