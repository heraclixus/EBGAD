"""Test path_energy_v2 using the exact best SOC configs from sweep results."""
from __future__ import annotations

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse, json, sys, time
import numpy as np
import torch
from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data


# Dynamically get valid SOCGADConfig fields
import dataclasses
CONFIG_KEYS = {f.name for f in dataclasses.fields(SOCGADConfig)}


def load_best_configs(path="results/best_soc_configs.json"):
    with open(path) as f:
        return json.load(f)


def params_to_config(params: dict, dataset: str, score_method: str, score_K: int = 20) -> SOCGADConfig:
    """Convert sweep params dict to SOCGADConfig, overriding score_method."""
    kwargs = {"dataset": dataset}
    for k, v in params.items():
        if k in CONFIG_KEYS:
            kwargs[k] = v
    kwargs["score_method"] = score_method
    kwargs["score_K"] = score_K
    return SOCGADConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--methods", type=str,
                        default="path_energy_v2,control_energy,magnitude")
    parser.add_argument("--configs-path", type=str,
                        default="results/best_soc_configs.json")
    args = parser.parse_args()

    ds = args.dataset.lower()
    all_configs = load_best_configs(args.configs_path)
    if ds not in all_configs:
        print("Unknown dataset: %s. Available: %s" % (ds, list(all_configs.keys())))
        sys.exit(1)

    entry = all_configs[ds]
    params = entry["params"]
    orig_method = params.get("score_method", "?")
    orig_auc = entry["auc"]
    print("Dataset: %s  |  Best sweep: AUC=%.4f  method=%s" % (ds, orig_auc, orig_method))
    print("Full config: %s" % json.dumps(params, sort_keys=True))
    print()

    # Load data (handle casing)
    ds_load = ds
    if ds == "yelpchi":
        ds_load = "YelpChi"
    data = load_data(ds_load)

    methods = [m.strip() for m in args.methods.split(",")]
    # Always include the original best method for reference
    if orig_method not in methods:
        methods.append(orig_method)

    results = []
    for method in methods:
        aucs = []
        t0 = time.time()
        for trial in range(args.trials):
            torch.manual_seed(trial)
            np.random.seed(trial)
            cfg = params_to_config(params, ds, score_method=method)
            r = evaluate_single_trial(data, cfg, device=args.device)
            aucs.append(r.auc)
            print("  %s trial %d: AUC=%.4f" % (method, trial, r.auc), flush=True)
        elapsed = time.time() - t0
        rec = {
            "dataset": ds, "score_method": method,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "trials": aucs, "elapsed_s": elapsed,
            "params": {**params, "score_method": method},
        }
        results.append(rec)
        mean_auc = np.mean(aucs) * 100
        std_auc = np.std(aucs) * 100
        print("%15s %18s  AUC=%.1f +/- %.1f  (%.0fs)" % (ds, method, mean_auc, std_auc, elapsed))
        print()

    if args.output:
        with open(args.output, "w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        print("Saved to %s" % args.output)


if __name__ == "__main__":
    main()
