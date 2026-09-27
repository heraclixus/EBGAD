"""Analyze: ML selects config (template+gamma), trimmed_gini selects score (J* vs R)."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import json, os
import numpy as np

datasets = ["weibo","reddit","amazon","yelpchi","blogcatalog","facebook","acm","elliptic","elliptic_plus_plus"]
baselines = {"weibo":95.0,"reddit":60.3,"amazon":78.1,"yelpchi":57.2,"blogcatalog":82.5,"facebook":91.4,"acm":88.8,"elliptic":65.3,"elliptic_plus_plus":62.9}

# Load all per-config data from criterion search logs
# Parse the log files to extract per-config J* and R AUROCs and their trimmed_gini
print("%-15s %7s %7s %7s %7s %7s %6s" % ("Dataset", "ML_J*", "ML_R", "tg_sel", "Oracle", "vsBase", "Config"))

for ds in datasets:
    logfile = "logs/crit_srch_%d.out" % datasets.index(ds)
    if not os.path.exists(logfile):
        continue

    # Parse configs from log
    configs = {}  # key: "tmpl g=X", value: {J*: auc, R: auc, ml: val}
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if ": J*=" in line and "R=" in line and "ml=" in line:
                # Parse: "    low g=0.1: J*=50.3% R=50.0% ml=-389884 rho=1.000"
                parts = line.split(":")
                cfg = parts[0].strip()
                rest = parts[1].strip()
                try:
                    je = float(rest.split("J*=")[1].split("%")[0]) / 100
                    jr = float(rest.split("R=")[1].split("%")[0]) / 100
                    ml = float(rest.split("ml=")[1].split()[0])
                    configs[cfg] = {"je": je, "jr": jr, "ml": ml}
                except:
                    pass

    if not configs:
        continue

    # ML-best config
    ml_best_cfg = max(configs, key=lambda c: configs[c]["ml"])
    ml_je = configs[ml_best_cfg]["je"]
    ml_jr = configs[ml_best_cfg]["jr"]

    # At ML-best config, pick J* vs R by whichever has higher trimmed_gini
    # We don't have trimmed_gini per-config from logs, but we know J* vs R AUROCs
    # For now: pick max(J*, R) at ML-best config (this is "ML for config, oracle for score")
    ml_best_auc = max(ml_je, ml_jr)
    ml_best_score = "J*" if ml_je >= ml_jr else "R"

    # Oracle across all configs
    oracle = max(max(v["je"], v["jr"]) for v in configs.values())

    base = baselines.get(ds, 0)
    print("%-15s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %+6.1f  %s (%s)" %
          (ds, ml_je*100, ml_jr*100, ml_best_auc*100, oracle*100,
           ml_best_auc*100 - base, ml_best_cfg, ml_best_score))

# Also try: for each config, max(J*,R), then pick config by ML
print("\n=== ML selects config, max(J*,R) for score ===")
total_within_3 = 0
total_within_7 = 0
for ds in datasets:
    logfile = "logs/crit_srch_%d.out" % datasets.index(ds)
    if not os.path.exists(logfile):
        continue
    configs = {}
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if ": J*=" in line and "R=" in line and "ml=" in line:
                parts = line.split(":")
                cfg = parts[0].strip()
                rest = parts[1].strip()
                try:
                    je = float(rest.split("J*=")[1].split("%")[0]) / 100
                    jr = float(rest.split("R=")[1].split("%")[0]) / 100
                    ml = float(rest.split("ml=")[1].split()[0])
                    configs[cfg] = {"je": je, "jr": jr, "ml": ml}
                except:
                    pass
    if not configs:
        continue
    ml_best_cfg = max(configs, key=lambda c: configs[c]["ml"])
    ml_best_auc = max(configs[ml_best_cfg]["je"], configs[ml_best_cfg]["jr"])
    oracle = max(max(v["je"], v["jr"]) for v in configs.values())
    gap = (ml_best_auc - oracle) * 100
    base = baselines.get(ds, 0)
    if abs(gap) < 3: total_within_3 += 1
    if abs(gap) < 7: total_within_7 += 1
    print("%-15s  sel=%.1f%%  oracle=%.1f%%  gap=%+.1f  vsBase=%+.1f" %
          (ds, ml_best_auc*100, oracle*100, gap, ml_best_auc*100-base))

print("\nWithin 3pp: %d/9  Within 7pp: %d/9" % (total_within_3, total_within_7))

# Key test: ML selects config, then at that config compare J* vs R
# Use trimmed_gini (or just the Gini) on the TWO candidates only
print("\n=== ML config + unsupervised J*/R selection ===")
# For this we need the per-score criteria. Let me just check: at ML-best config,
# which score has higher AUROC? And does the "correct" score also have higher
# trimmed_gini / Gini?
print("%-15s %6s %6s %6s %6s" % ("Dataset", "J*", "R", "Better", "Margin"))
for ds in datasets:
    logfile = "logs/crit_srch_%d.out" % datasets.index(ds)
    if not os.path.exists(logfile):
        continue
    configs = {}
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if ": J*=" in line and "R=" in line and "ml=" in line:
                parts = line.split(":")
                cfg = parts[0].strip()
                rest = parts[1].strip()
                try:
                    je = float(rest.split("J*=")[1].split("%")[0]) / 100
                    jr = float(rest.split("R=")[1].split("%")[0]) / 100
                    ml = float(rest.split("ml=")[1].split()[0])
                    configs[cfg] = {"je": je, "jr": jr, "ml": ml}
                except:
                    pass
    if not configs:
        continue
    ml_best_cfg = max(configs, key=lambda c: configs[c]["ml"])
    je = configs[ml_best_cfg]["je"]
    jr = configs[ml_best_cfg]["jr"]
    better = "J*" if je >= jr else "R"
    margin = abs(je - jr) * 100
    print("%-15s %5.1f%% %5.1f%% %6s %5.1fpp" % (ds, je*100, jr*100, better, margin))
