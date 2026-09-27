"""Extract full best SOC configs from sweep results."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import json, glob

datasets = {
    "disney": ["*disney*", "*Disney*"],
    "enron": ["*enron*", "*Enron*"],
    "weibo": ["*weibo*", "*Weibo*"],
    "reddit": ["*reddit*", "*Reddit*"],
    "amazon": ["*amazon*", "*Amazon*"],
    "yelpchi": ["*yelp*", "*Yelp*"],
    "blogcatalog": ["*blog*", "*Blog*"],
    "facebook": ["*face*", "*Face*"],
    "acm": ["*acm*", "*ACM*"],
    "t_finance": ["*t_finance*", "*T_Finance*"],
    "elliptic": ["*elliptic*"],
    "elliptic_plus_plus": ["*elliptic_plus*", "*Elliptic_plus*"],
    "dgraph": ["*dgraph*", "*DGraph*"],
}

best = {}
for name, pats in datasets.items():
    best_auc = 0
    best_params = None
    for pat in pats:
        for f in (glob.glob("results/" + pat + ".jsonl")
                  + glob.glob("results/sweep_soc_" + pat + ".jsonl")
                  + glob.glob("results/sweep_soc_*" + pat.replace("*", "") + "*.jsonl")):
            if name == "elliptic" and "plus" in f.lower():
                continue
            try:
                for line in open(f):
                    r = json.loads(line)
                    p = r.get("params", {})
                    if "score_method" not in p:
                        continue
                    a = r.get("auc_mean", r.get("auc", 0))
                    if a > best_auc:
                        best_auc = a
                        best_params = p
            except Exception:
                pass
    if best_params:
        best[name] = {"auc": best_auc, "params": best_params}

with open("results/best_soc_configs.json", "w") as f:
    json.dump(best, f, indent=2, sort_keys=True)

print("Saved %d configs" % len(best))
for k in sorted(best):
    v = best[k]
    sm = v["params"].get("score_method", "?")
    auc = v["auc"] * 100
    print("  %-20s  AUC=%.1f%%  method=%s" % (k, auc, sm))
