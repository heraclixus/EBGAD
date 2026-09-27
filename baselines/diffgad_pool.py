"""Pool the split DiffGAD runs (run_diffgad_cpu.py --trials K --tag _x, same autoencoder checkpoint) into one record
with the mean, std and max over all diffusion trials, i.e. the authors' 20-trial protocol run in parallel.
Usage: python diffgad_pool.py weibo  ->  results/baselines/pygod_rerun/diffgad_weibo.json"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, glob, numpy as np
ds = sys.argv[1]; d = "results/baselines/pygod_rerun"
parts = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, f"diffgad_{ds}_*.json")))]
if not parts:      # split runs stopped after their completed trials: pool the per-trial lines of the logs (same checkpoint, --ae-ckpt)
    import re
    logs = sorted(glob.glob(os.path.join(d, f"diffgad_{ds}_?.log")))
    parts = [dict(trials=[[100 * float(v) for v in l.split()[1:]] for l in open(f, errors="ignore").read().replace("\r", "\n").splitlines() if l.startswith("TRIAL ")],
                  ae_ckpt=int(re.search(r"loading checkpoint from (\d+)", open(f, errors="ignore").read()).group(1)), config=None, seconds=0.0) for f in logs]
    import yaml; cfg = yaml.load(open(os.path.join(os.environ.get("DIFFGAD_DIR", "third_party/DiffGAD"), "configs", ds + ".yaml")), Loader=yaml.Loader)
    for p in parts: p["config"] = cfg
trials = np.array([t for p in parts for t in p["trials"]]); assert len(trials) > 0 and len({p["ae_ckpt"] for p in parts}) == 1
rec = dict(dataset=ds, method="DiffGAD", auroc=float(trials[:, 0].mean()), auroc_std=float(trials[:, 0].std(ddof=1)), auroc_max=float(trials[:, 0].max()),
           auprc=float(trials[:, 2].mean()), n_trials=int(len(trials)), ae_ckpt=parts[0]["ae_ckpt"], config=parts[0]["config"],
           config_provenance="published configs/%s.yaml with main.py arguments; authors' 20-trial protocol (%d diffusion trials pooled over %d processes on the autoencoder checkpoint their code selected); CPU" % (ds, len(trials), len(parts)),
           seconds=sum(p["seconds"] for p in parts))
json.dump(rec, open(os.path.join(d, f"diffgad_{ds}.json"), "w"), indent=1)
print("%s DiffGAD AUROC %.1f +- %.1f (max %.1f) AUPRC %.1f over %d trials" % (ds, rec["auroc"], rec["auroc_std"], rec["auroc_max"], rec["auprc"], rec["n_trials"]))
