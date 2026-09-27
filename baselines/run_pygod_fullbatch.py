"""Rerun PyGOD detectors (DOMINANT, AnomalyDAE, CONAD, ANOMALOUS; pygod 1.1.0, defaults of run_graph_baselines.py:
hid_dim 64, 100 epochs, 5 trials, native score direction) on CPU without pyg-lib / torch-sparse.

PyGOD's base detector always iterates a NeighborLoader, which needs one of those extensions.  In full-batch mode
(batch_size = n, num_neigh = -1, the setting of run_graph_baselines.py) the loader yields exactly one batch holding
the whole graph with n_id = arange(n), so this runner replaces it with a loader that yields the data object itself
with batch_size = n and n_id = arange(n): identical computation, no sampling.  Metrics on the labeled nodes as in
run_graph_baselines.run_pygod_baseline.  Usage: python run_pygod_fullbatch.py weibo --methods DOMINANT AnomalyDAE CONAD
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, time, argparse, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import pygod.detector.base as pgb   # noqa: E402


class FullBatchLoader:
    def __init__(self, data, num_neigh, batch_size):
        n = data.x.shape[0]; assert batch_size == n and all(k == -1 for k in num_neigh), "full-batch shim only"
        data.batch_size = n; data.n_id = torch.arange(n); self.data = data
    def __iter__(self):
        yield self.data


pgb.NeighborLoader = FullBatchLoader
from run_graph_baselines import run_pygod_baseline   # noqa: E402
from data_utils import load_data                      # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--methods", nargs="+", default=["DOMINANT", "AnomalyDAE", "CONAD"])
ap.add_argument("--trials", type=int, default=5); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=os.path.join(REPO, "results/baselines/pygod_rerun")); args = ap.parse_args()
torch.set_num_threads(args.threads); os.makedirs(args.out, exist_ok=True)
for ds in args.datasets:
    data = load_data(ds)
    for method in args.methods:
        t0 = time.time(); trials = []
        for t in range(args.trials):
            torch.manual_seed(t); np.random.seed(t)
            r = run_pygod_baseline(data, method, device="cpu", epochs=args.epochs); trials.append(r)
            print("  %s %s trial %d: AUROC %.2f AUPRC %.2f (%.0f s)" % (ds, method, t + 1, 100 * r["auc"], 100 * r["ap"], time.time() - t0), flush=True)
        aucs = np.array([r["auc"] for r in trials]); aps = np.array([r["ap"] for r in trials])
        rec = dict(dataset=ds, method=method, auroc=100 * aucs.mean(), auroc_std=100 * aucs.std(), auprc=100 * aps.mean(), trials=[100 * a for a in aucs],
                   params=dict(hid_dim=64, epochs=args.epochs, trials=args.trials, runner="full-batch shim, pygod 1.1.0, CPU"), seconds=time.time() - t0)
        json.dump(rec, open(os.path.join(args.out, f"{method}_{ds}.json"), "w"), indent=1)
        print("%-12s %-10s AUROC %.1f +- %.1f AUPRC %.1f | %.0f s" % (ds, method, rec["auroc"], rec["auroc_std"], rec["auprc"], rec["seconds"]), flush=True)
