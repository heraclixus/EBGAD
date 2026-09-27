"""The model's no-graph limit (kappa^2 -> infinity, Proposition 3) on its own: the template is the column mean, the
score is the standardized-feature energy ||x_i||^2, read in both directions on the evaluated nodes (the same quantity
the runner reports as 'no graph'). Cheap on any graph size. Usage: PYTHONPATH=. python no_graph_limit.py tsocial [--pca 256]"""
import sys, json, argparse, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from run_ebgad import eval_mask
ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--pca", type=int, default=None); ap.add_argument("--out", default="results/ebgad_v4_new/no_graph.jsonl"); a = ap.parse_args()
for ds in a.datasets:
    data = load_data(ds); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    x = preprocess_features(data.x, data.edge_index, pca=a.pca).numpy(); xn = (x * x).sum(1)[m]
    rec = dict(dataset=ds, pca=a.pca, n=int(data.num_nodes), d=int(x.shape[1]), n_eval=int(m.sum()), prevalence=float(y.mean()),
               NG_DEV=[100 * roc_auc_score(y, xn), 100 * average_precision_score(y, xn)], NG_CAM=[100 * roc_auc_score(y, -xn), 100 * average_precision_score(y, -xn)])
    open(a.out, "a").write(json.dumps(rec) + "\n"); print("%-12s no-graph limit: DEV %.1f/%.1f  CAM %.1f/%.1f (n %d, d %d, prev %.3f)" % (ds, *rec["NG_DEV"], *rec["NG_CAM"], rec["n"], rec["d"], rec["prevalence"]), flush=True)
