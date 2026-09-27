"""Run the edge leg (ebgad/edgeleg.py) on datasets and write results/ebgad_v4_edge/<ds>.json + <ds>_edge.npz.
Records: EDGE-DEV / EDGE-CAM / EDGE-TWO-SIDED = [AUROC, AUPRC] on the evaluated nodes, population constants,
degree statistics, seconds.  Usage: edge_leg_run.py acm:256 blogcatalog:256 facebook ... [--out DIR]
"""
import sys, os, json, time, argparse, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features
from ebgad.edgeleg import edge_leg, edge_surprises
from run_ebgad import eval_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="results/ebgad_v4_edge")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for arg in args.datasets:
        name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
        t0 = time.time()
        data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
        xt = preprocess_features(data.x, data.edge_index, pca=pca); x = xt.double().numpy()
        t1 = time.time()
        res = edge_leg(x, data.edge_index.numpy()); sur = edge_surprises(res["e"])
        rec = {"dataset": name, "pca": pca, "n": int(x.shape[0]), "d": int(x.shape[1]), "n_edges": res["n_edges"],
               "deg_mean": float(res["deg"].mean()), "deg_median": float(np.median(res["deg"])),
               "isolated_eval": int(((res["deg"] == 0) & m).sum()), "r_med": res["r_med"], "z_med": res["z_med"], "z_mad": res["z_mad"],
               "seconds": {"load": t1 - t0, "edge_leg": time.time() - t1}}
        for k, v in sur.items():
            vv = np.nan_to_num(v[m], posinf=1e6)
            rec["EDGE-" + k] = [100 * roc_auc_score(y, vv), 100 * average_precision_score(y, vv)]
        json.dump(rec, open(f"{args.out}/{name}.json", "w"), indent=1)
        np.savez_compressed(f"{args.out}/{name}_edge.npz", e=res["e"], deg=res["deg"], y=y_all)
        print("%-20s DEV %5.1f/%4.1f  CAM %5.1f/%4.1f  2S %5.1f/%4.1f | deg %.1f r_med %.3f z %.2f/%.2f | %.0f s" % (
            name, *rec["EDGE-DEV"], *rec["EDGE-CAM"], *rec["EDGE-TWO-SIDED"], rec["deg_mean"], rec["r_med"], rec["z_med"], rec["z_mad"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
