"""Run FreeGAD (Zhao et al., CIKM 2025; github.com/yunf-zhao/FreeGAD) on our datasets with our evaluation sets.

Uses the authors' code unchanged (propagated, Affinity_Gated_Residual_Encoder, Anchor_Guided_Anomaly_Scoring from
FreeGAD/FreeGAD/{utils,model}.py) and their per-dataset hyperparameters (params/FreeGAD.csv) where they published
one for the graph: Amazon, Reddit, YelpChi, T-Finance, Elliptic and BlogCatalog; Elliptic++ is the Elliptic graph
with more features and takes Elliptic's row. For DGraph, ACM, Weibo and Facebook nothing is published and the
argparse defaults of run.py are used (alpha 0.1, beta 0.9, 4 hops, 30 anchors); we do not tune with labels.
Preprocessing follows load_anomaly_detection_dataset: row-normalized features for the count-like feature sets
(their list plus Elliptic++ and DGraph), self loops except for YelpChi and BlogCatalog. AUROC / AUPRC are computed
on our evaluated nodes (labels 0/1; unknown labels excluded), which for Elliptic and DGraph differs from the
paper's all-nodes evaluation. The method is deterministic given the graph, so one run per dataset.
Usage: FREEGAD_DIR=third_party/FreeGAD/FreeGAD python baselines/run_freegad_ours.py weibo acm ...
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, csv, json, time, argparse, numpy as np, scipy.sparse as sp, torch
from sklearn.metrics import roc_auc_score, average_precision_score
FREEGAD_DIR = os.environ.get("FREEGAD_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "FreeGAD", "FreeGAD"))
sys.path.insert(0, FREEGAD_DIR)
import utils as fg_utils                     # noqa: E402  (FreeGAD/FreeGAD/utils.py)
from model import Anchor_Guided_Anomaly_Scoring  # noqa: E402
from data_utils import load_data             # noqa: E402  (our repo)
from run_ebgad import eval_mask              # noqa: E402

PARAM_ROW = {"amazon": "Amazon", "reddit": "Reddit", "yelpchi": "YelpChi", "t_finance": "tfinance", "elliptic": "elliptic",
             "blogcatalog": "BlogCatalog", "elliptic_plus_plus": "elliptic"}
ROW_NORMALIZE = {"t_finance", "elliptic", "elliptic_plus_plus", "amazon", "dgraph"}      # their list: tolokers, elliptic, tfinance, Amazon, cora
NO_SELF_LOOP = {"yelpchi", "blogcatalog"}                                                  # their list: YelpChi, BlogCatalog, Flickr
DEFAULTS = dict(alpha=0.1, beta=0.9, num_hops=4, num_shot=30)


def params_for(ds):
    if ds in PARAM_ROW:
        rows = list(csv.reader(open(os.path.join(FREEGAD_DIR, "params", "FreeGAD.csv"))))
        hdr = rows[0]; row = [r for r in rows[1:] if r[0] == PARAM_ROW[ds]][0]
        return dict(alpha=float(row[hdr.index("alpha")]), beta=float(row[hdr.index("beta")]), num_hops=int(row[hdr.index("num_hops")]),
                    num_shot=int(row[hdr.index("num_shot")])), "published (%s)" % PARAM_ROW[ds]
    return dict(DEFAULTS), "defaults"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--out", default="results/baselines/freegad")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--search", type=int, default=0, help="random-search N configs with labels (the authors' protocol) instead of defaults")
    ap.add_argument("--search-all", action="store_true", help="search even where a published config exists"); args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); device = torch.device(args.device)
    for ds in args.datasets:
        t0 = time.time(); data = load_data(ds); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
        n = data.num_nodes; ei = data.edge_index.numpy()
        adj = sp.csr_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n)); adj = ((adj + adj.T) > 0).astype(np.float32)
        adj.setdiag(0); adj.eliminate_zeros()
        feat = data.x.numpy().astype(np.float32)
        if ds in ROW_NORMALIZE:
            feat = np.asarray(fg_utils.preprocess_features(sp.lil_matrix(feat)).todense(), dtype=np.float32)
        adj_norm = fg_utils.process_adj(adj if ds in NO_SELF_LOOP else adj + sp.eye(n, dtype=np.float32))
        p, prov = params_for(ds); fg_utils.set_seed(0)
        cache = {}

        def run(p):
            if p["num_hops"] not in cache:
                cache[p["num_hops"]] = fg_utils.propagated(adj_norm, feat, p["num_hops"], device)
            feat_ = [f.clone() for f in cache[p["num_hops"]]]
            emb = fg_utils.Affinity_Gated_Residual_Encoder(feat_); orig = feat_[0]
            sim = torch.nn.functional.cosine_similarity(emb, orig)
            _, imax = torch.topk(sim, p["num_shot"], largest=True); _, imin = torch.topk(sim, p["num_shot"], largest=False)
            pos = torch.zeros(n, dtype=torch.bool, device=device); neg = torch.zeros(n, dtype=torch.bool, device=device); pos[imax] = True; neg[imin] = True
            score = Anchor_Guided_Anomaly_Scoring(emb, pos, neg, argparse.Namespace(alpha=p["alpha"], beta=p["beta"])).detach().cpu().numpy()
            return score, 100 * roc_auc_score(y, score[m]), 100 * average_precision_score(y, score[m])

        score, auc, ap_ = run(p); search = None
        if args.search and (prov == "defaults" or args.search_all):
            rng = np.random.default_rng(0); best = (auc, ap_, dict(p), score); trials = []
            for t in range(args.search):
                q = dict(alpha=float(rng.uniform(0, 1)), beta=float(rng.uniform(0, 1)), num_hops=int(rng.integers(1, 21)), num_shot=int(rng.integers(10, 101)))
                sc, a_, p_ = run(q); trials.append((a_, p_, q))
                if a_ > best[0]:
                    best = (a_, p_, q, sc)
            auc, ap_, p, score = best; prov = "random search on labels (%d configs, seed 0)" % args.search
            search = dict(n_configs=args.search, default_auroc=trials and None, trials=[(round(a_, 2), round(p_, 2), q) for a_, p_, q in trials])
        rec = dict(dataset=ds, method="FreeGAD", auroc=auc, auprc=ap_, params=p, param_provenance=prov, row_normalized=ds in ROW_NORMALIZE,
                   self_loops=ds not in NO_SELF_LOOP, n=int(n), n_eval=int(m.sum()), seconds=time.time() - t0, search=search)
        json.dump(rec, open(os.path.join(args.out, f"{ds}.json"), "w"), indent=1); np.save(os.path.join(args.out, f"{ds}_scores.npy"), score)
        print("%-20s FreeGAD AUROC %5.1f AUPRC %5.1f | %s %s | %.0f s" % (ds, auc, ap_, prov, p, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
