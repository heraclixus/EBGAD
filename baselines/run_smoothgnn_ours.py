"""Run SmoothGNN (Dong et al., WWW 2025; github.com/xydong127/SmoothGNN) on our datasets with our evaluation sets.

Uses the authors' code unchanged (graphdata.Graph, model.NAD, utils.get_lap / get_infmatrix from the SmoothGNN
repository) and their hyperparameters by dataset size class (name.PARAMETERS: lr, hop, init, seed, eps): SMALL
for graphs they class as small (Reddit, Amazon) and for our new small graphs (Weibo, Facebook, BlogCatalog, ACM),
MEDIUM for T-Finance and YelpChi, LARGE for Elliptic, Elliptic++ and DGraph. Features are the raw features as
loaded (their protocol), the graph is symmetrized as loaded, and the evaluation index is our set of labeled nodes.
Training follows main.py (100 epochs of Adagrad on the full graph, unsupervised loss); as in main.py the reported
number is the last epoch's AUROC / AUPRC, and the best epoch is recorded as well.
Usage (needs DGL): SMOOTHGNN_DIR=third_party/SmoothGNN python baselines/run_smoothgnn_ours.py weibo ...
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, time, argparse, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOOTH_DIR = os.environ.get("SMOOTHGNN_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "SmoothGNN"))
sys.path.insert(0, SMOOTH_DIR); os.chdir(SMOOTH_DIR)
import dgl                                    # noqa: E402
import utils as sg_utils                      # noqa: E402
import name as sg_name                        # noqa: E402
from graphdata import Graph                   # noqa: E402
import model as sg_model                      # noqa: E402
from data_utils import load_data              # noqa: E402  (our repo)
from run_ebgad import eval_mask               # noqa: E402

SIZE = {"weibo": "small", "facebook": "small", "blogcatalog": "small", "acm": "small", "reddit": "small", "amazon": "small",
        "t_finance": "medium", "yelpchi": "medium", "elliptic": "large", "elliptic_plus_plus": "large", "dgraph": "large"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--out", default=os.path.join(_REPO, "results", "baselines", "smoothgnn"))
    ap.add_argument("--nepoch", type=int, default=100); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--decay", type=float, default=1e-6)
    ap.add_argument("--pca", type=int, default=0, help="run on PCA features (our pipeline's PCA-256 for ACM / BlogCatalog) instead of the raw ones")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    for ds in args.datasets:
        t0 = time.time(); data = load_data(ds); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); index = np.where(m)[0]
        size = SIZE.get(ds, "small" if data.num_nodes < 20000 else ("medium" if data.num_nodes < 100000 else "large")); lr, hop, init, seed, eps = sg_name.PARAMETERS[size]
        sg_utils.set_seed(seed)
        ei = data.edge_index; src, dst = ei[0], ei[1]
        graph = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=data.num_nodes)
        graph = dgl.to_simple(graph)
        features = data.x.float(); labels = torch.as_tensor(y_all).long()
        if args.pca:
            from ebgad.prep import preprocess_features
            features = preprocess_features(data.x, data.edge_index, pca=args.pca).float()
        edge_index = torch.vstack(graph.edges()); n = data.num_nodes; mm = edge_index.shape[1]
        lap = sg_utils.get_lap(edge_index, n); inf = sg_utils.get_infmatrix(edge_index, n, mm, eps)
        gd = Graph(graph, features, labels, edge_index, inf, lap, hop)
        net = sg_model.NAD(features.shape[1], args.hidden, 2, gd, init)
        opt = torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=args.decay)
        best = (0.0, 0.0, -1); last = None
        for epoch in range(args.nepoch):
            net.train(); recon, anom = net(); loss = torch.mean(recon[index]) + torch.mean(anom[index])
            opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); probs = anom[index].detach().numpy()
            auc = 100 * roc_auc_score(y_all[index], probs); ap_ = 100 * average_precision_score(y_all[index], probs)
            if auc > best[0]:
                best = (auc, ap_, epoch)
            last = (auc, ap_)
            if epoch % 20 == 0 or epoch == args.nepoch - 1:
                print("  %s epoch %3d loss %.4f auc %.1f ap %.1f (%.0f s)" % (ds, epoch, float(loss), auc, ap_, time.time() - t0), flush=True)
        rec = dict(dataset=ds, method="SmoothGNN", auroc=last[0], auprc=last[1], best_auroc=best[0], best_auprc=best[1], best_epoch=best[2],
                   params=dict(lr=lr, hop=hop, init=init, seed=seed, eps=eps, size_class=size, nepoch=args.nepoch, hidden=args.hidden, pca=args.pca),
                   n=int(n), n_eval=int(len(index)), seconds=time.time() - t0)
        json.dump(rec, open(os.path.join(args.out, f"{ds}.json"), "w"), indent=1)
        print("%-20s SmoothGNN last-epoch AUROC %5.1f AUPRC %5.1f | best %5.1f at epoch %d | %s | %.0f s" % (ds, last[0], last[1], best[0], best[2], size, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
