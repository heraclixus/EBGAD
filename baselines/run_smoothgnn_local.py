"""Run SmoothGNN (authors' graphdata.Graph and model.NAD, unchanged) locally without DGL.

The authors' code touches DGL only through graph.in_degrees(), graph.local_scope(), graph.ndata and
graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h')), i.e. one sparse product A @ h per hop, plus two
helpers in utils.py: get_lap (normalized Laplacian) and get_infmatrix (the rank-one matrix deg deg^T with a
threshold, which the authors materialize as an n x n sparse tensor).  This runner provides an exact shim for
those calls (scipy sparse product; the rank-one product applied as deg (deg^T X), identical values) and keeps
everything else as in run_smoothgnn_ours.py: size-class hyperparameters, 100 epochs of Adagrad, last-epoch
score, our evaluated nodes.  Usage: SMOOTHGNN_DIR=third_party/SmoothGNN python baselines/run_smoothgnn_local.py acm [--threads 4]
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, time, argparse, types, copy, numpy as np, torch, scipy.sparse as sp
from scipy.sparse import csgraph
from sklearn.metrics import roc_auc_score, average_precision_score
SMOOTH_DIR = os.environ.get("SMOOTHGNN_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "SmoothGNN"))


class _Fn:
    """dgl.function stand-in: only copy_u / sum are used."""
    @staticmethod
    def copy_u(src, out): return ("copy_u", src, out)
    @staticmethod
    def sum(msg, out): return ("sum", msg, out)


class ShimGraph:
    """The subset of the DGL graph API that graphdata.Graph uses."""
    def __init__(self, A):
        self.A = A.tocsr(); self.ndata = {}
    def in_degrees(self): return torch.from_numpy(np.asarray(self.A.sum(0)).ravel()).float()
    def num_nodes(self): return self.A.shape[0]
    def local_scope(self):
        g = self
        class _Scope:
            def __enter__(self_): self_.saved = dict(g.ndata); return g
            def __exit__(self_, *a): g.ndata = self_.saved
        return _Scope()
    def update_all(self, msg, red):
        assert msg[0] == "copy_u" and red[0] == "sum"
        h = self.ndata[msg[1]].detach().numpy()
        self.ndata[red[2]] = torch.from_numpy(np.asarray(self.A @ h)).float()


class RankOne:
    """deg deg^T with the authors' threshold, applied without materializing it."""
    def __init__(self, deg): self.deg = deg
    def matmul(self, X): return self.deg[:, None] * (self.deg[None, :] @ X)


def install_shim():
    dgl = types.ModuleType("dgl"); dgl.function = _Fn(); sys.modules["dgl"] = dgl; sys.modules["dgl.function"] = dgl.function
    # graphdata.py calls torch.spmm(infmatrix, features): route the rank-one case through RankOne
    _spmm = torch.spmm
    def spmm(a, b):
        return a.matmul(b) if isinstance(a, RankOne) else _spmm(a, b)
    torch.spmm = spmm


def get_lap(edge_index, n):
    ei = edge_index.numpy(); adj = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    L = csgraph.laplacian(adj, normed=True).tocoo().astype(np.float32)
    return torch.sparse_coo_tensor(torch.from_numpy(np.vstack([L.row, L.col])).long(), torch.from_numpy(L.data), (n, n))


def get_infmatrix(edge_index, n, m, eps):
    deg = torch.bincount(edge_index[0], minlength=n).float() + 1
    deg = torch.sqrt(deg / (2 * m + n)); deg = torch.where(deg < eps, torch.zeros_like(deg), deg)
    return RankOne(deg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--out", default="results/baselines/smoothgnn")
    ap.add_argument("--nepoch", type=int, default=100); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--decay", type=float, default=1e-6)
    ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads); install_shim()
    sys.path.insert(0, SMOOTH_DIR)
    import name as sg_name; from graphdata import Graph; import model as sg_model   # noqa: E402
    from data_utils import load_data; from run_ebgad import eval_mask                # noqa: E402
    SIZE = {"weibo": "small", "facebook": "small", "blogcatalog": "small", "acm": "small", "reddit": "small", "amazon": "small",
            "t_finance": "medium", "yelpchi": "medium", "elliptic": "large", "elliptic_plus_plus": "large", "dgraph": "large"}
    os.makedirs(args.out, exist_ok=True)
    for ds in args.datasets:
        t0 = time.time(); data = load_data(ds); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); index = np.where(m)[0]
        lr, hop, init, seed, eps = sg_name.PARAMETERS[SIZE[ds]]
        torch.manual_seed(seed); np.random.seed(seed)
        ei = data.edge_index; src, dst = ei[0], ei[1]; n = data.num_nodes
        A = sp.coo_matrix((np.ones(2 * len(src)), (torch.cat([src, dst]).numpy(), torch.cat([dst, src]).numpy())), shape=(n, n)).tocsr(); A.data[:] = 1.0
        graph = ShimGraph(A); ii, jj = A.nonzero(); edge_index = torch.from_numpy(np.vstack([ii, jj])).long(); mm = edge_index.shape[1]
        features = data.x.float(); labels = torch.as_tensor(y_all).long()
        lap = get_lap(edge_index, n); inf = get_infmatrix(edge_index, n, mm, eps)
        print("  %s: n=%d edges=%d d=%d, building hops..." % (ds, n, mm, features.shape[1]), flush=True)
        gd = Graph(graph, features, labels, edge_index, inf, lap, hop)
        print("  graph data ready (%.0f s)" % (time.time() - t0), flush=True)
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
                   params=dict(lr=lr, hop=hop, init=init, seed=seed, eps=eps, size_class=SIZE[ds], nepoch=args.nepoch, hidden=args.hidden, pca=0, runner="local shim"),
                   n=int(n), n_eval=int(len(index)), seconds=time.time() - t0)
        json.dump(rec, open(os.path.join(args.out, f"{ds}.json"), "w"), indent=1)
        print("%-20s SmoothGNN last-epoch AUROC %5.1f AUPRC %5.1f | best %5.1f at epoch %d | %s | %.0f s" % (ds, last[0], last[1], best[0], best[2], SIZE[ds], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
