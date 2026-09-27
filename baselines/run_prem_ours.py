"""Run PREM (Pan et al., ICDM 2023; github.com/CampanulaBells/PREM-GAD) on our datasets with our evaluation sets.

Uses the authors' code (modules.model.Model / Dataloader, modules.train.train_model / eval_model) with one exact,
memory-safe replacement: their get_diag(graph, k) aggregates a dense n x n identity to read the diagonal of
(D^-1/2 A D^-1/2)^k, which does not fit for graphs above ~30,000 nodes; for their default k = 2 that diagonal is
the row-wise sum of squares of the normalized adjacency, computed here sparsely (identical values).
Hyperparameters: the authors' published configuration for ACM (lr 1e-4, alpha 0.7, gamma 0.2, 200 epochs);
their argparse defaults elsewhere (lr 5e-4, alpha 0.3, gamma 0.4, 1500 epochs, hidden 128, k 2); no label tuning.
Features: the authors row-normalize count features (preprocess_features); we apply it when all features are
non-negative and use the features as loaded otherwise. Self loops are added as in load_dataset. Seed 1 as in run.py.
AUROC / AUPRC on our evaluated nodes. Usage (needs DGL): PREM_DIR=third_party/PREM-GAD python baselines/run_prem_ours.py acm weibo ...
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, time, argparse, numpy as np, scipy.sparse as sp, torch
from sklearn.metrics import roc_auc_score, average_precision_score
PREM_DIR = os.environ.get("PREM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "PREM-GAD"))
sys.path.insert(0, PREM_DIR)
import dgl                                                        # noqa: E402
import modules.model as pm                                        # noqa: E402
from modules.train import train_model, eval_model                 # noqa: E402
from modules.utils import set_random_seeds                        # noqa: E402
from data_utils import load_data                                  # noqa: E402
from run_ebgad import eval_mask                                   # noqa: E402

CONFIG = {"acm": dict(lr=1e-4, alpha=0.7, gamma=0.2, num_epoch=200)}
DEFAULT = dict(lr=5e-4, alpha=0.3, gamma=0.4, num_epoch=1500)


def sparse_get_diag(graph, k):
    """diag((D^-1/2 A D^-1/2)^k) for k = 2 without the dense identity: row sums of squares of the normalized adjacency."""
    assert k == 2, "sparse diagonal implemented for k = 2 (the authors' default)"
    src, dst = graph.edges(); n = graph.num_nodes()
    deg = graph.in_degrees().float().clamp(min=1).numpy(); norm = 1.0 / np.sqrt(deg)
    w = norm[src.numpy()] * norm[dst.numpy()]
    A = sp.csr_matrix((w, (dst.numpy(), src.numpy())), shape=(n, n))       # message from src to dst, as in aggregation()
    return torch.from_numpy(np.asarray(A.multiply(A).sum(1)).ravel()).float()


pm.get_diag = sparse_get_diag


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--out", default="results/baselines/prem")
    ap.add_argument("--workdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "work_prem")); ap.add_argument("--epochs", type=int, default=0, help="override the epoch budget (used for DGraph: 200, the authors' ACM budget; 1500 full-batch epochs on 3.7M nodes would take 15 h)")
    args_cli = ap.parse_args()
    os.makedirs(args_cli.out, exist_ok=True); out = os.path.abspath(args_cli.out)
    for ds in args_cli.datasets:
        t0 = time.time(); data = load_data(ds); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]; n = data.num_nodes
        wd = os.path.join(args_cli.workdir, ds); os.makedirs(wd, exist_ok=True); os.chdir(wd)
        ei = data.edge_index.numpy(); src = np.concatenate([ei[0], ei[1]]); dst = np.concatenate([ei[1], ei[0]])
        g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=n); g = dgl.to_simple(g.remove_self_loop()); g = dgl.add_self_loop(g)
        x = data.x.numpy().astype(np.float32); rownorm = bool((x >= 0).all())
        if rownorm:
            rs = x.sum(1); r_inv = np.where(rs > 0, 1.0 / np.maximum(rs, 1e-12), 0.0); x = x * r_inv[:, None]
        cfg = dict(CONFIG.get(ds, DEFAULT))
        if args_cli.epochs:
            cfg["num_epoch"] = args_cli.epochs
        args = argparse.Namespace(dataset=ds, n_hidden=128, k=2, device="cpu", seed=1, weight_decay=0.0, batch_size=-1, **cfg)
        set_random_seeds(args.seed); device = torch.device("cpu")
        feats = torch.from_numpy(x)
        dl = pm.Dataloader(g, feats, args.k, dataset_name=None)
        os.makedirs("./ckpt", exist_ok=True)
        model = pm.Model(g=dl.g, n_in=dl.en.shape[1], n_hidden=args.n_hidden, k=args.k).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        state_path, stats, t_train = train_model(args, dl, model, opt, torch.nn.BCELoss())
        model.load_state_dict(torch.load(state_path)); score, t_test = eval_model(args, dl, model, y_all)
        score = np.asarray(score).ravel()
        auc, ap_ = 100 * roc_auc_score(y, score[m]), 100 * average_precision_score(y, score[m])
        rec = dict(dataset=ds, method="PREM", auroc=auc, auprc=ap_, config=cfg, config_provenance=("published (ACM)" if ds in CONFIG else "authors' defaults") + (" with %d epochs" % args_cli.epochs if args_cli.epochs else ""),
                   row_normalized=rownorm, best_epoch=stats["best_epoch"], n=int(n), n_eval=int(m.sum()), seconds=time.time() - t0)
        json.dump(rec, open(os.path.join(out, f"{ds}.json"), "w"), indent=1); np.save(os.path.join(out, f"{ds}_scores.npy"), score)
        print("%-20s PREM AUROC %5.1f AUPRC %5.1f | %s | best epoch %d | %.0f s" % (ds, auc, ap_, rec["config_provenance"], stats["best_epoch"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
