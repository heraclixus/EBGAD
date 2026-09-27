"""Run GADAM (Chen et al., ICLR 2024; github.com/PasaLab/GADAM) on our datasets with our evaluation sets.

Uses the authors' code (model.LocalModel / GlobalModel, run.train_local, run.load_info_from_local) and the
README hyperparameters (local lr 1e-3, 100 local epochs, global lr 5e-4, 50 global epochs, hidden 64, seed 717);
no label tuning. The global training loop of run.py is reproduced here so that the final mixed score
(-(global score + local positive score), as in train_global) can be evaluated on our evaluated nodes instead of
all nodes. Features and graph as loaded (the model row-normalizes internally; no self loops, as in main()).
Each dataset runs in its own working directory because the authors' code writes checkpoints to the CWD.
Usage (needs DGL): GADAM_DIR=third_party/GADAM python baselines/run_gadam_ours.py acm weibo ...
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json, time, argparse, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
GADAM_DIR = os.environ.get("GADAM_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "GADAM"))
sys.path.insert(0, GADAM_DIR)
import dgl                                                         # noqa: E402
from model import LocalModel, GlobalModel                          # noqa: E402
from utils import seed_everything                                  # noqa: E402
import run as gadam_run                                            # noqa: E402
import model as gadam_model                                        # noqa: E402


def idx_sample(idxes):
    """The authors' idx_sample with the size argument torch 2 requires (same semantics: a random cyclic shift)."""
    num_idx = len(idxes); random_add = int(torch.randint(low=1, high=num_idx, size=(1,)))
    return torch.remainder(torch.arange(0, num_idx) + random_add, num_idx)


gadam_model.idx_sample = idx_sample
from data_utils import load_data                                   # noqa: E402
from run_ebgad import eval_mask                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--out", default="results/baselines/gadam")
    ap.add_argument("--workdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "work_gadam")); cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True); out = os.path.abspath(cli.out)
    for ds in cli.datasets:
        t0 = time.time(); data = load_data(ds); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]; n = data.num_nodes
        wd = os.path.join(cli.workdir, ds); os.makedirs(wd, exist_ok=True); os.chdir(wd)
        args = argparse.Namespace(data=ds, seed=717, dropout=0.0, gpu=-1, local_lr=1e-3, global_lr=5e-4, local_epochs=100, global_epochs=50,
                                  out_dim=64, train_ratio=0.05, weight_decay=0.0, patience=20, self_loop=True)
        seed_everything(args.seed)
        ei = data.edge_index; src = torch.cat([ei[0], ei[1]]); dst = torch.cat([ei[1], ei[0]])
        graph = dgl.to_simple(dgl.graph((src, dst), num_nodes=n).remove_self_loop())
        graph.ndata["feat"] = data.x.float(); graph.ndata["label"] = torch.as_tensor(y_all).long()
        feats = graph.ndata["feat"]; in_feats = feats.shape[1]
        local_net = LocalModel(graph, in_feats, args.out_dim, nn.PReLU())
        local_opt = torch.optim.Adam(local_net.parameters(), lr=args.local_lr, weight_decay=args.weight_decay)
        gadam_run.train_local(local_net, graph, feats, local_opt, args)
        memo, nor_idx, ano_idx, center = gadam_run.load_info_from_local(local_net, args.gpu)
        graph = memo["graph"]
        global_net = GlobalModel(graph, in_feats, args.out_dim, nn.PReLU(), nor_idx, ano_idx, center)
        opt = torch.optim.Adam(global_net.parameters(), lr=args.global_lr, weight_decay=args.weight_decay)
        pos = graph.ndata["pos"]
        for mod in global_net.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
        last = None
        for epoch in range(args.global_epochs):                           # train_global of run.py, scores kept
            global_net.train(); opt.zero_grad(); loss, scores = global_net(feats, epoch); loss.backward(); opt.step()
            last = -(scores + pos).detach().cpu().numpy()
        auc, ap_ = 100 * roc_auc_score(y, last[m]), 100 * average_precision_score(y, last[m])
        rec = dict(dataset=ds, method="GADAM", auroc=auc, auprc=ap_, config=dict(local_lr=1e-3, local_epochs=100, global_lr=5e-4, global_epochs=50, out_dim=64, seed=717),
                   config_provenance="README defaults", n=int(n), n_eval=int(m.sum()), seconds=time.time() - t0)
        json.dump(rec, open(os.path.join(out, f"{ds}.json"), "w"), indent=1); np.save(os.path.join(out, f"{ds}_scores.npy"), last)
        print("%-20s GADAM AUROC %5.1f AUPRC %5.1f | README defaults | %.0f s" % (ds, auc, ap_, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
