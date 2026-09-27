"""Camouflage statistic with a polynomial (hop-form) low-pass template: no eigendecomposition.
Template m = T_h x with T_h = (I + L_sym)^{-1}-like heat/hop filters approximated by
(a) h-step symmetric random walk with self-loops (GCN filter) and (b) truncated heat kernel exp(-t L).
Isolated nodes: template undefined -> population mean (delta = x)."""
import sys, numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
ds = sys.argv[1]
data = load_data(ds); n = data.num_nodes
ei = data.edge_index.numpy()
A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
A = ((A + A.T) > 0).astype(np.float64)
deg = np.asarray(A.sum(1)).ravel()
x = data.x.numpy().astype(np.float64); x = (x - x.mean(0)) / (x.std(0) + 1e-12)
from run_ebgad import eval_mask
y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
iso = deg == 0
def au(s): return 100 * roc_auc_score(y, s[m]), 100 * average_precision_score(y, s[m])
def rep(name, delta):
    e = (delta * delta).sum(1)
    a, p = au(-e); print(f"  {name:34s} AUROC {a:5.1f} AUPRC {p:5.1f}   spearman(-e,-||x||^2)={spearmanr(-e[m], -(x*x).sum(1)[m]).correlation:+.3f}")
print(f"{ds}: n={n} eval={m.sum()} prev={y.mean():.3f} isolated frac={iso.mean():.3f}")
rep("no graph: delta = x", x)
# (a) GCN-style symmetric normalized adjacency with self loops, h hops
d1 = deg + 1.0
Dm = sp.diags(1.0 / np.sqrt(d1)); As = Dm @ (A + sp.eye(n)) @ Dm
# neighbor-only mean (no self loop): m_i = mean of neighbors; isolated -> 0
Dn = sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)); P1 = Dn @ A
for h in (1, 2, 4, 8):
    z = x.copy()
    for _ in range(h): z = As @ z
    rep(f"GCN filter h={h}: delta = x - As^h x", x - z)
z = P1 @ x
rep("neighbor mean: delta = x - mean(nbrs)", x - z)
z2 = P1 @ z
rep("2-hop neighbor mean", x - z2)
# (b) heat kernel exp(-t L_sym) via Taylor (Krylov), t in {0.5, 1, 2, 4}
L = sp.eye(n) - Dm @ A @ Dm  # normalized Laplacian without self loops (isolated: L_ii=1 -> exp(-t) shrink; handle below)
for t in (0.5, 1.0, 2.0, 4.0):
    term = x.copy(); z = x.copy()
    for k in range(1, 30):
        term = (-t / k) * (L @ term); z = z + term
        if np.abs(term).max() < 1e-10: break
    z[iso] = 0.0
    rep(f"heat t={t}: delta = x - exp(-tL) x", x - z)
