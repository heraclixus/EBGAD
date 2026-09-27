"""Does an algebraic-multigrid preconditioner keep CG fast for strongly smoothed priors?
Solves B y = x, B = (1 + c kappa^2) I + c L, with plain CG and with PCG preconditioned by one
smoothed-aggregation AMG hierarchy built once for L + delta I (reused for every c).
Usage: PYTHONPATH=. python ebgad/experiments/amg_precond_test.py <dataset> [cols]"""
import sys, time, numpy as np, scipy.sparse as sp, pyamg
from data_utils import load_data
from ebgad.prep import preprocess_features, permute_data
from ebgad.ebsmooth import laplacian_csr, rcm_order, block_cg

name = sys.argv[1]; cols = int(sys.argv[2]) if len(sys.argv) > 2 else None
data = load_data(name); n = data.num_nodes
perm = rcm_order(laplacian_csr(data.edge_index, n)); data = permute_data(data, perm)
x = preprocess_features(data.x, data.edge_index).double().numpy()
if cols: x = x[:, :cols]
L = laplacian_csr(data.edge_index, n); d = x.shape[1]
print(f"[{name}] n={n} d={d} nnz={L.nnz}", flush=True)
t0 = time.time()
ml = pyamg.smoothed_aggregation_solver(L + 1e-3 * sp.eye(n), symmetry="symmetric", max_coarse=500)
print(f"AMG setup {time.time()-t0:.1f}s; levels={len(ml.levels)} complexity={ml.operator_complexity():.2f}", flush=True)

def pcg(Bmv, X, M, tol=1e-7, maxiter=500):
    """Preconditioned CG, columns independent; M applies the preconditioner to one vector."""
    Y = np.zeros_like(X); R = X.copy(); Z = np.column_stack([M(R[:, j]) for j in range(X.shape[1])]); P = Z.copy()
    rz = (R * Z).sum(0); bnorm = np.sqrt((X * X).sum(0)) + 1e-300
    for it in range(1, maxiter + 1):
        AP = Bmv(P); pap = (P * AP).sum(0)
        alpha = np.where(pap > 0, rz / np.where(pap > 0, pap, 1.0), 0.0)
        Y += alpha * P; R -= alpha * AP
        if np.all(np.sqrt((R * R).sum(0)) <= tol * bnorm): return Y, it
        Z = np.column_stack([M(R[:, j]) for j in range(X.shape[1])]); rz_new = (R * Z).sum(0)
        beta = np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
        P = Z + beta * P; rz = rz_new
    return Y, maxiter

M = lambda r: ml.solve(r, tol=1e-1, maxiter=1, cycle="V", accel=None)   # one V-cycle
for c, k2 in ((12.8, 0.0015), (300.0, 0.00014), (3000.0, 0.0001), (30000.0, 0.00003)):
    Bmv = lambda Y, c=c, k2=k2: (1.0 + c * k2) * Y + c * (L @ Y)
    t0 = time.time(); y1, it1 = block_cg(Bmv, x, tol=1e-7, maxiter=500); t1 = time.time() - t0
    t0 = time.time(); y2, it2 = pcg(Bmv, x, M, tol=1e-7); t2 = time.time() - t0
    res = np.linalg.norm(Bmv(y2) - x) / np.linalg.norm(x)
    print(f"c={c:>7g} kappa2={k2:.0e}: plain CG {it1:3d} iters {t1:6.1f}s | AMG-PCG {it2:3d} iters {t2:6.1f}s (rel. residual {res:.1e}) | agree {np.allclose(y1, y2, rtol=1e-4, atol=1e-6) if it1 < 500 else 'plain hit cap'}", flush=True)
