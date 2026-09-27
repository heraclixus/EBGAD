"""Robust EB via conditional imputation of flagged rows (scratch experiment)."""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import prepare
from ebgad.fit import fit_rho_kappa, fit_prepared, q_eigs
from ebgad.scores import compute_members, profile_specs, bank_specs, conditional_residual
name, template, gamma, pca = sys.argv[1], sys.argv[2], float(sys.argv[3]), (int(sys.argv[4]) if sys.argv[4] != 'none' else None)
data = load_data(name); y = data.y.numpy()
auc = lambda s: 100 * roc_auc_score(y, s)
prep = prepare(name, data, gamma, template=template, pca=pca, cache_dir='cache/ebgad')
V, lam, delta, n, d = prep.V, prep.lam, prep.delta, prep.n, prep.d
L = prep.L_sparse.coalesce()
Ls = sp.csr_matrix((L.values().numpy().astype(np.float64), (L.indices()[0].numpy(), L.indices()[1].numpy())), shape=(n, n))
I = sp.identity(n, format='csr')
def score_all(q, rho, kappa):
    specs = bank_specs(q, [(np.inf, np.inf)]) + profile_specs(q)
    mem = compute_members(prep, q, specs)
    mem.append(conditional_residual(prep, rho, kappa))
    out = {m.name: auc(m.score) for m in mem}
    return out, {m.name: m.score for m in mem}
fit = fit_prepared(prep); rho, kappa, q = fit.rho, fit.kappa, fit.q
res, sc = score_all(q, rho, kappa)
best = max(res, key=res.get)
print('[%s %s gamma=%g] iter 0: rho=%.3f kappa=%g  J*=%.2f P=%.2f  profile-oracle %s=%.2f' % (name, template, gamma, rho, kappa, res['J*'], res['P'], best, res[best]))
for trim in (0.05, 0.10):
    cur = sc['J*'].copy(); rho_t, kappa_t = rho, kappa
    for it in range(1, 4):
        A = np.argsort(-cur)[:int(trim * n)]; B = np.setdiff1d(np.arange(n), A)
        Q = rho_t * (kappa_t ** 2 * I + Ls) + (1 - rho_t) * I
        QAA = Q[A][:, A].tocsc(); QAB = Q[A][:, B]
        d_imp = delta.copy()
        d_imp[A] = spla.spsolve(QAA, -(QAB @ delta[B]))
        S_imp = ((V.T @ d_imp) ** 2).sum(1)
        f = fit_rho_kappa(S_imp, lam, n, d)
        rho_t, kappa_t, q_t = f['rho'], f['kappa'], f['q']
        res, sc_t = score_all(q_t, rho_t, kappa_t)
        cur = sc_t['J*']
        best = max(res, key=res.get)
        print('  trim %.0f%% iter %d: rho=%.3f kappa=%g  J*=%.2f P=%.2f R=%.2f  profile-oracle %s=%.2f' % (100 * trim, it, rho_t, kappa_t, res['J*'], res['P'], res['R'], best, res[best]))
