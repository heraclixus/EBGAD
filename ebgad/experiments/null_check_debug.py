import sys, numpy as np
sys.path.insert(0, '.')
from scipy.stats import chi2, kstest
from data_utils import load_data
from ebgad.prep import prepare
from ebgad.fit import fit_prepared
from ebgad.scores import compute_bank, bank_weights
from ebgad.nulls import simulate_residual
data = load_data('questions')
prep = prepare('questions', data, 1.0, cache_dir='cache/ebgad')
fit = fit_prepared(prep); q = fit.q
V = prep.V; d = prep.d; k = prep.k
G = V.T @ V
print('k', k, 'd', d, 'truncated', prep.truncated, 'max|VtV-I|', np.abs(G - np.eye(k)).max(), 'max lev', prep.lev.max())
rng = np.random.default_rng(1)
Xi = rng.standard_normal((k, d)) / np.sqrt(q)[:, None]
delta = V @ Xi
Xi_back = V.T @ delta
print('max|Xi_back - Xi|', np.abs(Xi_back - Xi).max())
c = bank_weights(q, np.inf, np.inf)
u = V @ (np.sqrt(c)[:, None] * Xi)
energy = (u*u).sum(1); sigma2 = (V*V) @ (c/np.maximum(q,1e-12))
x = energy / sigma2
print('mean x / d =', x.mean()/d, ' var x / (2d) =', x.var()/(2*d))
p = chi2.sf(x, d)
print('KS J*', kstest(p, 'uniform').statistic)
# per-node check on a few nodes with B draws
B = 400
idx = np.array([0, 100, 1000, 5000, 20000])
acc = np.zeros((B, len(idx)))
for b in range(B):
    Xi_b = rng.standard_normal((k, d)) / np.sqrt(q)[:, None]
    u_b = V[idx] @ (np.sqrt(c)[:, None] * Xi_b)
    acc[b] = (u_b*u_b).sum(1) / sigma2[idx]
print('per-node mean x/d over B draws:', np.round(acc.mean(0)/d, 4))
# now the full simulate_residual path
delta2 = simulate_residual(prep, q, rng, rough='isotropic')
Xi2 = V.T @ delta2
u2 = V @ (np.sqrt(c)[:, None] * Xi2)
x2 = (u2*u2).sum(1)/sigma2
print('simulate_residual path: mean x/d =', x2.mean()/d, 'KS', kstest(chi2.sf(x2, d), 'uniform').statistic)
