"""Band-energy diagnostics for the v3 surprise score (why does d_eff collapse?)."""
import sys, json, numpy as np
sys.path.insert(0, '.')
from data_utils import load_data
from ebgad.prep import prepare
from ebgad.fit import q_eigs
from ebgad.surprise import band_energies, fit_hierarchical_null
qs = [1, 10, 50, 90, 99]
for ds, pca in [('elliptic', None), ('t_finance', None), ('weibo', 64)]:
    r = json.load(open(f'results/ebgad_v3/{ds}.json'))
    data = load_data(ds); y = data.y.numpy(); m = (y == 0) | (y == 1)
    prep = prepare(ds, data, r['gamma'], template='low_unit', pca=pca, cache_dir='cache/ebgad')
    q = q_eigs(prep.lam, r['rho'], r['kappa'])
    E, sig2, names = band_energies(prep, q, n_bands=4)
    e = E / sig2; le = np.log(np.maximum(e, 1e-300))
    print('\n== %s: n=%d d=%d k=%d truncated=%s ==' % (ds, prep.n, prep.d, prep.k, prep.truncated))
    print('  %-10s %8s %8s %8s %8s %8s | %10s %10s %10s' % ('band', 'q01', 'q10', 'q50', 'q90', 'q99', 'frac<1e-6', 'med normal', 'med anom'))
    for a, nm in enumerate(names):
        v = le[m, a]
        print('  %-10s %8.2f %8.2f %8.2f %8.2f %8.2f | %10.3f %10.2f %10.2f' % (nm, *np.percentile(v, qs), (e[m, a] < 1e-6).mean(), np.median(le[m & (y == 0), a]), np.median(le[m & (y == 1), a])))
    T = le.shape[1]
    iu = np.triu_indices(T, 1)
    con = (le[m][:, :, None] - le[m][:, None, :])[:, iu[0], iu[1]]
    print('  contrast log e_a - log e_b: MAD-based sd per pair:', np.round(1.4826 * np.median(np.abs(con - np.median(con, 0)), 0), 2))
    print('  fit:', {k: round(v, 3) for k, v in fit_hierarchical_null(e, prep.d, mask=m).items()})
    # feature covariance rank of the residual (is d_eff small because features are redundant?)
    C = np.cov(prep.delta[m].T); w = np.linalg.eigvalsh(C)[::-1]; w = w / w.sum()
    print('  residual feature spectrum: top-1 share %.2f, dims for 90%% var %d of %d, participation ratio %.1f' % (w[0], int(np.searchsorted(np.cumsum(w), 0.9)) + 1, prep.d, 1.0 / (w ** 2).sum()))
