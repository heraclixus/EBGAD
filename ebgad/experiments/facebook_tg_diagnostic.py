"""Why did the old pipeline reach 91.4 on Facebook? Diagnostic over two-groups variants.

Computes, on the same prepared field: two-groups over the shape channel
(TG-R), over both channels jointly, the responsibility-entropy score the
old code used (tg_component_entropy), and single members, under the unit
template (x-likelihood gamma) and the paper's template (gamma 0.7, PCA 64).
"""
import sys, numpy as np
sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import prepare
from ebgad.fit import fit_prepared, select_gamma
from ebgad.scores import bank_specs, profile_specs, compute_members, conditional_residual
from ebgad.nulls import member_tails
from ebgad.select import z_from_tails
from ebgad.twogroups import two_groups_channel, calibrated_pvalues, fit_two_groups

data = load_data('facebook'); y = data.y.numpy()
auc = lambda s: 100 * roc_auc_score(y, np.nan_to_num(s))
for label, tmpl, gammas, pca in [('unit, EB gamma', 'low_unit', [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0], None),
                                  ('paper: low gamma .7 pca 64', 'low', [0.7], 64)]:
    cache = {}
    fits, preps = [], {}
    for g in gammas:
        p = prepare('facebook', data, g, template=tmpl, pca=pca, cache_dir='cache/ebgad', _spectrum_cache=cache)
        fits.append(fit_prepared(p)); preps[g] = p
    fit = select_gamma(fits, use_jacobian=tmpl.endswith('_unit')); prep = preps[fit.gamma]; q = fit.q
    specs = bank_specs(q, [(np.inf, np.inf)]) + profile_specs(q)
    members = compute_members(prep, q, specs) + [conditional_residual(prep, fit.rho, fit.kappa)]
    zs = {}
    for m in members:
        try:
            sf, cdf = member_tails(m, prep.d)
        except NotImplementedError:
            continue
        zs[m.name] = (m.kind, z_from_tails(sf, cdf))
    ratio = {k: v[1] for k, v in zs.items() if v[0] == 'ratio'}
    energy = {k: np.log(np.maximum(m.score, 1e-300) / np.maximum(m.null_mean(prep.d), 1e-300)) for m in members if m.kind in ('energy', 'cond') for k in [m.name]}
    print('\n== %s == gamma=%g rho=%.3f kappa=%g' % (label, fit.gamma, fit.rho, fit.kappa))
    print('  singles: ' + ', '.join('%s %.1f' % (m.name, auc(m.score)) for m in members if m.name in ('J*', 'R', 'P', 'DR@0', 'DR@0.03', 'DR@1e-05')))
    for tag, zd in (('ratio only', ratio), ('both channels', {**ratio, **energy})):
        f = two_groups_channel(zd)
        if f is None: continue
        # responsibilities and their entropy
        P = np.column_stack([calibrated_pvalues(zd[n]) for n in f['members']])
        logp = np.log(np.clip(P, 1e-300, 1)); a, w = f['a'], f['w']
        log_comp = np.log(w)[None, :] + np.log(a)[None, :] + (a[None, :] - 1) * logp
        c = np.exp(log_comp - log_comp.max(1, keepdims=True)); c /= c.sum(1, keepdims=True)
        ent = -(c * np.log(np.maximum(c, 1e-300))).sum(1)
        print('  TG %-14s pi=%.3f  posterior %.1f  logBF %.1f  resp-entropy %.1f  (-entropy %.1f)' % (tag, f['pi'], auc(f['posterior']), auc(f['log_bayes_factor']), auc(ent), auc(-ent)))
