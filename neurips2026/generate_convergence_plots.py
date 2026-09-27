"""Generate convergence and landscape plots for Algorithm 1."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import torch, numpy as np, json, os, sys, dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_gad import SOCGADConfig
from soc.prior_optimizer import (
    compute_spectral_energy, Q_rho_eigenvalues,
    marginal_log_likelihood, predictive_variance,
    optimize_rho, optimize_kappa,
)
from data_utils import load_data

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}

# =========================================================
# Helper: compute ML for given (rho, kappa) with fixed S, lam_L
# =========================================================
def compute_ml(S, lam_L, rho, kappa, n, d, nu=1.0):
    q = Q_rho_eigenvalues(lam_L, rho, kappa, nu)
    return sum(-S[j]*q[j]/2 + (n*d/2)*np.log(max(q[j], 1e-12)) for j in range(len(q)))


# =========================================================
# Setup: compute S, lam_L for 3 datasets
# =========================================================
datasets_to_plot = ['enron', 'weibo', 'reddit']
dataset_cache = {}

for ds in datasets_to_plot:
    print('Setting up %s...' % ds, flush=True)
    data = load_data(ds)
    n, d = data.x.shape

    trainer_keys = {f.name for f in dataclasses.fields(SOCTrainerConfig)}
    params = {k: v for k, v in best[ds]['params'].items() if k in trainer_keys}
    params['epochs'] = 0
    cfg = SOCTrainerConfig(**params)
    trainer = SOCTrainer(cfg)
    trainer._setup(data, device='cuda')

    x_normed = trainer.normalize(data.x.to('cuda').float())
    delta = (x_normed - trainer.template).cpu().numpy()
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
    S = compute_spectral_energy(delta, V)

    dataset_cache[ds] = {'S': S, 'lam_L': lam_L, 'n': n, 'd': d}


# =========================================================
# Figure 1: Convergence of coordinate descent
# =========================================================
print('=== Convergence plot ===', flush=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, ds in zip(axes, datasets_to_plot):
    cache = dataset_cache[ds]
    S, lam_L, n, d = cache['S'], cache['lam_L'], cache['n'], cache['d']

    # Run coordinate descent and record ML at each step
    rho, kappa = 0.5, 1.0  # initial
    ml_history = [compute_ml(S, lam_L, rho, kappa, n, d)]

    for iteration in range(15):
        # Optimize rho
        rho, _ = optimize_rho(S, lam_L, kappa, 1.0, 50.0, 2.0, n, d)
        ml_history.append(compute_ml(S, lam_L, rho, kappa, n, d))

        # Optimize kappa
        kappa, _ = optimize_kappa(S, lam_L, rho, 1.0, 50.0, 2.0, n, d)
        ml_history.append(compute_ml(S, lam_L, rho, kappa, n, d))

    steps = np.arange(len(ml_history))
    ax.plot(steps, ml_history, 'o-', markersize=3, color='steelblue')
    ax.set_xlabel('Optimization step')
    ax.set_ylabel('log p(D|$\\varphi$)')
    ax.set_title(ds.capitalize())
    # Mark convergence
    final = ml_history[-1]
    ax.axhline(y=final, color='red', linestyle='--', alpha=0.3)

fig.suptitle('Algorithm 1: Coordinate descent convergence (monotonic increase)', fontsize=13)
plt.tight_layout()
plt.savefig('figures/convergence.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved convergence.pdf', flush=True)


# =========================================================
# Figure 2: 1D profiles of ML vs rho and kappa
# =========================================================
print('=== 1D landscape profiles ===', flush=True)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

rho_grid = np.linspace(0.01, 0.99, 50)
kappa_grid = np.logspace(-2, 1.5, 50)

for col, ds in enumerate(datasets_to_plot):
    cache = dataset_cache[ds]
    S, lam_L, n, d = cache['S'], cache['lam_L'], cache['n'], cache['d']

    # Fix kappa at multiple values, sweep rho
    ax = axes[0, col]
    for kappa_fixed in [0.1, 1.0, 5.0]:
        ml_vals = [compute_ml(S, lam_L, rho, kappa_fixed, n, d) for rho in rho_grid]
        ax.plot(rho_grid, ml_vals, label='$\\kappa$=%.1f' % kappa_fixed)
    ax.set_xlabel('$\\rho$')
    ax.set_ylabel('log p(D|$\\varphi$)')
    ax.set_title('%s: ML vs $\\rho$' % ds.capitalize())
    ax.legend(fontsize=8)

    # Fix rho at multiple values, sweep kappa
    ax = axes[1, col]
    for rho_fixed in [0.0, 0.5, 1.0]:
        ml_vals = [compute_ml(S, lam_L, rho_fixed, kappa, n, d) for kappa in kappa_grid]
        ax.plot(kappa_grid, ml_vals, label='$\\rho$=%.1f' % rho_fixed)
    ax.set_xlabel('$\\kappa$')
    ax.set_ylabel('log p(D|$\\varphi$)')
    ax.set_title('%s: ML vs $\\kappa$' % ds.capitalize())
    ax.set_xscale('log')
    ax.legend(fontsize=8)

fig.suptitle('Marginal likelihood profiles: unimodal in each coordinate', fontsize=13)
plt.tight_layout()
plt.savefig('figures/ml_profiles.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved ml_profiles.pdf', flush=True)

print('=== All done ===', flush=True)
