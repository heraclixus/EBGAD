"""Generate side-by-side ML vs AUC landscape for 2 datasets.

Shows the disconnect: ML optimum is at the boundary (tight priors)
while AUC optimum is in the interior (moderate priors).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import torch, numpy as np, json, os, sys, dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import compute_spectral_energy, Q_rho_eigenvalues
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}

rho_vals = np.linspace(0.05, 0.95, 12)
kappa_vals = np.logspace(-1.5, 1.2, 12)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for col, ds in enumerate(['enron', 'reddit']):
    print('=== %s ===' % ds, flush=True)
    data = load_data(ds)
    n, d = data.x.shape

    # Fixed discrete choices (use a reasonable template)
    tpl = 'low'
    gt = 'affinity' if ds == 'enron' else 'original'
    gamma = 0.2 if ds == 'enron' else 1.0

    # Setup once for eigenvectors
    trainer_keys = {f.name for f in dataclasses.fields(SOCTrainerConfig)}
    base_params = dict(
        template_type=tpl, graph_type=gt, gamma=gamma,
        kappa=1.0, rho=0.5, nu=1.0, laplacian_variant='sym',
        d_hidden=64, n_hidden_layers=2, epochs=0, lam_penalty=50.0,
        alpha=2.0, T=1.0,
    )
    cfg_base = SOCTrainerConfig(**{k: v for k, v in base_params.items() if k in trainer_keys})
    trainer_base = SOCTrainer(cfg_base)
    trainer_base._setup(data, device='cuda')

    x_normed = trainer_base.normalize(data.x.to('cuda').float())
    template = trainer_base.template
    V = trainer_base.V.cpu().numpy()
    lam_L = trainer_base.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer_base.graph_eigen, 'lam_L') else np.ones(V.shape[1])
    delta = (x_normed - template).cpu().numpy()
    S = compute_spectral_energy(delta, V)

    ML_grid = np.zeros((len(rho_vals), len(kappa_vals)))
    AUC_grid = np.zeros((len(rho_vals), len(kappa_vals)))

    for i, rho in enumerate(rho_vals):
        for j, kappa in enumerate(kappa_vals):
            # ML (closed-form)
            q = Q_rho_eigenvalues(lam_L, rho, kappa)
            ml = sum(-S[jj]*q[jj]/2 + (n*d/2)*np.log(max(q[jj], 1e-12)) for jj in range(len(q)))
            ML_grid[i, j] = ml

            # AUC (need to score and evaluate)
            torch.manual_seed(0); np.random.seed(0)
            try:
                cfg_eval = SOCGADConfig(
                    dataset=ds, score_method='control_energy', score_K=1,
                    d_hidden=64, n_hidden_layers=2, lr=0.001, epochs=1,
                    parameterization='score', time_weighting='importance',
                    lam_penalty=50.0, alpha=2.0, laplacian_variant='sym',
                    template_type=tpl, graph_type=gt, gamma=gamma,
                    rho=rho, kappa=kappa,
                )
                r = evaluate_single_trial(data, cfg_eval, device='cuda')
                AUC_grid[i, j] = r.auc
            except:
                AUC_grid[i, j] = 0.5

        print('  rho=%.2f done' % rho, flush=True)

    # Plot ML landscape
    ax = axes[0, col]
    im = ax.pcolormesh(kappa_vals, rho_vals, ML_grid, shading='auto', cmap='viridis')
    ax.set_xscale('log')
    ax.set_xlabel('$\\kappa$', fontsize=12)
    ax.set_ylabel('$\\rho$', fontsize=12)
    ax.set_title('%s: Marginal Likelihood' % ds.capitalize(), fontsize=12)
    plt.colorbar(im, ax=ax, label='log p(D|$\\varphi$)')
    # Mark ML optimum
    ml_max_idx = np.unravel_index(np.argmax(ML_grid), ML_grid.shape)
    ax.plot(kappa_vals[ml_max_idx[1]], rho_vals[ml_max_idx[0]], 'r*', markersize=15, label='ML optimum')
    # Mark AUC optimum on ML plot
    auc_max_idx = np.unravel_index(np.argmax(AUC_grid), AUC_grid.shape)
    ax.plot(kappa_vals[auc_max_idx[1]], rho_vals[auc_max_idx[0]], 'w^', markersize=12, label='AUC optimum')
    ax.legend(fontsize=8, loc='lower left')

    # Plot AUC landscape
    ax = axes[1, col]
    im = ax.pcolormesh(kappa_vals, rho_vals, AUC_grid * 100, shading='auto', cmap='RdYlGn')
    ax.set_xscale('log')
    ax.set_xlabel('$\\kappa$', fontsize=12)
    ax.set_ylabel('$\\rho$', fontsize=12)
    ax.set_title('%s: AUROC (%%)' % ds.capitalize(), fontsize=12)
    plt.colorbar(im, ax=ax, label='AUROC (%)')
    # Mark both optima
    ax.plot(kappa_vals[ml_max_idx[1]], rho_vals[ml_max_idx[0]], 'k*', markersize=15, label='ML optimum')
    ax.plot(kappa_vals[auc_max_idx[1]], rho_vals[auc_max_idx[0]], 'k^', markersize=12, label='AUC optimum')
    ax.legend(fontsize=8, loc='lower left')

fig.suptitle('ML optimum vs AUC optimum: the disconnect motivating bounded search', fontsize=13)
plt.tight_layout()
plt.savefig('figures/ml_vs_auc_landscape.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved ml_vs_auc_landscape.pdf', flush=True)
