"""Generate paper figures: spectral diagnostic, CE vs CR distributions, ML landscape."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import torch, numpy as np, json, os, sys, dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import compute_spectral_energy, Q_rho_eigenvalues
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}

# =========================================================
# Figure D: CE vs CR score distributions for Elliptic
# =========================================================
print('=== Figure D: CE vs CR on Elliptic ===', flush=True)
data = load_data('elliptic')
params = {k: v for k, v in best['elliptic']['params'].items() if k in valid_keys and k not in skip}
params['epochs'] = 1
cfg = SOCGADConfig(dataset='elliptic', score_method='control_energy', score_K=1, **params)
trainer_cfg = cfg.to_trainer_config()
trainer = SOCTrainer(trainer_cfg)
trainer.train(data, device='cuda')
x_test = data.x.to('cuda').float()

ce_scores = compute_anomaly_scores(trainer, x_test, K=1, method='control_energy').cpu().numpy()
cr_scores = compute_anomaly_scores(trainer, x_test, K=1, method='ce_ratio').cpu().numpy()

y = data.y.numpy()
mask = y >= 0
y_masked = y[mask]
ce_masked = ce_scores[mask]
cr_masked = cr_scores[mask]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, scores, title in [(axes[0], ce_masked, 'Control Energy J*'),
                           (axes[1], cr_masked, 'Spectral Ratio R')]:
    normal = scores[y_masked == 0]
    anomaly = scores[y_masked == 1]
    ax.hist(normal, bins=50, alpha=0.6, label='Normal', density=True, color='steelblue')
    ax.hist(anomaly, bins=50, alpha=0.6, label='Anomaly', density=True, color='coral')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
fig.suptitle('Elliptic: Score distributions', fontsize=14)
plt.tight_layout()
plt.savefig('figures/ce_vs_cr_elliptic.pdf', bbox_inches='tight')
plt.close()
print('Saved figures/ce_vs_cr_elliptic.pdf', flush=True)

# =========================================================
# Figure A: Spectral diagnostic for 3 datasets
# =========================================================
print('=== Figure A: Spectral diagnostic ===', flush=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, ds in zip(axes, ['weibo', 'enron', 'elliptic']):
    data_ds = load_data(ds)
    params_ds = {k: v for k, v in best[ds]['params'].items() if k in valid_keys and k not in skip}
    trainer_keys = {f.name for f in dataclasses.fields(SOCTrainerConfig)}
    trainer_params = {k: v for k, v in params_ds.items() if k in trainer_keys}
    trainer_params['epochs'] = 0
    cfg_ds = SOCTrainerConfig(**trainer_params)
    trainer_ds = SOCTrainer(cfg_ds)
    trainer_ds._setup(data_ds, device='cuda')

    x_normed = trainer_ds.normalize(data_ds.x.to('cuda').float())
    template = trainer_ds.template
    V = trainer_ds.V.cpu().numpy()
    lam_L = trainer_ds.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer_ds.graph_eigen, 'lam_L') else np.ones(V.shape[1])

    delta = (x_normed - template).cpu().numpy()
    n, d = delta.shape
    S = compute_spectral_energy(delta, V)
    rho = params_ds.get('rho', 0.5)
    kappa = params_ds.get('kappa', 1.0)
    q = Q_rho_eigenvalues(lam_L, rho, kappa)
    r = q * S / (n * d)

    k = min(100, len(r))
    ax.semilogy(range(k), r[:k], 'o-', markersize=2, alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='r=1 (calibrated)')
    ax.set_xlabel('Eigenmode index')
    ax.set_ylabel('r_j')
    ax.set_title(ds.capitalize())
    ax.legend(fontsize=9)

fig.suptitle('Spectral diagnostic: per-mode calibration ratio', fontsize=14)
plt.tight_layout()
plt.savefig('figures/spectral_diagnostic.pdf', bbox_inches='tight')
plt.close()
print('Saved figures/spectral_diagnostic.pdf', flush=True)

# =========================================================
# Figure C: Marginal likelihood landscape for Enron
# =========================================================
print('=== Figure C: ML landscape for Enron ===', flush=True)
data_e = load_data('enron')
n_e, d_e = data_e.x.shape

rho_vals = np.linspace(0, 1, 20)
kappa_vals = np.logspace(-2, 1.5, 20)

trainer_params_e = {k: v for k, v in best['enron']['params'].items() if k in trainer_keys}
trainer_params_e['epochs'] = 0
trainer_params_e['template_type'] = 'affinity'
trainer_params_e['graph_type'] = 'affinity'
trainer_params_e['gamma'] = 0.2
cfg_e = SOCTrainerConfig(**trainer_params_e)
trainer_e = SOCTrainer(cfg_e)
trainer_e._setup(data_e, device='cuda')
x_normed_e = trainer_e.normalize(data_e.x.to('cuda').float())
template_e = trainer_e.template
V_e = trainer_e.V.cpu().numpy()
lam_L_e = trainer_e.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer_e.graph_eigen, 'lam_L') else np.ones(V_e.shape[1])
delta_e = (x_normed_e - template_e).cpu().numpy()
S_e = compute_spectral_energy(delta_e, V_e)

ML = np.zeros((len(rho_vals), len(kappa_vals)))
for i, rho in enumerate(rho_vals):
    for j, kappa in enumerate(kappa_vals):
        q = Q_rho_eigenvalues(lam_L_e, rho, kappa)
        ml = sum(-S_e[jj]*q[jj]/2 + (n_e*d_e/2)*np.log(max(q[jj], 1e-12)) for jj in range(len(q)))
        ML[i, j] = ml

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(kappa_vals, rho_vals, ML, shading='auto', cmap='viridis')
ax.set_xscale('log')
ax.set_xlabel('kappa')
ax.set_ylabel('rho')
ax.set_title('Enron: Log marginal likelihood landscape')
plt.colorbar(im, ax=ax, label='log p(D|phi)')
plt.tight_layout()
plt.savefig('figures/ml_landscape_enron.pdf', bbox_inches='tight')
plt.close()
print('Saved figures/ml_landscape_enron.pdf', flush=True)

print('=== All visualizations done ===', flush=True)
