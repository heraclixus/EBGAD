"""Generate improved paper figures."""

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
# Figure D v2: CE vs CR with log/clipped scales
# =========================================================
print('=== Figure D v2 ===', flush=True)
data = load_data('elliptic')
params = {k: v for k, v in best['elliptic']['params'].items() if k in valid_keys and k not in skip}
params['epochs'] = 1
cfg = SOCGADConfig(dataset='elliptic', score_method='control_energy', score_K=1, **params)
trainer = SOCTrainer(cfg.to_trainer_config())
trainer.train(data, device='cuda')
x_test = data.x.to('cuda').float()

ce_scores = compute_anomaly_scores(trainer, x_test, K=1, method='control_energy').cpu().numpy()
cr_scores = compute_anomaly_scores(trainer, x_test, K=1, method='ce_ratio').cpu().numpy()

y = data.y.numpy()
mask = y >= 0
y_m = y[mask]
ce_m = ce_scores[mask]
cr_m = cr_scores[mask]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: CE with log scale
ax = axes[0]
normal = np.log1p(ce_m[y_m == 0])
anomaly = np.log1p(ce_m[y_m == 1])
ax.hist(normal, bins=80, alpha=0.6, label='Normal', density=True, color='steelblue')
ax.hist(anomaly, bins=80, alpha=0.6, label='Anomaly', density=True, color='coral')
ax.set_title('Control Energy $J^*$ (log scale)', fontsize=12)
ax.set_xlabel('log(1 + $J^*$)')
ax.set_ylabel('Density')
ax.legend(fontsize=10)

# Right: CR clipped to 99th percentile
ax = axes[1]
clip = np.percentile(cr_m, 99.5)
normal = np.clip(cr_m[y_m == 0], 0, clip)
anomaly = np.clip(cr_m[y_m == 1], 0, clip)
ax.hist(normal, bins=80, alpha=0.6, label='Normal', density=True, color='steelblue')
ax.hist(anomaly, bins=80, alpha=0.6, label='Anomaly', density=True, color='coral')
ax.set_title('Spectral Control Ratio $R$', fontsize=12)
ax.set_xlabel('$R$')
ax.set_ylabel('Density')
ax.legend(fontsize=10)

fig.suptitle('Elliptic: $J^*$ cannot separate, $R$ succeeds', fontsize=13)
plt.tight_layout()
plt.savefig('figures/ce_vs_cr_elliptic_v2.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved ce_vs_cr_elliptic_v2.pdf', flush=True)

print('=== Done ===', flush=True)
