"""Regenerate density separation plots with EB-GAD name."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import torch, numpy as np, json, os, sys, dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}

# Generate for Weibo and Facebook (same datasets as original)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for col_offset, ds in enumerate(['weibo', 'facebook']):
    data = load_data(ds)
    params = {k: v for k, v in best[ds]['params'].items() if k in valid_keys and k not in skip}

    # TAM scores (load from saved results if available, otherwise skip)
    # For now, generate EB-GAD scores only
    params['epochs'] = 1
    cfg = SOCGADConfig(dataset=ds, score_method='control_energy', score_K=1, **params)
    trainer = SOCTrainer(cfg.to_trainer_config())
    trainer.train(data, device='cuda')
    x_test = data.x.to('cuda').float()
    scores = compute_anomaly_scores(trainer, x_test, K=1, method='control_energy').cpu().numpy()

    y = data.y.numpy()
    mask = y >= 0
    y_m = y[mask]
    s_m = np.log1p(scores[mask])

    normal = s_m[y_m == 0]
    anomaly = s_m[y_m == 1]

    # EB-GAD density plot
    ax = axes[col_offset * 2 + 1]
    # KDE for smooth curves
    if len(normal) > 10 and len(anomaly) > 10:
        x_range = np.linspace(min(s_m), np.percentile(s_m, 99.5), 200)
        kde_n = gaussian_kde(normal, bw_method=0.3)
        kde_a = gaussian_kde(anomaly, bw_method=0.3)
        ax.fill_between(x_range, kde_n(x_range), alpha=0.5, color='steelblue', label='Normal')
        ax.fill_between(x_range, kde_a(x_range), alpha=0.5, color='coral', label='Abnormal')
    ax.set_xlabel('log(1 + score)')
    ax.set_ylabel('Density')
    ax.set_title('EB-GAD %s' % ds.capitalize())
    ax.legend(fontsize=8)

    # Placeholder for TAM (left column)
    ax_tam = axes[col_offset * 2]
    ax_tam.text(0.5, 0.5, 'TAM %s\n(from baseline)' % ds.capitalize(),
                ha='center', va='center', fontsize=11, color='gray',
                transform=ax_tam.transAxes)
    ax_tam.set_xlabel('log(1 + score)')
    ax_tam.set_ylabel('Density')
    ax_tam.set_title('TAM %s' % ds.capitalize())

plt.tight_layout()
plt.savefig('figures/density_separation_ebgad.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved density_separation_ebgad.pdf', flush=True)
