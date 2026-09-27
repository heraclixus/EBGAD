"""Generate score density separation plots for ALL datasets.

For each dataset, shows normal vs anomaly score distributions using the
best scoring method (CE or CR). Helps visualize where separation succeeds/fails.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import torch, numpy as np, json, os, sys, dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig
from soc.soc_trainer import SOCTrainer
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}

# Which scoring method to use per dataset (based on our best results)
best_method = {
    'enron': 'ce_ratio',
    'weibo': 'control_energy',
    'reddit': 'control_energy',
    'amazon': 'control_energy',
    'yelpchi': 'control_energy',
    'blogcatalog': 'control_energy',
    'facebook': 'ce_ratio',
    'acm': 'control_energy',
    't_finance': 'control_energy',
    'elliptic': 'ce_ratio',
    'elliptic_plus_plus': 'ce_ratio',
}

# A100-sized datasets
a100_datasets = ['enron', 'weibo', 'reddit', 'amazon', 'yelpchi',
                 'blogcatalog', 'facebook', 'acm', 't_finance']

# Layout: 3 rows x 3 cols for 9 datasets (skip large H100 ones for now)
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes_flat = axes.flatten()

for idx, ds in enumerate(a100_datasets):
    ax = axes_flat[idx]
    print('Processing %s...' % ds, flush=True)

    try:
        data = load_data('YelpChi' if ds == 'yelpchi' else ds)
        params = {k: v for k, v in best[ds]['params'].items() if k in valid_keys and k not in skip}
        params['epochs'] = 1
        method = best_method.get(ds, 'control_energy')

        torch.manual_seed(0); np.random.seed(0)
        cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **params)
        trainer = SOCTrainer(cfg.to_trainer_config())
        trainer.train(data, device='cuda')
        x_test = data.x.to('cuda').float()
        scores = compute_anomaly_scores(trainer, x_test, K=1, method=method).cpu().numpy()

        y = data.y.numpy()
        mask = y >= 0
        y_m = y[mask]
        s_m = scores[mask]

        # Compute AUC for the title
        auc = eval_roc_auc(y_m, s_m)
        method_label = '$J^*$' if method == 'control_energy' else '$R$'

        # Log transform for better visualization
        s_log = np.log1p(s_m)
        normal = s_log[y_m == 0]
        anomaly = s_log[y_m == 1]

        # Clip to 99.5th percentile
        clip = np.percentile(s_log, 99.5)
        normal_c = np.clip(normal, None, clip)
        anomaly_c = np.clip(anomaly, None, clip)

        # KDE
        x_range = np.linspace(np.percentile(s_log, 0.5), clip, 200)
        if len(normal_c) > 10 and len(anomaly_c) > 5:
            bw = 0.3
            kde_n = gaussian_kde(normal_c, bw_method=bw)
            kde_a = gaussian_kde(anomaly_c, bw_method=bw)
            ax.fill_between(x_range, kde_n(x_range), alpha=0.5, color='steelblue', label='Normal')
            ax.fill_between(x_range, kde_a(x_range), alpha=0.5, color='coral', label='Anomaly')
        else:
            ax.hist(normal_c, bins=50, alpha=0.5, density=True, color='steelblue', label='Normal')
            ax.hist(anomaly_c, bins=50, alpha=0.5, density=True, color='coral', label='Anomaly')

        ax.set_title('%s (%s, AUC=%.1f%%)' % (ds.capitalize(), method_label, auc * 100), fontsize=11)
        ax.set_xlabel('log(1+score)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)

    except Exception as e:
        ax.text(0.5, 0.5, '%s\nError: %s' % (ds, str(e)[:30]),
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(ds.capitalize())
        print('  Error on %s: %s' % (ds, e), flush=True)

fig.suptitle('EB-GAD: Score density separation across datasets', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('figures/density_all_datasets.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved density_all_datasets.pdf', flush=True)
