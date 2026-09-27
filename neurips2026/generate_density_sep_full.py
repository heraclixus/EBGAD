"""Regenerate density separation: EB-GAD vs TAM on Weibo and Facebook.

Runs both TAM and EB-GAD scoring, generates side-by-side density plots.
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
from run_tam import run_tam_trial

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}


def plot_density(ax, scores, y, title):
    """Plot KDE density for normal vs anomaly."""
    mask = y >= 0
    y_m = y[mask]
    s_m = np.log1p(scores[mask])
    normal = s_m[y_m == 0]
    anomaly = s_m[y_m == 1]

    x_range = np.linspace(
        np.percentile(s_m, 0.5),
        np.percentile(s_m, 99.5), 200)
    kde_n = gaussian_kde(normal, bw_method=0.3)
    kde_a = gaussian_kde(anomaly, bw_method=0.3)
    ax.fill_between(x_range, kde_n(x_range), alpha=0.5, color='steelblue', label='Normal')
    ax.fill_between(x_range, kde_a(x_range), alpha=0.5, color='coral', label='Abnormal')
    ax.set_xlabel('log(1 + score)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for col_offset, ds in enumerate(['weibo', 'facebook']):
    data = load_data(ds if ds != 'facebook' else 'Facebook')
    y_np = data.y.numpy()

    # --- TAM scores ---
    print('Running TAM on %s...' % ds, flush=True)
    try:
        tam_result = run_tam_trial(data, device='cuda')
        tam_scores = tam_result['scores']
        plot_density(axes[col_offset * 2], tam_scores, y_np,
                     'TAM \u2014 %s' % ds.capitalize())
    except Exception as e:
        print('TAM failed on %s: %s' % (ds, e))
        axes[col_offset * 2].text(0.5, 0.5, 'TAM unavailable',
                                   ha='center', va='center', transform=axes[col_offset*2].transAxes)
        axes[col_offset * 2].set_title('TAM \u2014 %s' % ds.capitalize())

    # --- EB-GAD scores ---
    print('Running EB-GAD on %s...' % ds, flush=True)
    params = {k: v for k, v in best[ds.lower()]['params'].items() if k in valid_keys and k not in skip}
    params['epochs'] = 1
    torch.manual_seed(0); np.random.seed(0)
    cfg = SOCGADConfig(dataset=ds.lower(), score_method='control_energy', score_K=1, **params)
    trainer = SOCTrainer(cfg.to_trainer_config())
    trainer.train(data, device='cuda')
    x_test = data.x.to('cuda').float()
    eb_scores = compute_anomaly_scores(trainer, x_test, K=1, method='control_energy').cpu().numpy()
    plot_density(axes[col_offset * 2 + 1], eb_scores, y_np,
                 'EB-GAD \u2014 %s' % ds.capitalize())

plt.tight_layout()
plt.savefig('figures/density_separation_ebgad_vs_tam.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved density_separation_ebgad_vs_tam.pdf', flush=True)
