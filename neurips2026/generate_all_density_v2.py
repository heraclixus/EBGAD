"""Generate correct density plots using the ACTUAL best config per dataset."""

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
from pygod.metric.metric import eval_roc_auc

os.makedirs('figures', exist_ok=True)

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}

# ACTUAL best configs per dataset (from our experiments)
# Each entry: (scoring_method, config_overrides)
actual_best = {
    'enron': {
        'method': 'ce_ratio',
        'overrides': {'rho': 1.0, 'kappa': 0.1, 'gamma': 0.2,
                      'template_type': 'affinity', 'graph_type': 'affinity'},
    },
    'weibo': {
        'method': 'control_energy',
        'overrides': {},  # use best_soc_configs as-is (has encoder)
    },
    'reddit': {
        'method': 'control_energy',
        'overrides': {},  # original config gives 62.7
    },
    'amazon': {
        'method': 'control_energy',
        'overrides': {},
    },
    'yelpchi': {
        'method': 'control_energy',
        'overrides': {},
    },
    'blogcatalog': {
        'method': 'control_energy',
        'overrides': {},
    },
    'facebook': {
        'method': 'ce_ratio',
        'overrides': {'rho': 0.5, 'kappa': 10.0, 'gamma': 0.5,
                      'template_type': 'affinity', 'graph_type': 'original',
                      'normalize_mode': 'minmax'},
    },
    'acm': {
        'method': 'control_energy',
        'overrides': {},  # use best_soc_configs (has encoder, gives 86.6 from sweep)
    },
    't_finance': {
        'method': 'control_energy',
        'overrides': {'rho': 1.0, 'kappa': 1.0, 'gamma': 0.2,
                      'template_type': 'affinity', 'graph_type': 'original'},
    },
}

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes_flat = axes.flatten()

datasets = ['enron', 'weibo', 'reddit', 'amazon', 'yelpchi',
            'blogcatalog', 'facebook', 'acm', 't_finance']

for idx, ds in enumerate(datasets):
    ax = axes_flat[idx]
    print('Processing %s...' % ds, flush=True)

    try:
        data = load_data('YelpChi' if ds == 'yelpchi' else ds)

        # Build config from best_soc_configs + overrides
        params = {k: v for k, v in best[ds]['params'].items() if k in valid_keys and k not in skip}
        method = actual_best[ds]['method']
        for k, v in actual_best[ds]['overrides'].items():
            params[k] = v
        params['epochs'] = 1 if not params.get('use_encoder', False) else params.get('epochs', 500)

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
        auc = eval_roc_auc(y_m, s_m)

        method_label = '$J^*$' if method == 'control_energy' else '$R$'

        s_log = np.log1p(s_m)
        normal = s_log[y_m == 0]
        anomaly = s_log[y_m == 1]

        clip = np.percentile(s_log, 99.5)
        normal_c = np.clip(normal, None, clip)
        anomaly_c = np.clip(anomaly, None, clip)

        x_range = np.linspace(np.percentile(s_log, 0.5), clip, 200)
        if len(normal_c) > 10 and len(anomaly_c) > 5:
            kde_n = gaussian_kde(normal_c, bw_method=0.3)
            kde_a = gaussian_kde(anomaly_c, bw_method=0.3)
            ax.fill_between(x_range, kde_n(x_range), alpha=0.5, color='steelblue', label='Normal')
            ax.fill_between(x_range, kde_a(x_range), alpha=0.5, color='coral', label='Anomaly')
        else:
            ax.hist(normal_c, bins=50, alpha=0.5, density=True, color='steelblue', label='Normal')
            ax.hist(anomaly_c, bins=50, alpha=0.5, density=True, color='coral', label='Anomaly')

        display_name = ds.replace('_', '-').capitalize()
        if ds == 'yelpchi': display_name = 'YelpChi'
        if ds == 't_finance': display_name = 'T-Finance'
        ax.set_title('%s (%s, AUC=%.1f%%)' % (display_name, method_label, auc * 100), fontsize=11)
        ax.set_xlabel('log(1+score)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)
        print('  %s: AUC=%.1f%%' % (ds, auc * 100), flush=True)

    except Exception as e:
        ax.text(0.5, 0.5, '%s\nError: %s' % (ds, str(e)[:40]),
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(ds.capitalize())
        print('  Error on %s: %s' % (ds, e), flush=True)

fig.suptitle('EB-GAD: Score density separation across datasets', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('figures/density_all_datasets_v2.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('Saved density_all_datasets_v2.pdf', flush=True)
