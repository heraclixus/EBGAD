"""Final paper results: 10-trial evaluation + density plots from saved best configs.

Run AFTER test_prior_optimizer_v3.py has saved eb_gad_best_configs.json.
Produces:
1. Per-dataset 10-trial AUC mean ± std
2. Density separation plots for all datasets
3. Summary table ready for LaTeX
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch
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
import dataclasses


def load_configs():
    """Load best configs from v3 optimizer output."""
    configs = {}
    for path in ['results/eb_gad_best_configs.json', 'results/eb_gad_best_configs_h100.json']:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                for ds, rec in data.items():
                    configs[ds] = rec
    return configs


def run_trials(ds, cfg_dict, n_trials=10, device='cuda'):
    """Run n_trials and return AUCs."""
    data = load_data('YelpChi' if ds == 'yelpchi' else ds)
    config = cfg_dict['best_config']
    valid_keys = {f.name for f in dataclasses.fields(SOCGADConfig)}

    aucs = []
    for seed in range(n_trials):
        torch.manual_seed(seed)
        np.random.seed(seed)
        params = {k: v for k, v in config.items() if k in valid_keys}
        cfg = SOCGADConfig(dataset=ds, **params)
        r = evaluate_single_trial(data, cfg, device=device)
        aucs.append(r.auc)

    return aucs


def plot_density(ax, data, cfg_dict, ds, device='cuda'):
    """Plot score density for one dataset."""
    config = cfg_dict['best_config']
    valid_keys = {f.name for f in dataclasses.fields(SOCGADConfig)}

    torch.manual_seed(0); np.random.seed(0)
    params = {k: v for k, v in config.items() if k in valid_keys}
    cfg = SOCGADConfig(dataset=ds, **params)
    trainer = SOCTrainer(cfg.to_trainer_config())
    trainer.train(data, device=device)
    x_test = data.x.to(device).float()

    method = config.get('score_method', 'precision_energy')
    scores = compute_anomaly_scores(trainer, x_test, K=1, method=method).cpu().numpy()

    y = data.y.numpy()
    mask = y >= 0
    y_m = y[mask]
    s_m = scores[mask]
    auc = eval_roc_auc(y_m, s_m)

    s_log = np.log1p(s_m)
    normal = s_log[y_m == 0]
    anomaly = s_log[y_m == 1]
    clip = np.percentile(s_log, 99.5)
    normal_c = np.clip(normal, None, clip)
    anomaly_c = np.clip(anomaly, None, clip)

    x_range = np.linspace(np.percentile(s_log, 0.5), clip, 200)
    method_label = '$J^*$' if 'energy' in method else '$R$'

    if len(normal_c) > 10 and len(anomaly_c) > 5:
        kde_n = gaussian_kde(normal_c, bw_method=0.3)
        kde_a = gaussian_kde(anomaly_c, bw_method=0.3)
        ax.fill_between(x_range, kde_n(x_range), alpha=0.5, color='steelblue', label='Normal')
        ax.fill_between(x_range, kde_a(x_range), alpha=0.5, color='coral', label='Anomaly')
    else:
        ax.hist(normal_c, bins=50, alpha=0.5, density=True, color='steelblue', label='Normal')
        ax.hist(anomaly_c, bins=50, alpha=0.5, density=True, color='coral', label='Anomaly')

    display = ds.replace('_', '-').capitalize()
    if ds == 'yelpchi': display = 'YelpChi'
    if ds == 't_finance': display = 'T-Finance'
    if ds == 'elliptic_plus_plus': display = 'Elliptic++'
    ax.set_title('%s (%s, AUC=%.1f%%)' % (display, method_label, auc*100), fontsize=10)
    ax.set_xlabel('log(1+score)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=7)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    configs = load_configs()
    if not configs:
        print('ERROR: No config files found. Run test_prior_optimizer_v3.py first.')
        sys.exit(1)

    print('Loaded configs for: %s' % list(configs.keys()))
    os.makedirs('figures', exist_ok=True)

    # 1. Run trials and collect results
    results = {}
    for ds in configs:
        print('\n=== %s ===' % ds, flush=True)
        aucs = run_trials(ds, configs[ds], n_trials=args.trials, device=args.device)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        cfg = configs[ds]['best_config']
        method = cfg.get('score_method', 'precision_energy')
        results[ds] = {
            'mean': mean_auc, 'std': std_auc, 'trials': aucs,
            'method': method, 'config': cfg,
        }
        print('  %s: %.1f +/- %.1f  (%s)' % (ds, mean_auc*100, std_auc*100, method))

    # 2. Save results
    with open('results/final_paper_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nSaved results/final_paper_results.json')

    # 3. Generate density plots
    a100_datasets = [ds for ds in configs if ds not in ['elliptic', 'elliptic_plus_plus', 'dgraph']]
    h100_datasets = [ds for ds in configs if ds in ['elliptic', 'elliptic_plus_plus', 'dgraph']]

    if a100_datasets:
        ncols = min(3, len(a100_datasets))
        nrows = (len(a100_datasets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()
        for idx, ds in enumerate(a100_datasets):
            data = load_data('YelpChi' if ds == 'yelpchi' else ds)
            plot_density(axes_flat[idx], data, configs[ds], ds, device=args.device)
        for idx in range(len(a100_datasets), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle('EB-GAD: Score density separation', fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig('figures/density_final_a100.pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print('Saved figures/density_final_a100.pdf')

    # 4. Print LaTeX table rows
    print('\n=== LaTeX Table Rows ===')
    for ds in sorted(results.keys()):
        r = results[ds]
        print('%-18s & %.1f $\\pm$ %.1f \\\\' % (ds, r['mean']*100, r['std']*100))


if __name__ == '__main__':
    main()
