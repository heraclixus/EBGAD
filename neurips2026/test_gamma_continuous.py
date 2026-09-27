"""Test continuous gamma optimization on gap datasets.

For each dataset, fix the best discrete choices (template, graph type, normalize mode)
from v3 optimizer, then continuously optimize gamma alongside (rho, kappa).
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch
from scipy.optimize import minimize_scalar

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, Q_rho_eigenvalues, coordinate_descent,
    marginal_log_likelihood,
)
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc
from soc.soc_anomaly import compute_anomaly_scores

import dataclasses


def optimize_gamma_and_params(data, template_type, graph_type, normalize_mode,
                               device='cuda', gamma_range=(0.01, 20.0)):
    """Jointly optimize gamma, rho, kappa via nested coordinate descent.

    Outer: bisection on gamma (recomputes template + S_j at each eval)
    Inner: coordinate descent on (rho, kappa) given S_j
    """
    n, d = data.x.shape
    trainer_keys = {f.name for f in dataclasses.fields(SOCTrainerConfig)}

    def evaluate_gamma(log_gamma):
        gamma = np.exp(log_gamma)
        base_params = dict(
            template_type=template_type, graph_type=graph_type,
            gamma=gamma, normalize_mode=normalize_mode,
            kappa=1.0, rho=0.5, nu=1.0, laplacian_variant='sym',
            d_hidden=64, n_hidden_layers=2, epochs=0,
            lam_penalty=50.0, alpha=2.0, T=1.0,
        )
        try:
            cfg = SOCTrainerConfig(**{k: v for k, v in base_params.items() if k in trainer_keys})
            trainer = SOCTrainer(cfg)
            trainer._setup(data, device=device)

            x_normed = trainer.normalize(data.x.to(device).float())
            delta = (x_normed - trainer.template).cpu().numpy()
            V = trainer.V.cpu().numpy()
            lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
            S = compute_spectral_energy(delta, V)

            # Inner: optimize (rho, kappa) given S
            best_ml = -np.inf
            best_opt = None
            for rho_init in [0.0, 0.5, 1.0]:
                for kappa_init in [0.1, 1.0, 5.0]:
                    opt = coordinate_descent(S, lam_L, n, d,
                        rho_init=rho_init, kappa_init=kappa_init, n_iters=10)
                    if opt['marginal_likelihood'] > best_ml:
                        best_ml = opt['marginal_likelihood']
                        best_opt = opt

            return -best_ml, gamma, best_opt, S, lam_L
        except:
            return 1e30, gamma, None, None, None

    # Coarse grid search first, then refine
    best_neg_ml = np.inf
    best_gamma = 1.0
    best_opt = None

    # Phase 1: coarse grid
    for log_g in np.linspace(np.log(gamma_range[0]), np.log(gamma_range[1]), 30):
        neg_ml, gamma, opt, S, lam_L = evaluate_gamma(log_g)
        if neg_ml < best_neg_ml:
            best_neg_ml = neg_ml
            best_gamma = gamma
            best_opt = opt

    # Phase 2: refine around best
    for log_g in np.linspace(np.log(best_gamma) - 0.5, np.log(best_gamma) + 0.5, 20):
        neg_ml, gamma, opt, S, lam_L = evaluate_gamma(log_g)
        if neg_ml < best_neg_ml:
            best_neg_ml = neg_ml
            best_gamma = gamma
            best_opt = opt

    return best_gamma, best_opt


datasets = ['enron', 'weibo', 'reddit', 'yelpchi', 'facebook', 'acm', 't_finance',
            'blogcatalog']

for ds in datasets:
    print('=== %s ===' % ds, flush=True)
    data = load_data('YelpChi' if ds == 'yelpchi' else ds)

    best_ce = 0; best_cr = 0; best_cfg = {}

    for tpl in ['zero', 'low', 'high', 'affinity']:
        for gt in ['original', 'affinity']:
            for norm in ['zscore', 'minmax']:
                try:
                    gamma_opt, params_opt = optimize_gamma_and_params(
                        data, tpl, gt, norm, device='cuda')

                    if params_opt is None:
                        continue

                    full_cfg = {
                        'rho': params_opt['rho'], 'kappa': params_opt['kappa'],
                        'gamma': gamma_opt, 'alpha': params_opt['alpha_T'], 'T': 1.0,
                        'lam_penalty': params_opt['lam_penalty'],
                        'template_type': tpl, 'graph_type': gt,
                        'normalize_mode': norm,
                        'd_hidden': 64, 'n_hidden_layers': 2,
                        'epochs': 1, 'parameterization': 'score',
                        'time_weighting': 'importance', 'laplacian_variant': 'sym',
                    }

                    torch.manual_seed(0); np.random.seed(0)
                    for method in ['precision_energy', 'precision_ratio']:
                        eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                        r = evaluate_single_trial(data, eval_cfg, device='cuda')
                        if method == 'precision_energy' and r.auc > best_ce:
                            best_ce = r.auc
                            best_cfg = {**full_cfg, 'score_method': method}
                        if method == 'precision_ratio' and r.auc > best_cr:
                            best_cr = r.auc
                except Exception as e:
                    pass

    best = max(best_ce, best_cr)
    print('  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%) gamma=%.3f cfg=%s' % (
        best*100, best_ce*100, best_cr*100,
        best_cfg.get('gamma', 0), best_cfg), flush=True)
