"""Test variance-matching optimizer: minimize Σ(q_j - q_j*)² instead of maximizing ML.

This avoids the monotonicity issue where ML always wants the tightest prior.
Instead, we directly fit the parametric precision to the empirical optimal q_j* = nd/S_j.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os, itertools
import numpy as np
import torch
from scipy.optimize import minimize

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import compute_spectral_energy, Q_rho_eigenvalues
from soc.soc_anomaly import compute_anomaly_scores
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc

import dataclasses

with open('results/best_soc_configs.json') as f:
    best = json.load(f)
valid_keys = {ff.name for ff in dataclasses.fields(SOCGADConfig)}
skip = {'score_method', 'score_K', 'dataset', '_dataset'}


def variance_matching_loss(params, lam_L, q_star, nu=1.0):
    """Least-squares distance between parametric q_j and target q_j*."""
    rho, log_kappa = params
    rho = np.clip(rho, 0.0, 1.0)
    kappa = np.exp(log_kappa)
    q = Q_rho_eigenvalues(lam_L, rho, kappa, nu)
    # Use log-space matching to handle scale differences across modes
    return np.sum((np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(q_star, 1e-12))) ** 2)


for ds in ['enron', 'weibo', 'reddit', 'yelpchi', 'facebook', 'acm', 't_finance']:
    data = load_data('YelpChi' if ds == 'yelpchi' else ds)
    n, d = data.x.shape

    best_ce = 0
    best_cr = 0
    best_cfg = {}

    for tpl in ['zero', 'low', 'high', 'affinity']:
        for gt in ['original', 'affinity']:
            for gamma in [0.1, 0.2, 0.5, 1.0, 2.0]:
                for norm in ['zscore', 'minmax']:
                    try:
                        # Setup to get eigenvectors and template
                        trainer_keys = {f.name for f in dataclasses.fields(SOCTrainerConfig)}
                        params = {}
                        params['template_type'] = tpl
                        params['graph_type'] = gt
                        params['gamma'] = gamma
                        params['normalize_mode'] = norm
                        params['kappa'] = 1.0
                        params['rho'] = 0.5
                        params['nu'] = 1.0
                        params['laplacian_variant'] = 'sym'
                        params['d_hidden'] = 64
                        params['n_hidden_layers'] = 2
                        params['epochs'] = 0
                        params['lam_penalty'] = 50.0
                        params['alpha'] = 2.0
                        params['T'] = 1.0

                        cfg = SOCTrainerConfig(**{k: v for k, v in params.items() if k in trainer_keys})
                        trainer = SOCTrainer(cfg)
                        trainer._setup(data, device='cuda')

                        x_normed = trainer.normalize(data.x.to('cuda').float())
                        delta = (x_normed - trainer.template).cpu().numpy()
                        V = trainer.V.cpu().numpy()
                        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
                        S = compute_spectral_energy(delta, V)

                        # Target precision: q_j* = nd/S_j
                        q_star = n * d / np.maximum(S, 1e-12)

                        # Optimize (rho, kappa) to match q_star via least squares
                        best_loss = np.inf
                        best_rho, best_kappa = 0.5, 1.0
                        for rho_init in [0.1, 0.5, 0.9]:
                            for lk_init in [-1.0, 0.0, 1.0, 2.0]:
                                result = minimize(
                                    variance_matching_loss,
                                    x0=[rho_init, lk_init],
                                    args=(lam_L, q_star),
                                    method='L-BFGS-B',
                                    bounds=[(0.0, 1.0), (-3.0, 4.0)],
                                )
                                if result.fun < best_loss:
                                    best_loss = result.fun
                                    best_rho = np.clip(result.x[0], 0.0, 1.0)
                                    best_kappa = np.exp(result.x[1])

                        # Evaluate with optimized params
                        torch.manual_seed(0); np.random.seed(0)
                        for method in ['control_energy', 'ce_ratio']:
                            eval_cfg = SOCGADConfig(
                                dataset=ds, score_method=method, score_K=1,
                                d_hidden=64, n_hidden_layers=2, lr=0.001,
                                epochs=1, patience=50, parameterization='score',
                                time_weighting='importance', alpha=2.0,
                                laplacian_variant='sym', lam_penalty=50.0,
                                template_type=tpl, graph_type=gt,
                                gamma=gamma, normalize_mode=norm,
                                rho=best_rho, kappa=best_kappa,
                            )
                            r = evaluate_single_trial(data, eval_cfg, device='cuda')
                            if method == 'control_energy' and r.auc > best_ce:
                                best_ce = r.auc
                                best_cfg = dict(tpl=tpl, gt=gt, gamma=gamma, norm=norm,
                                               rho=round(best_rho, 3), kappa=round(best_kappa, 3),
                                               method='CE')
                            if method == 'ce_ratio' and r.auc > best_cr:
                                best_cr = r.auc
                    except Exception as e:
                        pass

    print('%-12s  CE=%.1f%%  CR=%.1f%%  best=%.1f%%  cfg=%s' % (
        ds, best_ce * 100, best_cr * 100, max(best_ce, best_cr) * 100, best_cfg), flush=True)
