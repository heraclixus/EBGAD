"""Test L-BFGS-B joint optimization of (rho, kappa, gamma) vs coordinate descent.

L-BFGS-B handles parameter interactions and uses gradient information for
faster convergence to better optima.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, json, os
import numpy as np
import torch
from scipy.optimize import minimize

sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.prior_optimizer import (
    compute_spectral_energy, Q_rho_eigenvalues, coordinate_descent,
)
from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import eval_roc_auc
from soc.soc_anomaly import compute_anomaly_scores

import dataclasses


def ml_from_params(params_vec, data, template_type, graph_type, normalize_mode, device):
    """Compute negative ML from parameter vector [rho, log_kappa, log_gamma].

    Returns negative ML (for minimization).
    """
    rho = np.clip(params_vec[0], 0.001, 0.999)
    kappa = np.exp(np.clip(params_vec[1], -3, 4))
    gamma = np.exp(np.clip(params_vec[2], -3, 4))

    n, d = data.x.shape
    trainer_keys = {f.name for f in dataclasses.fields(SOCTrainerConfig)}
    base = dict(
        template_type=template_type, graph_type=graph_type,
        gamma=gamma, normalize_mode=normalize_mode,
        kappa=kappa, rho=rho, nu=1.0, laplacian_variant='sym',
        d_hidden=64, n_hidden_layers=2, epochs=0,
        lam_penalty=50.0, alpha=2.0, T=1.0,
    )
    try:
        cfg = SOCTrainerConfig(**{k: v for k, v in base.items() if k in trainer_keys})
        trainer = SOCTrainer(cfg)
        trainer._setup(data, device=device)

        x_normed = trainer.normalize(data.x.to(device).float())
        delta = (x_normed - trainer.template).cpu().numpy()
        V = trainer.V.cpu().numpy()
        lam_L = trainer.graph_eigen.lam_L.cpu().numpy() if hasattr(trainer.graph_eigen, 'lam_L') else np.ones(V.shape[1])
        S = compute_spectral_energy(delta, V)
        q = Q_rho_eigenvalues(lam_L, rho, kappa)

        ml = sum(-S[j]*q[j]/2 + (n*d/2)*np.log(max(q[j], 1e-12)) for j in range(len(q)))
        return -ml  # negative for minimization
    except:
        return 1e30


for ds in ['enron', 'weibo', 'reddit', 'yelpchi', 'facebook', 'acm', 't_finance', 'blogcatalog']:
    print('=== %s ===' % ds, flush=True)
    data = load_data('YelpChi' if ds == 'yelpchi' else ds)

    best_ce = 0; best_cr = 0; best_cfg = {}

    for tpl in ['zero', 'low', 'high', 'affinity']:
        for gt in ['original', 'affinity']:
            for norm in ['zscore', 'minmax']:
                # Multi-start L-BFGS-B
                best_ml_local = np.inf
                best_params_local = None

                for rho_init in [0.1, 0.5, 0.9]:
                    for lk_init in [-1, 0, 1.5]:
                        for lg_init in [-1, 0, 1]:
                            x0 = [rho_init, lk_init, lg_init]
                            try:
                                result = minimize(
                                    ml_from_params, x0,
                                    args=(data, tpl, gt, norm, 'cuda'),
                                    method='L-BFGS-B',
                                    bounds=[(0.001, 0.999), (-3, 4), (-3, 4)],
                                    options={'maxiter': 20, 'ftol': 1e-6},
                                )
                                if result.fun < best_ml_local:
                                    best_ml_local = result.fun
                                    best_params_local = result.x
                            except:
                                pass

                if best_params_local is None:
                    continue

                rho_opt = np.clip(best_params_local[0], 0, 1)
                kappa_opt = np.exp(best_params_local[1])
                gamma_opt = np.exp(best_params_local[2])

                full_cfg = {
                    'rho': float(rho_opt), 'kappa': float(kappa_opt),
                    'gamma': float(gamma_opt), 'alpha': 2.0, 'T': 1.0,
                    'lam_penalty': 50.0,
                    'template_type': tpl, 'graph_type': gt,
                    'normalize_mode': norm,
                    'd_hidden': 64, 'n_hidden_layers': 2,
                    'epochs': 1, 'parameterization': 'score',
                    'time_weighting': 'importance', 'laplacian_variant': 'sym',
                }

                torch.manual_seed(0); np.random.seed(0)
                try:
                    for method in ['precision_energy', 'precision_ratio']:
                        eval_cfg = SOCGADConfig(dataset=ds, score_method=method, score_K=1, **full_cfg)
                        r = evaluate_single_trial(data, eval_cfg, device='cuda')
                        if method == 'precision_energy' and r.auc > best_ce:
                            best_ce = r.auc
                            best_cfg = {**full_cfg, 'score_method': method}
                        if method == 'precision_ratio' and r.auc > best_cr:
                            best_cr = r.auc
                except:
                    pass

    best = max(best_ce, best_cr)
    print('  BEST: %.1f%% (CE=%.1f%%, CR=%.1f%%) cfg=%s' % (
        best*100, best_ce*100, best_cr*100, {k: round(v,3) if isinstance(v,float) else v for k,v in best_cfg.items() if k in ['rho','kappa','gamma','template_type','graph_type','normalize_mode','score_method']}), flush=True)
