"""Compare zscore vs minmax normalization on Facebook, ACM, BlogCat."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.prior_optimizer import (Q_rho_eigenvalues, marginal_log_likelihood_stationary,
                                  compute_spectral_energy, solve_rho_newton)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

def ks_null(s):
    s2 = s[s > 0]
    if len(s2) < 10: return 0.0
    mu, var = s2.mean(), s2.var()
    if var < 1e-12 or mu < 1e-12: return 0.0
    k = max(0.01, 2*mu**2/var); a = var/(2*mu)
    try: return kstest(s2/a, 'chi2', args=(k,))[0]
    except: return 0.0

def run(data, tmpl, gamma, pca_dim, norm, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
               normalize_mode=norm, prior_mean_mode="stationary",
               laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
               epochs=0, normalize_features=True,
               template_type=tmpl, graph_type="original", gamma=gamma)
    if pca_dim and d > pca_dim:
        cfg.update(dict(use_encoder=True, encoder_type="pca",
                       encoder_hid_dim=pca_dim, encoder_num_layers=1,
                       encoder_dropout=0.0, encoder_lr=0.0, encoder_epochs=0,
                       encoder_alpha=1.0, encoder_weight_decay=0.0))
    trainer = SOCTrainer(SOCTrainerConfig(**cfg))
    trainer.train(data, device=device)
    V = trainer.V.cpu().numpy(); lL = trainer.graph_eigen.lam_L.cpu().numpy()
    tm = trainer.template.cpu(); x = trainer.x_T.cpu()
    n, de = x.shape; delta = x.numpy() - tm.numpy()
    S = compute_spectral_energy(delta, V)
    bml, rho, kap = -np.inf, 0.5, 0
    for k in KAPPAS:
        r = solve_rho_newton(S, lL, k, de); q = Q_rho_eigenvalues(lL, r, k)
        ml = marginal_log_likelihood_stationary(S, q, n, de)
        if ml > bml: bml, rho, kap = ml, r, k
    lQ = torch.from_numpy(Q_rho_eigenvalues(lL, rho, kap)).float()
    Vt = torch.from_numpy(V).float()
    with torch.no_grad():
        je = precision_energy_anomaly(x, tm, Vt, lQ, 2, 1, prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tm, Vt, lQ, 2, 1, prior_mean_mode="stationary").cpu().numpy()
    return je, jr, bml, rho

datasets = {"facebook": "Facebook", "acm": "acm", "blogcatalog": "BlogCatalog"}
baselines = {"facebook": 91.4, "acm": 88.8, "blogcatalog": 82.5}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

for ds, load_name in datasets.items():
    print("\n=== %s ===" % ds, flush=True)
    data = load_data(load_name)
    y = data.y.numpy(); mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None

    for norm in ["zscore", "minmax"]:
        for tmpl in ["low", "affinity"]:
            for gamma in [0.1, 0.2, 0.5, 1.0]:
                try:
                    je, jr, ml, rho = run(data, tmpl, gamma, pca_dim, norm, args.device)
                    if np.any(np.isnan(je)) or np.any(np.isnan(jr)): continue
                    auc_je = roc_auc_score(y[mask], je[mask])
                    auc_jr = roc_auc_score(y[mask], jr[mask])
                    ks_je = ks_null(je[mask]); ks_jr = ks_null(jr[mask])
                    best = max(auc_je, auc_jr)
                    print("  %s %s g=%s: J*=%.1f%% R=%.1f%% rho=%.3f" %
                          (norm, tmpl[:3], gamma, auc_je*100, auc_jr*100, rho), flush=True)
                except Exception as e:
                    print("  %s %s g=%s: ERROR %s" % (norm, tmpl[:3], gamma, str(e)[:40]))

    print("  Baseline: %.1f%%" % baselines[ds])
