"""Local Disney CE sweep — fast iteration on 124-node graph."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys
sys.stdout.reconfigure(line_buffering=True)

from soc.soc_gad import SOCGADConfig, evaluate_single_trial
from data_utils import load_data
import numpy as np, torch

data = load_data("disney")

configs = [
    # Normalization variants with zero template
    ("CE zero minmax",   dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="zero", graph_type="original", normalize_mode="minmax", lam_penalty=100)),
    ("CE zero robust",   dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="zero", graph_type="original", normalize_mode="robust", lam_penalty=100)),
    # Affinity graph combos
    ("CE high+aff",      dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="high", graph_type="affinity", lam_penalty=100)),
    ("CE aff+aff",       dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="affinity", graph_type="affinity", lam_penalty=100)),
    # Extreme kappa
    ("CE zero k=0.01",   dict(rho=1.0, kappa=0.01, gamma=1.0, template_type="zero", graph_type="original", lam_penalty=100)),
    ("CE zero k=50",     dict(rho=1.0, kappa=50.0, gamma=1.0, template_type="zero", graph_type="original", lam_penalty=100)),
    # Low template
    ("CE low k=5",       dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="low", graph_type="original", lam_penalty=100)),
    ("CE low aff k=5",   dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="low", graph_type="affinity", lam_penalty=100)),
    # High lambda
    ("CE zero lam=5k",   dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="zero", graph_type="original", lam_penalty=5000)),
    ("CE high lam=5k",   dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="high", graph_type="original", lam_penalty=5000)),
    # Partial graph trust
    ("CE zero rho=0.5",  dict(rho=0.5, kappa=5.0, gamma=1.0, template_type="zero", graph_type="original", lam_penalty=100)),
    # Encoder variants
    ("CE enc gcn",       dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="high", graph_type="original", lam_penalty=100,
                              use_encoder=True, encoder_type="gcn", encoder_hid_dim=32, encoder_epochs=300)),
    ("CE enc mlp",       dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="high", graph_type="original", lam_penalty=100,
                              use_encoder=True, encoder_type="mlp", encoder_hid_dim=32, encoder_epochs=300)),
    # Zero template + minmax + various kappa
    ("CE z mm k=0.1",    dict(rho=1.0, kappa=0.1, gamma=1.0, template_type="zero", graph_type="original", normalize_mode="minmax", lam_penalty=100)),
    ("CE z mm k=1",      dict(rho=1.0, kappa=1.0, gamma=1.0, template_type="zero", graph_type="original", normalize_mode="minmax", lam_penalty=100)),
    ("CE z mm k=20",     dict(rho=1.0, kappa=20.0, gamma=1.0, template_type="zero", graph_type="original", normalize_mode="minmax", lam_penalty=100)),
    # High template + minmax
    ("CE high minmax",   dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="high", graph_type="original", normalize_mode="minmax", lam_penalty=100)),
    # Affinity template + minmax
    ("CE aff minmax",    dict(rho=1.0, kappa=5.0, gamma=0.5, template_type="affinity", graph_type="original", normalize_mode="minmax", lam_penalty=100)),
    # Reference: magnitude (needs training)
    ("MAG reference",    dict(rho=1.0, kappa=5.0, gamma=1.0, template_type="high", graph_type="original",
                              lam_penalty=100, epochs=300, parameterization="epsilon", d_hidden=256,
                              n_hidden_layers=3, score_method="magnitude", score_K=30)),
]

for label, kwargs in configs:
    method = kwargs.pop("score_method", "control_energy")
    epochs = kwargs.pop("epochs", 1)
    d_hid = kwargs.pop("d_hidden", 64)
    n_lay = kwargs.pop("n_hidden_layers", 2)
    param = kwargs.pop("parameterization", "score")
    sk = kwargs.pop("score_K", 1)
    aucs = []
    for seed in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = SOCGADConfig(dataset="disney", d_hidden=d_hid, n_hidden_layers=n_lay,
                           score_method=method, score_K=sk, epochs=epochs,
                           parameterization=param, **kwargs)
        r = evaluate_single_trial(data, cfg, device="cpu")
        aucs.append(r.auc)
    print("%-20s  AUC=%.1f +/- %.1f" % (label, np.mean(aucs)*100, np.std(aucs)*100))
