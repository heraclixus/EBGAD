"""Facebook: sweep PCA × template × gamma to find best unsupervised config."""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import numpy as np, torch, sys, os
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data
from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import precision_energy_anomaly, precision_ratio_anomaly
from soc.prior_optimizer import (Q_rho_eigenvalues, marginal_log_likelihood_stationary,
                                  compute_spectral_energy, solve_rho_newton)

def ks_null(s):
    s2 = s[s > 0]
    if len(s2) < 10: return 0.0
    mu, var = s2.mean(), s2.var()
    if var < 1e-12 or mu < 1e-12: return 0.0
    k = max(0.01, 2*mu**2/var); a = var/(2*mu)
    try: return kstest(s2/a, 'chi2', args=(k,))[0]
    except: return 0.0

data = load_data("Facebook")
y = data.y.numpy(); mask = (y >= 0) & (y <= 1)
KAPPAS = [0, 0.01, 0.1, 0.5, 1, 2, 5, 10]

print("Facebook: n=%d d=%d anom=%.1f%%" % (data.x.shape[0], data.x.shape[1], y[mask].mean()*100))
candidates = []

for tmpl in ["low", "affinity"]:
    for gamma in [0.1, 0.2, 0.5, 1.0]:
        for pca in [16, 32, 64]:
            cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
                       normalize_mode="zscore", prior_mean_mode="stationary",
                       laplacian_variant="sym", d_hidden=64, n_hidden_layers=2,
                       epochs=0, normalize_features=True,
                       template_type=tmpl, graph_type="original", gamma=gamma,
                       use_encoder=True, encoder_type="pca", encoder_hid_dim=pca,
                       encoder_num_layers=1, encoder_dropout=0.0, encoder_lr=0.0,
                       encoder_epochs=0, encoder_alpha=1.0, encoder_weight_decay=0.0)
            try:
                t = SOCTrainer(SOCTrainerConfig(**cfg))
                t.train(data, device="cuda")
                V = t.V.cpu().numpy(); lL = t.graph_eigen.lam_L.cpu().numpy()
                tm = t.template.cpu(); x = t.x_T.cpu()
                d2 = x.numpy() - tm.numpy(); S = compute_spectral_energy(d2, V)
                n, de = x.shape
                bml, rho, kap = -np.inf, 0.5, 0
                for k in KAPPAS:
                    r = solve_rho_newton(S, lL, k, de)
                    q = Q_rho_eigenvalues(lL, r, k)
                    ml = marginal_log_likelihood_stationary(S, q, n, de)
                    if ml > bml: bml, rho, kap = ml, r, k
                lQ = torch.from_numpy(Q_rho_eigenvalues(lL, rho, kap)).float()
                Vt = torch.from_numpy(V).float()
                with torch.no_grad():
                    je = precision_energy_anomaly(x, tm, Vt, lQ, 2, 1, prior_mean_mode="stationary").cpu().numpy()
                    jr = precision_ratio_anomaly(x, tm, Vt, lQ, 2, 1, prior_mean_mode="stationary").cpu().numpy()
                ks_j = ks_null(je[mask]); ks_r = ks_null(jr[mask])
                auc_j = roc_auc_score(y[mask], je[mask])
                auc_r = roc_auc_score(y[mask], jr[mask])
                tag = "%s p%d g=%s" % (tmpl[:3], pca, gamma)
                for sn, auc, ks in [("J*", auc_j, ks_j), ("R", auc_r, ks_r)]:
                    candidates.append({"tag": "%s %s" % (sn, tag), "auc": auc, "ks": ks})
                print("%s: J*=%.1f%%(KS=%.3f) R=%.1f%%(KS=%.3f) rho=%.3f" %
                      (tag, auc_j*100, ks_j, auc_r*100, ks_r, rho), flush=True)
            except Exception as e:
                print("%s p%d g=%s: ERROR %s" % (tmpl[:3], pca, gamma, str(e)[:50]))

# Results
oracle = max(candidates, key=lambda c: c["auc"])
ks_best = max(candidates, key=lambda c: c["ks"])
print("\nOracle: %.1f%% (%s)" % (oracle["auc"]*100, oracle["tag"]))
print("KS-sel: %.1f%% (%s)" % (ks_best["auc"]*100, ks_best["tag"]))
print("Baseline TAM: 91.4%%")
