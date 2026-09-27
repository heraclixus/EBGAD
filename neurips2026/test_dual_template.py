"""Test: add affinity template to pipeline (both templates, KS picks).

At density-adjusted gamma, run {low, affinity} × {J*, R} = 4 candidates.
KS selects the best. Does this improve Facebook, ACM, BlogCat?
"""

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
from soc.prior_optimizer import (
    Q_rho_eigenvalues, marginal_log_likelihood_stationary,
    compute_spectral_energy, solve_rho_newton,
)

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
GAMMA_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
BASELINES_NO_ECOD = {
    "weibo": ("ANOM.", 95.0), "reddit": ("CoLA", 60.3), "amazon": ("TAM", 70.6),
    "yelpchi": ("LOF", 57.2), "blogcatalog": ("TAM", 82.5), "facebook": ("TAM", 91.4),
    "acm": ("TAM", 88.8), "elliptic": ("CoLA", 65.3),
    "elliptic_plus_plus": ("CoLA", 62.9), "t_finance": ("CONAD", 63.1),
}
# Second best for the 3 focus datasets
SECOND_BEST = {"facebook": ("ANOM.", 90.2), "acm": ("DOMINANT", 85.7), "blogcatalog": ("CoLA", 77.5)}


def ks_null(scores):
    s = scores[scores > 0]
    if len(s) < 10: return 0.0
    mu, var = s.mean(), s.var()
    if var < 1e-12 or mu < 1e-12: return 0.0
    k = max(0.01, 2*mu**2/var); a = var/(2*mu)
    try: return kstest(s/a, 'chi2', args=(k,))[0]
    except: return 0.0


def compute_graph_stats(data):
    edge_index = data.edge_index.numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = data.edge_index.shape[1] // 2 / n
    n_sample = min(100000, edge_index.shape[1])
    idx = np.random.RandomState(42).choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1)
    return float(cos.mean()), density


def get_gamma_range(h, density):
    density_factor = np.sqrt(max(density, 10) / 10.0)
    gamma_center = (0.5 + h) / density_factor
    gamma_center = np.clip(gamma_center, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    nearest = [g for g in GAMMA_GRID if g <= max_gamma]
    if not nearest: nearest = [GAMMA_GRID[0]]
    if len(nearest) > 3:
        nearest.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        nearest = sorted(nearest[:3])
    return nearest


def run_template(data, tmpl, gamma, pca_dim, device):
    d = data.x.shape[1]
    cfg = dict(kappa=1, nu=1, rho=0.5, lam_penalty=50, alpha=2, T=1,
               normalize_mode="zscore", prior_mean_mode="stationary",
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
    V = trainer.V.cpu().numpy()
    lam_L = trainer.graph_eigen.lam_L.cpu().numpy()
    tmpl_t = trainer.template.cpu()
    x = trainer.x_T.cpu()
    n, d_eff = x.shape
    delta = x.numpy() - tmpl_t.numpy()
    S = compute_spectral_energy(delta, V)
    best_ml, rho, kappa = -np.inf, 0.5, 0.0
    for k in KAPPAS:
        r = solve_rho_newton(S, lam_L, k, d_eff)
        q = Q_rho_eigenvalues(lam_L, r, k)
        ml = marginal_log_likelihood_stationary(S, q, n, d_eff)
        if ml > best_ml: best_ml, rho, kappa = ml, r, k
    lam_Q = torch.from_numpy(Q_rho_eigenvalues(lam_L, rho, kappa)).float()
    V_t = torch.from_numpy(V).float()
    with torch.no_grad():
        je = precision_energy_anomaly(x, tmpl_t, V_t, lam_Q, 2, 1,
                                      prior_mean_mode="stationary").cpu().numpy()
        jr = precision_ratio_anomaly(x, tmpl_t, V_t, lam_Q, 2, 1,
                                     prior_mean_mode="stationary").cpu().numpy()
    return je, jr, best_ml, rho


def run_dataset(ds, device="cpu"):
    load_name = ("YelpChi" if ds == "yelpchi" else
                 "Facebook" if ds == "facebook" else ds)
    data = load_data(load_name)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    d = data.x.shape[1]
    pca_dim = 64 if d > 100 else None
    h, density = compute_graph_stats(data)
    gamma_range = get_gamma_range(h, density)

    print("  h=%.3f dens=%.1f gamma_range=%s pca=%s" %
          (h, density, gamma_range, pca_dim), flush=True)

    # ML selects gamma (using low template, same as current pipeline)
    best_ml = -np.inf
    best_gamma = gamma_range[0]
    for gamma in gamma_range:
        try:
            _, _, ml, _ = run_template(data, "low", gamma, pca_dim, device)
            if ml > best_ml:
                best_ml = ml
                best_gamma = gamma
        except:
            pass

    print("  ML-selected gamma=%s" % best_gamma, flush=True)

    # At selected gamma, run BOTH templates
    candidates = []
    for tmpl in ["low", "affinity"]:
        try:
            je, jr, ml, rho = run_template(data, tmpl, best_gamma, pca_dim, device)
            if np.any(np.isnan(je)) or np.any(np.isnan(jr)):
                continue
            auc_je = roc_auc_score(y[mask], je[mask])
            auc_jr = roc_auc_score(y[mask], jr[mask])
            ks_je = ks_null(je[mask])
            ks_jr = ks_null(jr[mask])
            for sn, auc, ks in [("J*", auc_je, ks_je), ("R", auc_jr, ks_jr)]:
                candidates.append({
                    "tag": "%s %s" % (sn, tmpl[:3]),
                    "auc": auc, "ks": ks, "tmpl": tmpl, "rho": rho,
                })
            print("    %s: J*=%.1f%%(KS=%.3f) R=%.1f%%(KS=%.3f) rho=%.3f" %
                  (tmpl[:3], auc_je*100, ks_je, auc_jr*100, ks_jr, rho), flush=True)
        except Exception as e:
            print("    %s: ERROR %s" % (tmpl[:3], str(e)[:60]))

    if not candidates:
        return None

    # KS selects from all 4 candidates
    ks_best = max(candidates, key=lambda c: c["ks"])

    # Current pipeline (low template only, KS between J*/R)
    low_only = [c for c in candidates if c["tmpl"] == "low"]
    pipeline = max(low_only, key=lambda c: c["ks"]) if low_only else ks_best

    oracle = max(candidates, key=lambda c: c["auc"])

    return {
        "pipeline_auc": pipeline["auc"], "pipeline_tag": pipeline["tag"],
        "dual_auc": ks_best["auc"], "dual_tag": ks_best["tag"],
        "oracle_auc": oracle["auc"], "oracle_tag": oracle["tag"],
        "gamma": best_gamma,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(BASELINES_NO_ECOD.keys())
    results = {}

    for ds in datasets:
        print("\n=== %s ===" % ds, flush=True)
        try:
            r = run_dataset(ds, args.device)
            if r:
                bname, bauc = BASELINES_NO_ECOD.get(ds, ("?", 0))
                sb = SECOND_BEST.get(ds, None)
                print("  Pipeline (low only): %.1f%% (%s)" % (r["pipeline_auc"]*100, r["pipeline_tag"]))
                print("  Dual template (KS):  %.1f%% (%s)" % (r["dual_auc"]*100, r["dual_tag"]))
                print("  Oracle (4 cands):    %.1f%% (%s)" % (r["oracle_auc"]*100, r["oracle_tag"]))
                print("  vs %s: pipe=%+.1f dual=%+.1f" %
                      (bname, r["pipeline_auc"]*100-bauc, r["dual_auc"]*100-bauc))
                if sb:
                    print("  vs 2nd best %s: pipe=%+.1f dual=%+.1f" %
                          (sb[0], r["pipeline_auc"]*100-sb[1], r["dual_auc"]*100-sb[1]))
                results[ds] = r
        except Exception as e:
            import traceback
            print("  ERROR: %s" % str(e)[:80])
            traceback.print_exc()

    if len(results) > 1:
        print("\n=== SUMMARY ===")
        print("%-15s %7s %7s %7s %7s" % ("Dataset", "Pipe", "Dual", "Oracle", "Base"))
        for ds in datasets:
            if ds in results:
                r = results[ds]
                _, bauc = BASELINES_NO_ECOD.get(ds, ("?", 0))
                print("%-15s %6.1f%% %6.1f%% %6.1f%% %6.1f%%" %
                      (ds, r["pipeline_auc"]*100, r["dual_auc"]*100,
                       r["oracle_auc"]*100, bauc))

    os.makedirs("results", exist_ok=True)
    with open("results/dual_template.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
