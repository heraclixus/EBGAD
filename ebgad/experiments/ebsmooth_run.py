"""Run the matrix-free EB smoothing template on datasets of any size and report both tails.
Usage: PYTHONPATH=. python ebgad/experiments/ebsmooth_run.py <dataset> ... [--variant sym|comb] [--seed k]
       [--permute-check] [--out results/ebgad_v4_ebsmooth]
Env: PROBES (Lanczos probes, 20), LSTEPS (Lanczos steps, 80), HUTCH (Hutchinson probes, 128).
Scores: s (standardized residual energy), CAM / DEV / TWO-SIDED (calibrated lower / upper / two-sided
tail surprise of log s under the EB log-normal scale prior fitted on the evaluated nodes).
--permute-check rescores a node-relabeled copy of the graph with the same seed and reports the Spearman
correlation (exact invariance up to the Monte Carlo probes)."""
import sys, os, json, time
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from data_utils import load_data
from ebgad.prep import preprocess_features, permute_data
from ebgad.ebsmooth import EBSmoother, laplacian_csr, rcm_order
from ebgad.hier import fit_scale_prior
from ebgad.nulls import ratio_tails
from ebgad.surprise import surprise_from_bands
from ebgad.select import z_from_tails
from run_ebgad import eval_mask


def shape_scores(res, d):
    """Upper-tail z of the conditional-roughness ratio under its exact null, and -log sf."""
    sf, cdf = ratio_tails(res["r"], res["r_var_u"], res["r_var_d"], res["r_cov"], d)
    z = z_from_tails(sf, cdf)
    return {"SHAPE": z, "SHAPE-logp": -np.log(np.clip(sf, 1e-300, 1.0))}


def calibrated_tails(s, d, mask):
    """Lower / upper / two-sided surprise of log s under the fitted log-normal scale prior."""
    ref = s[mask] if mask is not None else s
    mu_s, sigma_s2, m_d, v_d = fit_scale_prior(ref, d)
    z = (np.log(np.maximum(s, 1e-300)) - mu_s - m_d) / np.sqrt(sigma_s2 + v_d)
    lo, up = norm.logcdf(z), norm.logsf(z)
    return {"CAM": -lo, "DEV": -up, "TWO-SIDED": -np.log(2.0) - np.minimum(lo, up)}, {"mu_s": mu_s, "sigma_s": float(np.sqrt(sigma_s2))}


def run_once(data, variant, seed, pca=None):
    """Fit and score; nodes are processed in reverse Cuthill-McKee order (cache-friendly products) and
    every per-node array is mapped back to the original order before returning."""
    n0 = data.num_nodes
    if cli.reorder:
        perm = rcm_order(laplacian_csr(data.edge_index, n0, variant=variant))
        data = permute_data(data, perm)          # new index i holds old node perm[i]
    x = preprocess_features(data.x, data.edge_index, pca=pca).double().numpy(); n, d = x.shape
    L = laplacian_csr(data.edge_index, n, variant=variant)
    t1 = time.time()
    from ebgad.ebsmooth import spectral_bound
    sm = EBSmoother(L, x, n_probes=P, m=M, seed=seed, verbose=cli.verbose,
                    lam_max=2.0 if variant == "sym" else spectral_bound(L))
    t2 = time.time(); fit = sm.fit_column_scales(n_starts=cli.starts) if cli.col_scales else sm.fit(n_starts=cli.starts); t3 = time.time()
    x = sm.x                                                  # scores are on the (possibly rescaled) features
    if not fit.converged:
        print(f"  [WARNING] EB fit did not move from its initial point: {fit.message}", flush=True)
    res = sm.residual_and_scale(fit, n_hutch=H, seed=seed + 1)
    if cli.bands > 0:
        E, sig2, names = sm.band_profile(fit, res, n_bands=cli.bands, degree=cli.degree, n_hutch=H, seed=seed + 2)
        res["band_E"], res["band_sig2"], res["band_names"] = E, sig2, names
    if cli.regional:
        res.update(sm.regional_scale(fit, res, n_hutch=H, seed=seed + 3))
    t4 = time.time()
    if cli.reorder:
        def back(v):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                out = np.empty_like(v); out[perm] = v; return out
            return v
        res = {k: back(v) for k, v in res.items()}
        x = back(x)
    return x, L, sm, fit, res, {"lanczos": t2 - t1, "fit": t3 - t2, "score": t4 - t3}

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+")
ap.add_argument("--variant", default="sym", choices=["sym", "comb"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--permute-check", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--pca", type=int, default=None, help="PCA dimension for wide feature matrices (paper: ACM 256)")
ap.add_argument("--verbose", action="store_true", help="log every likelihood evaluation (CG iterations, time)")
ap.add_argument("--no-reorder", dest="reorder", action="store_false", help="disable the RCM node reordering")
ap.add_argument("--starts", type=int, default=3, help="multi-start count for the EB fit")
ap.add_argument("--col-scales", action="store_true", help="EB column scales (columns are replicates up to a scale)")
ap.add_argument("--regional", action="store_true", help="also compute the regional leg g_i (template vs population)")
ap.add_argument("--bands", type=int, default=4, help="relaxation bands for the surprise (shape) channel; 0 disables")
ap.add_argument("--degree", type=int, default=50, help="Chebyshev degree of the band filters")
cli = ap.parse_args()
args, variant, seed, permute_check = cli.datasets, cli.variant, cli.seed, cli.permute_check
out_dir = cli.out or ("results/ebgad_v4_ebsmooth" + ("" if variant == "sym" else "_" + variant))
os.makedirs(out_dir, exist_ok=True)
P, M, H = int(os.environ.get("PROBES", 20)), int(os.environ.get("LSTEPS", 80)), int(os.environ.get("HUTCH", 128))

def au(y, s):
    return 100 * roc_auc_score(y, s), 100 * average_precision_score(y, s)

for name in args:
    t0 = time.time()
    data = load_data(name)
    y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    x, L, sm, fit, res, secs = run_once(data, variant, seed, cli.pca)
    n, d = x.shape
    s = res["s"]; xn = (x * x).sum(1)
    tails, prior = calibrated_tails(s, d, m)
    tails.update(shape_scores(res, d))
    if cli.regional:
        from scipy.stats import chi2 as _chi2
        g = res["g"]; lp_s = np.nan_to_num(-_chi2.logsf(d * s, d), posinf=1e6); lp_g = np.nan_to_num(-_chi2.logsf(d * g, d), posinf=1e6)
        lo_s = np.nan_to_num(-_chi2.logcdf(d * s, d), posinf=1e6); lo_g = np.nan_to_num(-_chi2.logcdf(d * g, d), posinf=1e6)
        tails.update({"REG-DEV": lp_g, "REG-CAM": lo_g, "EITHER-DEV": np.maximum(lp_s, lp_g), "EITHER-CAM": np.maximum(lo_s, lo_g)})
    surprise_diag = None
    if cli.bands > 0:
        try:
            sp_out = surprise_from_bands(res["band_E"], res["band_sig2"], res["band_names"], d, m)
            tails.update({k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in sp_out["scores"].items()})
            surprise_diag = sp_out["diag"]
            surprise_diag["bands_active_frac"] = float(np.isfinite(res["band_E"] / np.where(res["band_sig2"] > 0, res["band_sig2"], np.nan)).mean())
        except Exception as e:      # the surprise channel is a fallback score; never lose the scale channel to it
            print(f"  [surprise channel failed: {e!r}]")
    zc = np.clip(tails["SHAPE"], -40, 40)
    tails["COMBINED"] = tails["TWO-SIDED"] + np.log(2.0) - norm.logsf(zc)     # Fisher-type sum of the two channels' surprises
    dev, cam = au(y, s[m]), au(y, -s[m])
    ng_dev, ng_cam = au(y, xn[m]), au(y, -xn[m])
    lam_ref = 1.0 if variant == "sym" else float(np.median(L.diagonal()))
    g0 = fit.g2 / (fit.g2 + fit.kappa2); g1 = fit.g2 / (fit.g2 + fit.kappa2 + lam_ref)
    print(f"[{name}/{variant}/seed{seed}] n={n} d={d} eval={m.sum()} prev={y.mean():.3f} lam_max={sm.lam_max:.3g}  EB: sigma2={fit.sigma2:.3f} a={fit.a:.3g} kappa2={fit.kappa2:.3g} g2={fit.g2:.3g} "
          f"gain(0)={g0:.3f} gain(typ)={g1:.3f} smooth-frac={1 - fit.sigma2:.3f} S_ii med={np.median(res['S_ii']):.3f} nll evals={fit.n_eval}  prior mu_s={prior['mu_s']:+.2f} sigma_s={prior['sigma_s']:.2f}")
    line = (f"  EB template: DEV {dev[0]:5.1f}/{dev[1]:4.1f}  CAM {cam[0]:5.1f}/{cam[1]:4.1f}  TWO-SIDED {au(y, tails['TWO-SIDED'][m])[0]:5.1f}/{au(y, tails['TWO-SIDED'][m])[1]:4.1f}"
            f"  SHAPE {au(y, tails['SHAPE'][m])[0]:5.1f}/{au(y, tails['SHAPE'][m])[1]:4.1f}  COMBINED {au(y, tails['COMBINED'][m])[0]:5.1f}/{au(y, tails['COMBINED'][m])[1]:4.1f}"
            f"   | no graph: DEV {ng_dev[0]:5.1f}/{ng_dev[1]:4.1f}  CAM {ng_cam[0]:5.1f}/{ng_cam[1]:4.1f}"
            f"   | time: lanczos {secs['lanczos']:.0f}s fit {secs['fit']:.0f}s score {secs['score']:.0f}s")
    print(line)
    if cli.regional:
        print(f"  regional leg: REG-DEV {au(y, tails['REG-DEV'][m])[0]:.1f}/{au(y, tails['REG-DEV'][m])[1]:.1f}  REG-CAM {au(y, tails['REG-CAM'][m])[0]:.1f}/{au(y, tails['REG-CAM'][m])[1]:.1f}"
              f"  EITHER-DEV {au(y, tails['EITHER-DEV'][m])[0]:.1f}  EITHER-CAM {au(y, tails['EITHER-CAM'][m])[0]:.1f}   (var_m median {np.median(res['var_m']):.3g})")
    if surprise_diag is not None:
        print(f"  surprise (bands={surprise_diag['n_bands']} d_eff={surprise_diag['d_eff']:.1f} sigma_s={surprise_diag['sigma_s']:.2f}): "
              + "  ".join(f"{k} {au(y, tails[k][m])[0]:.1f}/{au(y, tails[k][m])[1]:.1f}"
                          for k in ("SURPRISE", "SURPRISE-emp", "SURPRISE-emp2", "S-scale-emp", "S-shape", "S-shape-emp", "S-shape-emp-max")))
    perm_rho = None
    if permute_check:
        rng = np.random.default_rng(12345)
        perm = rng.permutation(n)
        data_p = permute_data(data, perm)
        _, _, _, fit_p, res_p, _ = run_once(data_p, variant, seed, cli.pca)
        s_back = np.empty(n); s_back[perm] = res_p["s"]        # node perm[i] of the original is node i of the copy
        perm_rho = float(spearmanr(s, s_back).correlation)
        print(f"  [permute-check] fit on relabeled graph: sigma2={fit_p.sigma2:.3f} a={fit_p.a:.3g} kappa2={fit_p.kappa2:.3g}; Spearman(s, s_permuted) = {perm_rho:.4f}; "
              f"CAM on permuted {100*roc_auc_score(y, -s_back[m]):.1f}")
    np.savez_compressed(os.path.join(out_dir, f"{name}_scores.npz"), s=s, S_ii=res["S_ii"], tau2=res["tau2"],
                        delta_sq=(res["delta"] ** 2).sum(1), y=y_all, **tails)
    json.dump({"dataset": name, "variant": variant, "seed": seed, "pca": cli.pca, "n": n, "d": d, "fit": fit.as_dict(), "scale_prior": prior,
               "probes": P, "lanczos_steps": M, "hutch": H, "DEV": dev, "CAM": cam,
               "TWO-SIDED": au(y, tails["TWO-SIDED"][m]), "SHAPE": au(y, tails["SHAPE"][m]), "COMBINED": au(y, tails["COMBINED"][m]),
               "surprise": {k: au(y, tails[k][m]) for k in tails if k.startswith("S")} if surprise_diag else None,
               "surprise_diag": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in (surprise_diag or {}).items()},
               "regional": ({k: au(y, tails[k][m]) for k in ("REG-DEV", "REG-CAM", "EITHER-DEV", "EITHER-CAM")} if cli.regional else None),
               "shape_bulk": {"median_z": float(np.median(tails["SHAPE"][m])), "mad_z": float(1.4826 * np.median(np.abs(tails["SHAPE"][m] - np.median(tails["SHAPE"][m]))))},
               "no_graph_DEV": ng_dev, "no_graph_CAM": ng_cam,
               "permute_spearman": perm_rho, "seconds": {"load": secs["lanczos"] and (time.time() - t0), **secs}},
              open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
