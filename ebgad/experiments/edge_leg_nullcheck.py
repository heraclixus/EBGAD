"""Monte Carlo check of the edge leg's null: synthetic features from the fitted model, x_f = V diag(v^{1/2}) z_f
(dense spectrum, seed-0 fit), then the edge leg exactly as on real data (population calibration included).
Reports, pooled over draws: the KS distance of e_i to N(0, 1) on random nodes, the tail fractions
P(e < -1.645) and P(e > 1.645) (nominal 0.05), and, for the kappa^2 -> infinity limit (independent isotropic
rows), the KS distance of the per-edge t_{d-1} statistic to its exact law (Proposition 6 (i)).
Usage: edge_leg_nullcheck.py amazon yelpchi [--draws 20] [--nodes 500]
"""
import sys, json, argparse, numpy as np
from scipy.stats import kstest, norm, t as tdist
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.edgeleg import edge_leg, undirected_edges

ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--draws", type=int, default=20)
ap.add_argument("--nodes", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fits", default="results/ebgad_v4_final/seed0"); args = ap.parse_args()
rng = np.random.default_rng(args.seed)
for name in args.datasets:
    data = load_data(name); xt = preprocess_features(data.x, data.edge_index); n, d = xt.shape
    fit = json.load(open(f"{args.fits}/{name}.json"))["fit"]; s2, a, k2 = fit["sigma2"], fit["a"], fit["kappa2"]
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad",
                            cache_tag=f"{name}_pcaNone_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
    v = s2 + 1.0 / (a * (k2 + lam)); sd = np.sqrt(v)
    ei = data.edge_index.numpy(); ia, ib = undirected_edges(ei, n)
    idx = rng.choice(n, size=min(args.nodes, n), replace=False)
    pooled = []; lo = hi = 0; cnt = 0; pooled_t = []; degs_p = []; abs_e_lim = []
    for r in range(args.draws):
        z = rng.standard_normal((n, d)); x = V @ (sd[:, None] * z)
        res = edge_leg(x, ei); e = res["e"]; ee = e[idx]; keep = np.isfinite(ee); ee = ee[keep]
        pooled.append(ee); degs_p.append(res["deg"][idx][keep]); lo += (ee < -1.645).sum(); hi += (ee > 1.645).sum(); cnt += len(ee)
        # kappa^2 -> infinity limit: independent isotropic rows; exact t_{d-1} per edge
        xi = rng.standard_normal((n, d)); xu = xi / np.sqrt((xi * xi).sum(1))[:, None]
        rr = (xu[ia] * xu[ib]).sum(1); pooled_t.append(np.sqrt(d - 1) * rr / np.sqrt(1 - rr ** 2))
    pooled = np.concatenate(pooled); pooled_t = np.concatenate(pooled_t); degs_p = np.concatenate(degs_p)
    from scipy.stats import spearmanr
    q = np.quantile(degs_p, [1 / 3, 2 / 3]); terc = np.digitize(degs_p, q)
    sdev = [pooled[terc == k].std() for k in range(3)]; up = [(pooled[terc == k] > 1.645).mean() for k in range(3)]
    print("    fitted model: Spearman(e, deg) = %.3f, Spearman(|e|, deg) = %.3f; sd of e by degree tercile %.2f / %.2f / %.2f; P(e>1.645) by tercile %.3f / %.3f / %.3f"
          % (spearmanr(pooled, degs_p)[0], spearmanr(np.abs(pooled), degs_p)[0], *sdev, *up), flush=True)
    # limit model (independent isotropic rows): node-level e_i
    xi = rng.standard_normal((n, d)); res_l = edge_leg(xi, ei); el = res_l["e"][idx]; dl = res_l["deg"][idx]; ok = np.isfinite(el) & (dl > 0)
    print("    limit model:  e_i KS %.4f, P(e>1.645)=%.3f, Spearman(|e|, deg) = %.3f" % (kstest(el[ok], norm.cdf).statistic, (el[ok] > 1.645).mean(), spearmanr(np.abs(el[ok]), dl[ok])[0]), flush=True)
    ks = kstest(pooled, norm.cdf).statistic; ks_t = kstest(pooled_t, tdist(d - 1).cdf).statistic
    level = 1.63 / np.sqrt(len(pooled))
    print("%-10s n=%d d=%d draws=%d | e_i vs N(0,1): KS %.4f (1%% level %.4f), P(e<-1.645)=%.3f P(e>1.645)=%.3f (nominal 0.05) | t_{d-1} per edge (limit): KS %.5f on %d edges"
          % (name, n, d, args.draws, ks, level, lo / cnt, hi / cnt, ks_t, len(pooled_t)), flush=True)
