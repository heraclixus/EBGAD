"""Degree-conditional calibration of the edge leg's node statistic.

Under a fitted smooth component the edges of a node are correlated through the field and the single
population constant leaves a degree-dependent bias in e_i (synthetic Amazon: P(e > 1.645) = 0.28 / 0.16 / 0.08
by degree tercile).  Second hierarchical step, on the population of nodes: robust location and scale of e_i
as a function of log degree (quantile bins, median / MAD, linear interpolation in log degree), then
e'_i = (e_i - m(deg_i)) / s(deg_i).  Reports AUROC of e and e' in both tails, Spearman(e, deg) on real data,
and, with --null, the synthetic tail rates by degree tercile before and after the correction.
Usage: edge_degree_probe.py acm:256 facebook t_finance ... [--null amazon] [--bins 12]
"""
import sys, json, argparse, numpy as np
from scipy.stats import spearmanr, kstest, norm
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.edgeleg import edge_leg
from run_ebgad import eval_mask


def degree_calibrate(e, deg, bins=12):
    ok = deg > 0; ld = np.log(np.maximum(deg, 1.0))
    edges_q = np.quantile(ld[ok], np.linspace(0, 1, bins + 1)); edges_q[-1] += 1e-9
    centers, meds, mads = [], [], []
    for k in range(bins):
        sel = ok & (ld >= edges_q[k]) & (ld < edges_q[k + 1])
        if sel.sum() < 20:
            continue
        v = e[sel]; m = np.median(v); s = 1.4826 * np.median(np.abs(v - m)) + 1e-9
        centers.append(np.median(ld[sel])); meds.append(m); mads.append(s)
    if len(centers) < 2:
        return e.copy()
    m_i = np.interp(ld, centers, meds); s_i = np.interp(ld, centers, mads)
    out = (e - m_i) / s_i; out[~ok] = 0.0
    return out


A = lambda y, z: 100 * roc_auc_score(y, z)
ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="*"); ap.add_argument("--null", default=None)
ap.add_argument("--bins", type=int, default=12); ap.add_argument("--draws", type=int, default=10); args = ap.parse_args()
if args.datasets:
    print("%-14s | %6s | %-13s | %-13s | %s" % ("dataset", "deg", "e D/C", "e' (deg) D/C", "Spearman(e,deg) real | Spearman(e',deg)"))
for arg in args.datasets:
    name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
    data = load_data(name); y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    x = preprocess_features(data.x, data.edge_index, pca=pca).double().numpy()
    res = edge_leg(x, data.edge_index.numpy()); e, deg = res["e"], res["deg"]
    ep = degree_calibrate(e, deg, args.bins)
    print("%-14s | %6.1f | %5.1f / %5.1f | %5.1f / %5.1f | %.3f | %.3f" % (name, deg.mean(), A(y, -e[m]), A(y, e[m]), A(y, -ep[m]), A(y, ep[m]),
          spearmanr(e[deg > 0], deg[deg > 0])[0], spearmanr(ep[deg > 0], deg[deg > 0])[0]), flush=True)
if args.null:
    name = args.null; rng = np.random.default_rng(0)
    data = load_data(name); xt = preprocess_features(data.x, data.edge_index); n, d = xt.shape
    fit = json.load(open(f"results/ebgad_v4_final/seed0/{name}.json"))["fit"]; s2, a, k2 = fit["sigma2"], fit["a"], fit["kappa2"]
    spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad", cache_tag=f"{name}_pcaNone_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64); sd = np.sqrt(s2 + 1.0 / (a * (k2 + lam)))
    ei = data.edge_index.numpy(); idx = rng.choice(n, size=min(500, n), replace=False)
    P0, P1, D = [], [], []
    for r in range(args.draws):
        x = V @ (sd[:, None] * rng.standard_normal((n, d))); res = edge_leg(x, ei); e, deg = res["e"], res["deg"]
        ep = degree_calibrate(e, deg, args.bins); ok = deg[idx] > 0
        P0.append(e[idx][ok]); P1.append(ep[idx][ok]); D.append(deg[idx][ok])
    P0, P1, D = map(np.concatenate, (P0, P1, D)); q = np.quantile(D, [1 / 3, 2 / 3]); terc = np.digitize(D, q)
    for lab, P in (("e ", P0), ("e'", P1)):
        up = [(P[terc == k] > 1.645).mean() for k in range(3)]; lo = [(P[terc == k] < -1.645).mean() for k in range(3)]
        print("null %s (%s): KS %.4f | P(>1.645) by tercile %.3f / %.3f / %.3f | P(<-1.645) %.3f / %.3f / %.3f | Spearman(|.|,deg) %.3f"
              % (name, lab, kstest(P, norm.cdf).statistic, *up, *lo, spearmanr(np.abs(P), D)[0]), flush=True)
