"""The edge leg of the EB model: is a node more or less similar to its neighbors than normal?

Per edge (i, j) the observed similarity is the uncentered cosine r_ij = <x_i, x_j> / (|x_i| |x_j|) over the d
standardized feature columns.  Under the Gaussian model the d column pairs are i.i.d. bivariate normal with
correlation rho_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj), so Fisher's z, sqrt(d-3) (atanh r_ij - atanh rho_ij),
is N(0, 1) per edge.  Two calibrations are compared:
  model    : rho_ij from the fitted covariance (dense spectrum, n <= --dense-max only), then the population of
             edges re-centers and re-scales z (robust location / MAD), exactly as the log-normal scale prior
             does for s_i;
  corr std : rho_ij replaced by the population's typical neighbor correlation, i.e. z = sqrt(d-3) atanh r_ij
             standardized by the robust location / MAD over all edges (one-parameter EB at the edge level;
             needs no solve, O(|E| d)).
Node statistic e_i = sum_j z_ij / sqrt(deg_i) (z-score of the sum under edge independence); DEV = -e_i (less
similar to its neighbors than normal: injected cliques, heterophilous edges), CAM = +e_i (more similar:
camouflaged fraud clusters).  The local leg s_i is loaded from the final run (results/ebgad_v4_final/seed0)
and the "either" union max(-log p_local, -log p_edge) is reported in both tails.
Usage: edge_leg_test.py acm:256 blogcatalog:256 facebook ... [--dense-max 25000] [--fits DIR]
"""
import sys, json, time, argparse, os, numpy as np
from scipy.stats import norm, chi2
from sklearn.metrics import roc_auc_score
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from run_ebgad import eval_mask


def au(y, z):
    return 100 * roc_auc_score(y, z)


def edges_undirected(edge_index, n):
    ei = np.asarray(edge_index)
    a = np.minimum(ei[0], ei[1]); b = np.maximum(ei[0], ei[1])
    m = a != b
    key = np.unique(a[m].astype(np.int64) * n + b[m])
    return key // n, key % n


def sigma_entries_dense(V, lam, s2, a, k2, ia, ib):
    w = 1.0 / (a * (k2 + lam))
    diag = s2 + (V * V) @ w
    off = np.empty(len(ia))
    ch = max(1, int(2e8 / V.shape[1]))
    for s in range(0, len(ia), ch):
        off[s:s + ch] = ((V[ia[s:s + ch]] * w) * V[ib[s:s + ch]]).sum(1)
    return diag, off


def robust_std(z):
    med = np.median(z); mad = 1.4826 * np.median(np.abs(z - med)) + 1e-12
    return (z - med) / mad, med, mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--dense-max", type=int, default=25000)
    ap.add_argument("--fits", default="results/ebgad_v4_final/seed0")
    ap.add_argument("--out", default="results/ebgad_v4_edge")
    args = ap.parse_args()
    fin = lambda v: np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=0.0)
    print("%-14s | %-13s | %-13s | %-13s | %-13s | %-13s | %-13s | %-13s | %s" % (
        "dataset", "affinity D/C", "corr mean D/C", "corr std D/C", "model std D/C", "local D/C", "either D/C", "no-graph D/C", "edge stats"), flush=True)
    for arg in args.datasets:
        name, pca = (arg.split(":") + [None])[:2]; pca = int(pca) if pca else None
        t0 = time.time()
        data = load_data(name); y_all = data.y.numpy().astype(int); m_ = eval_mask(y_all); y = y_all[m_]
        xt = preprocess_features(data.x, data.edge_index, pca=pca); x = xt.double().numpy(); n, d = x.shape
        ia, ib = edges_undirected(data.edge_index.numpy(), n)
        deg = (np.bincount(ia, minlength=n) + np.bincount(ib, minlength=n)).astype(float); degs = np.maximum(deg, 1.0)
        nrm = np.sqrt((x * x).sum(1)) + 1e-12; xu = x / nrm[:, None]
        r = np.empty(len(ia))
        for s in range(0, len(ia), 2_000_000):
            r[s:s + 2_000_000] = (xu[ia[s:s + 2_000_000]] * xu[ib[s:s + 2_000_000]]).sum(1)
        r = np.clip(r, -0.999, 0.999)
        sq = np.sqrt(max(d - 3, 1))
        z_corr = sq * np.arctanh(r)

        def node_sum(zv):
            acc = np.zeros(n); np.add.at(acc, ia, zv); np.add.at(acc, ib, zv)
            return acc / np.sqrt(degs)

        aff = node_sum(r) / np.sqrt(degs)
        corr_mean = node_sum(z_corr) / np.sqrt(degs)
        z_std, zmed, zmad = robust_std(z_corr)
        corr_std = node_sum(z_std)
        model_std = None; rho_note = ""
        if n <= args.dense_max:
            fit = json.load(open(f"{args.fits}/{name}.json"))["fit"]; s2, a, k2 = fit["sigma2"], fit["a"], fit["kappa2"]
            spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad",
                                    cache_tag=f"{name}_pca{pca}_code_f64", force_full=True)
            V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
            diag, off = sigma_entries_dense(V, lam, s2, a, k2, ia, ib)
            rho = np.clip(off / np.sqrt(np.maximum(diag[ia] * diag[ib], 1e-300)), -0.999, 0.999)
            z_model = sq * (np.arctanh(r) - np.arctanh(rho))
            zm_std, mmed, mmad = robust_std(z_model)
            model_std = node_sum(zm_std)
            rho_note = " | rho med %.3f, model z med %.2f mad %.2f" % (np.median(rho), mmed, mmad)
        sc = f"{args.fits}/{name}_scores.npz"
        s_loc = np.load(sc)["s"] if os.path.exists(sc) else None
        cols = [name.ljust(14)]
        for zv in (aff, corr_mean, corr_std, model_std):
            if zv is None:
                cols.append("   --  /   -- "); continue
            zz = zv[m_]; cols.append("%5.1f / %5.1f" % (au(y, -zz), au(y, zz)))
        if s_loc is not None:
            sl = s_loc[m_]; cols.append("%5.1f / %5.1f" % (au(y, sl), au(y, -sl)))
            lp_loc_dev = fin(-chi2.logsf(d * sl, d)); lp_loc_cam = fin(-chi2.logcdf(d * sl, d))
            ee = corr_std[m_]; lp_e_dev = fin(-norm.logcdf(ee)); lp_e_cam = fin(-norm.logsf(ee))
            cols.append("%5.1f / %5.1f" % (au(y, np.maximum(lp_loc_dev, lp_e_dev)), au(y, np.maximum(lp_loc_cam, lp_e_cam))))
        else:
            cols += ["   --  /   -- ", "   --  /   -- "]
        nz = ((x * x).sum(1) / d)[m_]; cols.append("%5.1f / %5.1f" % (au(y, nz), au(y, -nz)))
        cols.append("r med %.3f, z med %.2f mad %.2f, deg %.1f%s | %.0f s" % (np.median(r), zmed, zmad, deg.mean(), rho_note, time.time() - t0))
        print(" | ".join(cols), flush=True)
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(f"{args.out}/{name}_edge_leg.npz", aff=aff, corr_mean=corr_mean, corr_std=corr_std,
                            model_std=(model_std if model_std is not None else np.zeros(0)), deg=deg, z_med=zmed, z_mad=zmad)


if __name__ == "__main__":
    main()
