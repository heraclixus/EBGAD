"""EB smoothing template: does the marginal likelihood of x pick the anomaly-relevant horizon?

Model: x = f + eps per feature column, f ~ N(0, (a (kappa^2 I + L))^{-1}), eps ~ N(0, sigma^2 I).
The template is the posterior mean S x with Wiener gain
    t_j = 1 / (1 + a sigma^2 (kappa^2 + lam_j)) = g2 / (g2 + kappa^2 + lam_j),  g2 = 1 / (a sigma^2),
i.e. the paper's unit-gain low-pass template, but with (sigma^2, a, kappa^2) fitted by the
marginal likelihood of the observed features (closed-form spectral Gaussian). The residual
delta = x - S x has null covariance sigma^2 (I - S), so the node statistic
    s_i = ||delta_i||^2 / (d sigma^2 (1 - S_ii))
has an exact chi^2_d / d null per node. kappa^2 >> g2 collapses the template to the population
mean (no-graph statistic); kappa^2 -> 0, g2 moderate gives neighborhood smoothing.

Reports AUROC/AUPRC of -s (camouflage) and +s (deviation) at the EB fit and along a horizon
sweep, so the EB choice can be compared with the label-optimal horizon. Usage:
    python ebgad/experiments/eb_smoothing.py yelpchi weibo ...
"""
import sys, time
import numpy as np, torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from run_ebgad import eval_mask


def fit_eb(lam, S, d):
    """Maximize the spectral marginal likelihood over (log sigma2, log a, log kappa2)."""
    lam = np.asarray(lam, float); S = np.asarray(S, float)

    def nll(p):
        s2, a, k2 = np.exp(p)
        v = s2 + 1.0 / (a * (k2 + lam))
        return float(np.sum(d * np.log(v) + S / v))

    tot = S.sum() / (d * lam.size)
    best = None
    for ls2 in np.log(tot * np.array([0.05, 0.3, 0.7, 0.95])):
        for la in np.log(np.array([0.03, 0.3, 3.0, 30.0]) / tot):
            for lk in np.log([1e-4, 1e-2, 0.3, 3.0]):
                r = minimize(nll, [ls2, la, lk], method="L-BFGS-B", bounds=[(-40, 40)] * 3)
                if best is None or r.fun < best.fun:
                    best = r
    s2, a, k2 = np.exp(best.x)
    return dict(sigma2=s2, a=a, kappa2=k2, g2=1.0 / (a * s2), nll=float(best.fun))


def node_stat(x, V, xhat, t, s2):
    m = V @ (t[:, None] * xhat)
    delta = x - m
    S_ii = (V * V) @ t
    tau2 = s2 * np.maximum(1.0 - S_ii, 1e-12)
    return (delta * delta).sum(1) / (x.shape[1] * tau2)


def au(y, s):
    return 100 * roc_auc_score(y, s), 100 * average_precision_score(y, s)


def main(name):
    t0 = time.time()
    data = load_data(name)
    x = preprocess_features(data.x, data.edge_index)
    n, d = x.shape
    spec = compute_spectrum(x, data.edge_index, n, graph="original", k=None, seed=0,
                            cache_dir="cache/ebgad", cache_tag=f"{name}_pcaNone_code_f64", force_full=True)
    V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
    x = x.double().numpy()
    xhat = V.T @ x
    S = (xhat ** 2).sum(1)
    y_all = data.y.numpy().astype(int); m = eval_mask(y_all); y = y_all[m]
    fit = fit_eb(lam, S, d)
    t_eb = fit["g2"] / (fit["g2"] + fit["kappa2"] + lam)
    s_eb = node_stat(x, V, xhat, t_eb, fit["sigma2"])
    cam, dev = au(y, -s_eb[m]), au(y, s_eb[m])
    xn = (x * x).sum(1)
    print(f"[{name}] n={n} d={d} k={lam.size} eval={m.sum()} prev={y.mean():.3f}  "
          f"EB: sigma2={fit['sigma2']:.3f} a={fit['a']:.3g} kappa2={fit['kappa2']:.3g} g2={fit['g2']:.3g}  "
          f"gain(lam=0)={t_eb.max():.3f} gain(median lam)={np.median(t_eb):.3f} smooth-var-frac={1 - fit['sigma2']:.3f}")
    print(f"  EB template:  CAM {cam[0]:5.1f}/{cam[1]:4.1f}   DEV {dev[0]:5.1f}/{dev[1]:4.1f}   |  no graph: CAM {au(y, -xn[m])[0]:5.1f}  DEV {au(y, xn[m])[0]:5.1f}")
    # horizon sweep at the EB kappa2 and at kappa2 = 0 (intrinsic prior: null modes unshrunk)
    for k2_name, k2 in (("EB kappa2", fit["kappa2"]), ("kappa2=0", 0.0)):
        rows = []
        for g2 in np.logspace(-4, 3, 15):
            t = g2 / (g2 + k2 + lam)
            s = node_stat(x, V, xhat, t, fit["sigma2"])
            rows.append((g2, au(y, -s[m])[0], au(y, s[m])[0]))
        best_c = max(rows, key=lambda r: r[1]); best_d = max(rows, key=lambda r: r[2])
        print(f"  sweep ({k2_name}): CAM best {best_c[1]:5.1f} at g2={best_c[0]:.2g}; DEV best {best_d[2]:5.1f} at g2={best_d[0]:.2g}   "
              + "  ".join(f"{g:.0e}:{c:.0f}/{dv:.0f}" for g, c, dv in rows[::2]))
    print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        try:
            main(nm)
        except Exception as e:  # keep going across datasets
            print(f"[{nm}] FAILED: {e!r}")
