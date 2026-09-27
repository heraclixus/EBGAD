"""Monte Carlo check of the v4 scale statistic's exact null on a dataset with a dense spectrum.

Draw synthetic features x ~ N(0, v(L)) at the EB fit (V diag(sqrt v) z, exact), rerun the
matrix-free pipeline at the same hyperparameters (its own Hutchinson tau^2), and test that
d * s_i ~ chi^2_d per node: pooled KS of the p-values over n_sub nodes x B draws, and the
KS of each node's B p-values (median over nodes). Usage: ... ebsmooth_nullcheck.py <dataset> [B] [n_sub]"""
import sys, numpy as np
from scipy.stats import chi2, kstest
from data_utils import load_data
from ebgad.prep import preprocess_features, compute_spectrum
from ebgad.ebsmooth import EBSmoother, EBSmoothFit, laplacian_csr
name = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 20; n_sub = int(sys.argv[3]) if len(sys.argv) > 3 else 500
data = load_data(name); xt = preprocess_features(data.x, data.edge_index); n, d = xt.shape
spec = compute_spectrum(xt, data.edge_index, n, graph="original", k=None, seed=0, cache_dir="cache/ebgad",
                        cache_tag=f"{name}_pcaNone_code_f64", force_full=True)
V, lam = spec.V.astype(np.float64), spec.lam.astype(np.float64)
L = laplacian_csr(data.edge_index, n)
sm = EBSmoother(L, xt.double().numpy(), n_probes=20, m=80, seed=0)
fit = sm.fit()
v = fit.sigma2 + 1.0 / (fit.a * (fit.kappa2 + lam))
rng = np.random.default_rng(7)
sub = rng.choice(n, size=min(n_sub, n), replace=False)
P = np.zeros((B, sub.size))
for b in range(B):
    xs = V @ (np.sqrt(v)[:, None] * rng.standard_normal((n, d)))
    sm.x = np.ascontiguousarray(xs); sm._y_cache = None
    res = sm.residual_and_scale(fit, n_hutch=128, seed=100 + b, shape=False)
    P[b] = chi2.sf(d * res["s"][sub], d)
pooled = kstest(P.ravel(), "uniform").statistic
per_node = np.median([kstest(P[:, j], "uniform").statistic for j in range(sub.size)])
print(f"[{name}] fit sigma2={fit.sigma2:.3f} a={fit.a:.3g} kappa2={fit.kappa2:.3g}; {sub.size} nodes x {B} draws: "
      f"pooled KS={pooled:.4f} (iid 1% level {1.63/np.sqrt(B*sub.size):.4f}), median per-node KS={per_node:.3f} (iid 50% level ~{0.83/np.sqrt(B):.3f}); "
      f"mean p={P.mean():.3f} frac p<0.05={np.mean(P<0.05):.3f} frac p>0.95={np.mean(P>0.95):.3f}")
