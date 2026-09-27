# EB smoothing: a matrix-free signal-plus-noise template

`ebgad/ebsmooth.py` models every feature column as a graph-smooth field observed through white noise,

    x_f = h_f + eps_f,    h_f ~ N(0, (a (kappa^2 I + L))^-1),    eps_f ~ N(0, sigma^2 I),

with `L` the symmetric-normalized Laplacian. The template is the posterior mean of the field, a low-pass filter
whose bandwidth is fitted by the marginal likelihood of the features. The residual scale of node `i`,
`s_i = ||Delta_i||^2 / (d tau_i^2)`, satisfies `d s_i ~ chi^2_d` exactly under the model. Nothing needs an
eigendecomposition: the likelihood, its gradient, the leverage and the score use block conjugate gradients, a
reusable stochastic Lanczos quadrature and an algebraic multigrid preconditioner.

## Run one dataset

    PYTHONPATH=. python ebgad/experiments/ebsmooth_run.py t_finance --seed 0 --permute-check --out results/ebsmooth/seed0

This prints the fit (`sigma^2`, `a`, `kappa^2`, template gains), the score read in both directions and two-sided,
the no-graph limit and the timings, and writes `<out>/<dataset>.json` (fit, metrics, gates) and
`<dataset>_scores.npz` (per-node statistics and labels). Options: `--bands 0` skips the band surprise, which is
slow on the large graphs; `--pca 256` for very wide features; `--starts 1` for a single start of the optimizer;
`--regional` adds the template-against-population statistic; `--verbose` logs every likelihood evaluation.

## All datasets

    bash ebgad/scripts/smoothing_batch.sh            # probe seeds 0-2, relabeling gate on seed 0

## Gates

- `ebgad/experiments/ebsmooth_nullcheck.py <dataset>`: Monte Carlo check of the exact null.
- `ebgad/experiments/ebsmooth_check.py`: matrix-free fit against the dense spectral fit.
- `ebgad/experiments/ebsmooth_profile.py <dataset>`: likelihood profile in `sigma^2`.
- `ebgad/experiments/amg_precond_test.py <dataset>`: plain against preconditioned conjugate gradients.

## Notes

- The torch CSR product of a single vector is unreliable on macOS; scipy is used there.
- The Chebyshev band filters need the true spectral range `[0, 2]`.
- Solves at smoothing strength `c >= 100` use the multigrid preconditioner.
- The fit must use exact solves: an iteration cap breaks L-BFGS-B.
