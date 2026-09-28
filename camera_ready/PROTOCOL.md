# Protocol for the regenerated EB-GAD results

Every choice below is a function of the unlabeled graph.

1. Features: z-scored; PCA to 64 dimensions when D_x > 100 (pipeline default). Candidate-pool families
   (rules 2, 3 and the energy branch of rule 5) use the template, bandwidth and PCA grids attached to
   the rule in `test_h12_transport_filterbank.py`.
2. Spectrum: full when n < 20,000; otherwise the k smallest eigenpairs with k = 500 (n < 100k),
   300 (n < 500k), 200 (n < 1M), 128 (above). Dense decompositions in float64; LOBPCG bases are
   orthonormalized and Ritz-refined in float64.
3. Modeled subspace: all computed modes are kept (see the amendment below).
4. Template bandwidth: three grid values nearest to gamma_0 (from h and d_e), chosen by the
   residual-space profile likelihood, no Jacobian term (Algorithm 1 as submitted).
5. Equilibrium anchor: J* or R of the EB-fitted configuration, chosen by NullKS (ties: J*). The
   per-dataset anchor constants of the submitted code (`PAPER_EBGAD_CONFIGS`) are not used.
6. Family: the ordered rules of Appendix B evaluated on (h, d_e, D_x, k/n) and the unlabeled
   diagnostics of the run. Rule 1 always selects R.
7. Within-family score: anchor as in 5; profile aggregation uses the two-groups component entropy
   (scale-entropy branch), the mixture posterior over the p-value bank (sparse-tail branch) or the
   NullKS-selected path band (trajectory branch); control-energy and control-ratio families use the
   frozen graph-statistic rule (`h14_rule`).
8. Ranks average ties (no dependence on node order).
9. Reporting: full spectra are deterministic and must agree between the original and a relabeled
   node order; truncated spectra are reported as mean and standard deviation over five node orders
   (original and relabel seeds 123 to 126).

## Amendment to item 3 (2026-09-27, after the first regeneration)

The first version of this file, written before any regenerated result was read, said: "the
Laplacian nullspace is dropped when the full spectrum is used; all computed modes are kept when the
spectrum is truncated". That was a misreading of the method. The paper (Appendix A, modeling
details) states that null modes are not removed, and the submitted runs behaved accordingly: their
flag removed modes with eigenvalue below 1e-8, which in single precision removed a null mode only
when rounding made its computed eigenvalue that small. With double precision the two conventions
are well defined, and keeping all modes reproduces the submitted Amazon (78.1), BlogCatalog (78.6)
and Facebook (91.4) entries under the same label-free rules, while dropping the nullspace gives
76.5, 72.0 and 90.8. Item 3 therefore follows the paper's text. Because the amendment was made after
results had been seen, the results under the first version of item 3 are reported next to the main
results (appendix, sensitivity table).
