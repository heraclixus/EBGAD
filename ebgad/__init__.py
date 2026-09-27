"""EB-GAD: training-free graph anomaly detection by empirical Bayes on a graph Ornstein-Uhlenbeck prior.

Modules
-------
prep       : features -> spectrum -> template -> residual
graphstats : feature homophily and edge density (default template-bandwidth grid)
fit        : stationary empirical-Bayes fit of (rho, kappa) per template bandwidth
scores     : transport bank (J*, R, C, CR) and the conditional GMRF residual
nulls      : exact per-node null distributions under the fitted prior
select     : empirical null and non-null mass, label-free selection
twogroups  : two-groups aggregation over a channel of statistics
hier       : hierarchical node-scale prior
surprise   : marginal surprise of the relaxation profile
ebsmooth   : matrix-free EB smoothing template (no eigendecomposition), exact null

Entry points: run_ebgad.py at the repository root; ebgad/experiments/ebsmooth_run.py.
"""
