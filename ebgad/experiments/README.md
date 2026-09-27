# ebgad/experiments

Diagnostics and gates for the `ebgad` package. Run from the repository root with the root on the path, for example `PYTHONPATH=. python ebgad/experiments/null_check_debug.py`. Scripts read the score dumps written by `run_ebgad.py` or recompute from cached spectra.

## Pipeline diagnostics

- `cam_trivial_baselines.py`: Is CAM a degree or feature-norm artifact?
- `evidence_mixture.py`: Feasibility: two-groups EB with a learned Gaussian alternative in the calibrated evidence space.
- `evidence_plane.py`: Selection-free 2-D score
- `facebook_tg_diagnostic.py`: Why did the old pipeline reach 91.4 on Facebook?
- `fdr_calibration.py`: Calibration of the two-groups posterior
- `fraud_table.py`: Final fraud-family table from results/ebgad_v2_fraud/seed{0,1,2}.
- `label_efficient_selection.py`: Label-efficient selection
- `learned_alternative.py`: Two-groups EB with a LEARNED alternative on the calibrated magnitude statistic.
- `null_check_debug.py`: Per-node against pooled validation of the closed-form nulls on a truncated spectrum.
- `robust_eb_imputation.py`: Robust EB via conditional imputation of flagged rows (scratch experiment).
- `scale_mixture.py`: Can a two-component mixture with a FREE majority component find the anomalous scale group label-free?
- `score_shape_quantiles.py`: Quantiles of the log statistics and percentile ranks of the anomalies for selected candidates.
- `split_half_channel.py`: Split-half reproducibility of the energy and ratio channels (channel diagnostic).
- `suite_summary.py`: Consolidated suite table from results/ebgad_v2_scan (unit template, EB gamma).
- `surprise_diagnostic.py`: Band-energy diagnostics for the v3 surprise score (why does d_eff collapse?).
- `weibo_band_diagnostic.py`: AUROC of feature energy, smoothed-feature energy and Laplacian-band energies on Weibo, with a per-mode calibration of the fit.

## EB smoothing

- `amg_precond_test.py`: Does an algebraic-multigrid preconditioner keep CG fast for strongly smoothed priors?
- `eb_smoothing.py`: EB smoothing template
- `ebsmooth_check.py`: Validate the matrix-free EB smoother against the dense spectral fit on full-spectrum datasets.
- `ebsmooth_nullcheck.py`: Monte Carlo check of the v4 scale statistic's exact null on a dataset with a dense spectrum.
- `ebsmooth_profile.py`: Marginal-likelihood profile in the noise variance sigma^2 (a, kappa^2 re-optimized at each value), with the two tails' AUROC along the profile.
- `ebsmooth_run.py`: Run the matrix-free EB smoothing template on datasets of any size and report both tails.
- `hop_template_camouflage.py`: Camouflage statistic with a polynomial (hop-form) low-pass template
- `no_graph_limit.py`: The model's no-graph limit (kappa^2 -> infinity, Proposition 3) on its own
- `preprocessing_variants.py`: Feature preprocessing under the iid-column model
- `robust_template_test.py`: Robust EB template: x_i = h_i + eps_i, eps_i ~ N(0, sigma^2 zeta_i I_d), zeta_i ~ InvGamma(nu/2, nu/2) (row-wise Student-t noise).
- `two_scale_test.py`: Two legs of the relaxation under the EB model
- `whitened_energy_test.py`: Node share of the evidence

## Edge leg and connections source

- `edge_count_probe.py`: Aggregation probe for the edge leg
- `edge_degree_probe.py`: Degree-conditional calibration of the edge leg's node statistic.
- `edge_leg_aligned_probe.py`: Probe: the edge leg in the graph-aligned feature subspace.
- `edge_leg_nullcheck.py`: Monte Carlo check of the edge leg's null
- `edge_leg_probe.py`: Feasibility probe for an *edge leg*
- `edge_leg_run.py`: Run the edge leg (ebgad/edgeleg.py) on datasets and write results/ebgad_v4_edge/<ds>.json + <ds>_edge.npz.
- `edge_leg_structural_probe.py`: Edge leg on the connections source
- `edge_leg_test.py`: The edge leg of the EB model
- `feature_structure_probe.py`: What do the reconstruction baselines exploit?
- `joint_source_test.py`: One model, two data sources
- `leg_combination.py`: Combine the local leg (residual magnitude, exact chi^2_d null) with the edge leg (direction relative to the neighbors, population-calibrated Fisher z) into one label-free statistic in the declared tail.
- `structural_leg_test.py`: Structural leg: the same EB smoothing model applied to structural features X_s = A_n R, a Gaussian random sketch (d_s columns) of the node's normalized-adjacency row (row geometry preserved, columns ~ i.i.d.
- `structure_probe.py`: Do the deviation benchmarks carry structural anomalies?
- `structure_union_probe.py`: Structural channel and its union with the feature channel.
