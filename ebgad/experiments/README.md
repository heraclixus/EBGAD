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
