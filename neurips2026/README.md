# neurips2026

Scripts behind the experiments of the paper. Run them from the repository root, for example

    python neurips2026/test_dynamic_score_bank.py --dataset weibo

Every script puts the repository root on `sys.path` through `_paths.py`, reads datasets through
`data_utils.load_data` (see `DATA.md`), and writes to `results/`, `cache/` or `figures/` under the current
directory. Scripts with a command line list their options with `--help`. Labels are used only to report AUROC
and AUPRC.

Three modules are imported by many of the others: `test_dynamic_score_bank.py` (score bank, selectors, rank
aggregation), `test_dynamic_hypotheses.py` (path dissipation, latent-time mixture, two-groups, dynamic profile) and
`test_multiscale_gou_poe.py` (horizon aggregation). The `precompute_*.py` scripts cache the truncated
eigendecompositions of the large graphs.

The index below is generated from the scripts' docstrings.

## Transport score bank and label-free selection

- `extract_best_configs.py`: Extract full best SOC configs from sweep results.
- `final_paper_results.py`: Final paper results: 10-trial evaluation + density plots from saved best configs.
- `run_final_trials.py`: Run 10-trial evaluation on the best config for each dataset.
- `test_dynamic.py`: Comprehensive test of the finite-Gamma dynamic framework.
- `test_dynamic_hypotheses.py`: Prototype trajectory-level GOU anomaly scores.
- `test_dynamic_hypotheses_dgraph_cached.py`: DGraph cached-eigenpair H6 dynamic-profile prototype.
- `test_dynamic_score_bank.py`: Prototype dynamic GOU score banks.
- `test_dynamic_v2.py`: Dynamic framework v2
- `test_dynamic_v3.py`: Dynamic framework v3
- `test_dynamic_v4.py`: Dynamic framework v4
- `test_dynamic_v5.py`: Dynamic framework v5
- `test_dynamic_v6.py`: Dynamic framework v6
- `test_dynamic_v7.py`: Dynamic framework v7
- `test_dynamic_v8.py`: Dynamic framework v8
- `test_h11_filterbank_anchor.py`: H11 targeted filter-bank anchors for the remaining H10 gaps.
- `test_h12_transport_filterbank.py`: H12 targeted dynamic transport filter bank.
- `test_multiscale.py`: Multi-scale score aggregation
- `test_multiscale_gou_poe.py`: Prototype multiscale GOU horizon aggregation.
- `test_path_energy.py`: Compare path_energy vs control_energy vs magnitude on best known hyperparams.
- `test_path_energy_v3.py`: Test path_energy_v2 using the exact best SOC configs from sweep results.

## Ablations and robustness studies

- `analyze_matched_single_score.py`: Analyze matched single-score results.
- `test_ablation_gaps.py`: Quick ablation: how much does each discrete choice matter?
- `test_ablation_tables.py`: Compute data for ablation tables
- `test_bank_horizon_collapse.py`: Bank vs infinite-horizon collapse .
- `test_e13_edge_crossterms.py`: E13: do the edge cross-terms Q_ij <delta_i, delta_j> detect anomalous edges directly?
- `test_e14_calibrated_generics.py`: E14: can EB calibration rescue the generic spectral families?
- `test_e14b_seed_stability.py`: E14b: label-free selection stability under eigensolver randomness.
- `test_e16_amazon_contamination.py`: E16: contamination robustness at an interior EB fit (Amazon).
- `test_e2_dgraph_cached.py`: E2 isolation ablation for DGraph using the cached labeled-subgraph eigenpairs.
- `test_e2_isolation_banks.py`: E2 isolation ablation
- `test_e5_contamination.py`: E5: robustness of the EB fit to anomaly contamination.
- `test_e6_misspecification.py`: E6: misspecification stress test.
- `test_e8_neighbor_anomalies.py`: E8: do neighboring (clustered) anomalies evade the contextual score?
- `test_encoder_sweep.py`: Encoder sweep for datasets missing from the encoder comparison table.
- `test_hybrid_score.py`: Hybrid scoring: combine graph-spectral J*/R with per-feature tail analysis.
- `test_label_efficiency.py`: Label efficiency ablation
- `test_label_efficiency_v2.py`: Label efficiency ablation v2
- `test_large_encoder_small.py`: Small trained-encoder ablation for T-Finance and DGraph.
- `test_matched_single_score.py`: Matched single-score horizon comparison.
- `test_norm_compare.py`: Compare zscore vs minmax normalization on Facebook, ACM, BlogCat.
- `test_permutation_leakage.py`: Permutation test for node-order leakage in the bank machinery.
- `test_raw_hf.py`: Compute gamma-invariant high-frequency energy fraction from RAW features.
- `test_validation_split.py`: Test: Validation split for discrete config selection.
- `test_valsplit_full.py`: Validation split with comprehensive grid covering sweep-best configs.

## Empirical-Bayes prior optimization

- `test_eb_design_options.py`: Smoke tests for backward-compatible EB design options.
- `test_eb_option_probe.py`: Label-free probe for new EB design choices on small benchmark datasets.
- `test_eb_v2.py`: Empirical Bayes test v2
- `test_empirical_bayes.py`: Test empirical Bayes hypothesis
- `test_fixed_config_ml.py`: Test: Fixed default config, ML only for rho/kappa.
- `test_gamma_continuous.py`: Test continuous gamma optimization on gap datasets.
- `test_gamma_density.py`: Gamma selection: density-adjusted homophily center.
- `test_gamma_prior.py`: MAP with log-normal prior on gamma, centered by homophily.
- `test_gamma_restricted.py`: Gamma selection: restrict ML search range by edge homophily + density.
- `test_gamma_rule.py`: Test gamma selection rules.
- `test_lbfgsb_optimizer.py`: Test L-BFGS-B joint optimization of (rho, kappa, gamma) vs coordinate descent.
- `test_lbfgsb_sweep.py`: Optimizer comparison
- `test_map_prior.py`: MAP estimation with Beta prior on rho.
- `test_ml_vs_auroc_diagnostic.py`: Diagnostic: ML vs AUROC correlation per config.
- `test_nokappa_sweep.py`: No-kappa optimizer: Q_prior = L^nu, Q_rho = rho*L^nu + (1-rho)*I.
- `test_ppc_gamma.py`: Posterior Predictive Check for gamma selection.
- `test_predictive_check.py`: Compare prior selection strategies
- `test_prior_optimizer.py`: Test the closed-form prior optimizer on all datasets.
- `test_prior_optimizer_v2.py`: Prior optimizer v2: adds gamma sweep, multi-start, normalize_mode.
- `test_prior_optimizer_v3.py`: Prior optimizer v3: SAVES the winning config for each dataset.
- `test_prior_optimizer_v4.py`: Prior optimizer v4: Data-driven kappa bounds from variance matching.
- `test_prior_optimizer_v5_enc.py`: Prior optimizer v5: v4 + encoder as discrete hyperparameter.
- `test_prior_optimizer_v6_enc.py`: Prior optimizer v6: PCA, GAT, GraphSAGE encoders.
- `test_prior_optimizer_v7_stationary.py`: Prior optimizer v7: Stationary prior mean (delta = x - m).
- `test_prior_optimizer_v8_comprehensive.py`: Prior optimizer v8: Comprehensive sweep with stationary prior.
- `test_profile_newton_validate.py`: Validate profile-Newton matches L-BFGS-B on all datasets.
- `test_profile_newton_validate2.py`: Validate profile-Newton vs L-BFGS-B vs CD with CONSISTENT data-driven bounds.
- `test_rho_constraint.py`: Test constrained rho optimization
- `test_tempered_ml.py`: Tempered ML for gamma selection.
- `test_trimmed_ml.py`: Test trimmed ML for gamma selection.
- `test_two_gamma.py`: Minimal pipeline: 2 gammas (0.5 and 1.0), low template, original graph.
- `test_variance_matching_opt.py`: Test variance-matching optimizer

## Selection criteria

- `analyze_criteria.py`: Analyze: ML selects config (template+gamma), trimmed_gini selects score (J* vs R).
- `test_chisq_rule.py`: Test chi-squared goodness-of-fit rule for J* vs R selection.
- `test_corr_rule.py`: Test: ML selects gamma, corr(J*,R) selects score.
- `test_criterion_search.py`: Systematic criterion search on the restricted v2 grid.
- `test_dual_template.py`: Test: add affinity template to pipeline (both templates, KS picks).
- `test_gap_fix.py`: Targeted fix for BlogCat, Facebook, ACM gaps.
- `test_graph_stats_prediction.py`: Test: Can graph statistics predict the best discrete config?
- `test_majority_vote.py`: Majority vote ensemble
- `test_pipeline_full.py`: Full unsupervised pipeline with PCA search, C score, and rho fallback.
- `test_pipeline_ks.py`: Final unsupervised pipeline with KS null-deviation score selection.
- `test_pipeline_ks_v2.py`: Pipeline v2: ML selects gamma from {0.5, 1.0}, KS selects J* vs R.
- `test_pipeline_ks_v3.py`: Pipeline v3: KS selects from all (gamma, score) pairs.
- `test_pipeline_minmax.py`: Run full density-adjusted pipeline with minmax normalization.
- `test_score_predictors.py`: Compute unsupervised graph/feature statistics to predict J* vs R.
- `test_score_relationship.py`: Analyze the relationship between J*, R, and ||delta||^2.
- `test_template_rule.py`: Isolate template selection
- `test_unsupervised_criteria.py`: Test unsupervised criteria for config selection.
- `test_unsupervised_criteria_v2.py`: Test multiple unsupervised selection criteria across the full config grid.
- `test_unsupervised_final.py`: Final unsupervised EB-GAD pipeline.
- `test_unsupervised_pipeline.py`: Test the fully unsupervised pipeline.
- `test_unsupervised_pipeline_v2.py`: Fully unsupervised EB-GAD pipeline.

## Dataset-specific and large-graph runs

- `precompute_dgraph_eigen.py`: Precompute and cache DGraph eigendecomposition.
- `precompute_dgraph_lowrank.py`: Precompute DGraph eigendecomposition.
- `precompute_medium_eigen.py`: Precompute eigendecomposition for medium-sized datasets.
- `sweep_ce_improve.py`: Targeted CE-only sweeps for 5 datasets where control_energy underperforms.
- `sweep_disney_ce_local.py`: Local Disney CE sweep — fast iteration on 124-node graph.
- `test_acm_blog_targeted.py`: Targeted optimizer for ACM and BlogCatalog.
- `test_acm_capped.py`: ACM: test capped J* (= J* with eigenvalue floor) as alternative to C score.
- `test_acm_reproduce.py`: Reproduce ACM reported best
- `test_acm_sweep.py`: ACM targeted sweep: PCA dimensions, truncated eigendecomposition, all scores.
- `test_blogcatalog_bandpass.py`: BlogCatalog bandpass template sweep.
- `test_blogcatalog_focused.py`: Focused BlogCatalog sweep to improve from 78.8%.
- `test_dgraph_cached.py`: DGraph sweep using cached eigendecomposition.
- `test_dgraph_comprehensive.py`: Comprehensive DGraph optimizer.
- `test_dgraph_encoder.py`: DGraph encoder-specific optimizer.
- `test_dgraph_fast.py`: Fast DGraph training-free sweep using lightweight eigendecomposition.
- `test_dgraph_full.py`: DGraph comprehensive test with cached eigenpairs.
- `test_dgraph_highk.py`: DGraph sweep with high-k eigendecomposition.
- `test_dgraph_trainfree.py`: DGraph training-free sweep (PCA + no encoder only).
- `test_dgraph_unsup_final.py`: DGraph unsupervised pipeline using cached eigenpairs.
- `test_dgraph_v2.py`: DGraph scoring with recomputed eigenpairs from cache/dgraph_eigen_v2/.
- `test_dgraph_v4.py`: DGraph scoring with eigsh/lobpcg eigenpairs from cache/dgraph_eigen_v4/.
- `test_ecod_tfinance.py`: Re-run ECOD on T-Finance to confirm 83.5% AUROC.
- `test_enron_dgraph.py`: Quick test of the density-adjusted pipeline on Enron and DGraph.
- `test_facebook_pca.py`: Facebook: sweep PCA × template × gamma to find best unsupervised config.
- `test_tfinance_fast.py`: T-Finance fast optimizer

## Figures

- `generate_all_density.py`: Generate score density separation plots for ALL datasets.
- `generate_all_density_v2.py`: Generate correct density plots using the ACTUAL best config per dataset.
- `generate_auroc_vs_kappa.py`: Plot AUROC vs kappa for multiple datasets and rho values.
- `generate_contour_landscape.py`: Generate 2D contour plots of the ML landscape over (rho, kappa).
- `generate_contour_sweep.py`: Generate ML contour plots for many (dataset, profile) combinations.
- `generate_convergence_plots.py`: Generate convergence and landscape plots for Algorithm 1.
- `generate_convergence_v2.py`: Generate improved convergence plots
- `generate_density_comparison.py`: Generate density comparison
- `generate_density_plots_final.py`: Generate density separation plots for datasets where EB-GAD beats baselines.
- `generate_density_sep_full.py`: Regenerate density separation
- `generate_density_separation.py`: Regenerate density separation plots with EB-GAD name.
- `generate_fig1_components.py`: Generate individual components for Figure 1 (for PowerPoint assembly).
- `generate_fig1_final_parts.py`: Generate Figure 1 components that need LaTeX rendering.
- `generate_fig1_formula_boxes.py`: Generate standalone green formula boxes for Figure 1 assembly.
- `generate_figure1.py`: Regenerate Figure 1 by updating formulas in the original Keynote figure.
- `generate_figures.py`: Generate paper figures
- `generate_figures_v2.py`: Generate improved paper figures.
- `generate_final_contour.py`: Generate publication-quality contour + convergence figures.
- `generate_landscape_comparison.py`: Generate side-by-side ML vs AUC landscape for 2 datasets.
- `generate_optimization_analysis.py`: Generate optimization analysis figures for the paper.
- `generate_optimization_figures.py`: Generate optimization analysis figures for Section 4.3.
- `generate_prior_diagnostics.py`: Generate prior selection diagnostic figures.
- `generate_spectral_analysis.py`: Generate spectral analysis figures for §4.4.
- `generate_theory_validation.py`: Visualizations validating Corollary 1 and Remark 1.
- `generate_timing_and_convergence.py`: Generate runtime comparison and Newton convergence plots.
- `plot_density_all.py`: Generate score density separation plots for all datasets.
- `plot_density_comparison.py`: Generate TAM vs EB-GAD density comparison plots for Weibo and Amazon.
- `plot_density_weibo_tfinance.py`: Generate TAM vs EB-GAD density comparison for Weibo and T-Finance.
