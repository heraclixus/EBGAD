#!/bin/bash
# Studies of the appendix, under the protocol of Table 1. Run after run_table1.sh and select_table1.py.
# Every command is independent; on a many-core machine they can be started in parallel.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export EBGAD_SUBSPACE=keep
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
O=results/camera_ready/studies
mkdir -p $O/parts $O/parts_anchor $O/bandwidth $O/logs results/camera_ready/figures
H="0.02,0.05,0.1,0.2,0.5,1,2,5,10,50,inf"
K="0,0.001,0.003,0.01,0.03,0.1,0.3,0.5,1,2,3,5,7,10,15,20"
ALL="weibo reddit amazon yelpchi blogcatalog facebook acm elliptic elliptic_plus_plus t_finance"

# isolation of the spectral precision family
for ds in $ALL; do $PY neurips2026/test_e2_isolation_banks.py --datasets $ds --out $O/e2_$ds.json; done
$PY neurips2026/test_e2_dgraph_cached.py --out $O/e2_dgraph.json
# matched single-score comparison at the configurations and EB fits of Table 1
$PY camera_ready/make_table1_configs.py
for ds in $ALL; do
  $PY neurips2026/test_matched_single_score.py --datasets $ds --impose-fit \
      --config-json camera_ready/table1_configs.json --out $O/matched_table1_$ds.json
done
# equilibrium anchors at every bandwidth of the grid (bandwidth table, averaged-anchor variant)
for ds in $ALL dgraph; do $PY camera_ready/anchor_bandwidth.py --dataset $ds; done
# what each label-free bandwidth criterion selects, including gamma_0 itself
for ds in $ALL; do $PY camera_ready/bandwidth_criteria.py --dataset $ds --out $O/bandwidth/$ds.json; done
# contamination, neighboring anomalies, pairwise readout
for ds in weibo reddit; do for f in 0 0.05 0.1 0.2; do
  $PY neurips2026/test_e5_contamination.py --datasets $ds --fracs $f --out $O/parts/e5_${ds}_$f.json
done; done
for seed in 0 1 2 3 4; do
  for pl in scattered clustered; do
    $PY neurips2026/test_e8_neighbor_anomalies.py --datasets reddit --seeds $seed --cluster-size 10 \
        --placements $pl --out $O/parts/e8_${pl}_$seed.json
  done
  $PY neurips2026/test_e13_edge_crossterms.py --seeds $seed --out $O/parts/e13_$seed.json
done
# the same studies at the anchor bandwidth (numbers quoted in the text of the appendix)
for ds in weibo reddit; do for f in 0 0.05 0.1 0.2; do
  EBGAD_STUDY_BANDWIDTH=anchor $PY neurips2026/test_e5_contamination.py --datasets $ds --fracs $f \
      --out $O/parts_anchor/e5_${ds}_$f.json
done; done
for seed in 0 1 2 3 4; do
  for pl in scattered clustered; do
    EBGAD_STUDY_BANDWIDTH=anchor $PY neurips2026/test_e8_neighbor_anomalies.py --datasets reddit --seeds $seed \
        --cluster-size 10 --placements $pl --out $O/parts_anchor/e8_${pl}_$seed.json
  done
  EBGAD_STUDY_BANDWIDTH=anchor $PY neurips2026/test_e13_edge_crossterms.py --seeds $seed --out $O/parts_anchor/e13_$seed.json
done
EBGAD_STUDY_BANDWIDTH=anchor $PY neurips2026/test_e6_misspecification.py --out $O/e6_misspec_anchor.json
EBGAD_STUDY_BANDWIDTH=anchor $PY camera_ready/karate_example.py --out $O/karate_anchor.json
# misspecification, worked example, template variants
$PY neurips2026/test_e6_misspecification.py --out $O/e6_misspec.json
$PY camera_ready/karate_example.py --out $O/karate.json
for t in affinity affinity_high affinity_mix; do
  $PY neurips2026/test_dynamic_hypotheses.py --dataset blogcatalog --template-type $t --horizons $H --kappas $K \
      --modeled-subspace legacy --out $O/e10_blogcatalog_$t.json
done
# ordering artifact of the submitted version, with its tie-breaking and with averaged ties
EBGAD_INDEX_TIES=1 $PY neurips2026/test_permutation_leakage.py --out $O/ordering_index_ties.json
$PY neurips2026/test_permutation_leakage.py --out $O/ordering_average_ties.json
# the variant that PROTOCOL.md rejects: the bandwidth nearest to gamma_0 for every family (Facebook)
OUT_ROOT=results/camera_ready_center HYP_EXTRA="--bandwidth-rule center" camera_ready/run_table1.sh A - facebook
$PY camera_ready/uniform_center.py
# selector studies, sensitivity and figures (need the runs and dumps of Table 1)
$PY camera_ready/selector_studies.py cutoffs
$PY camera_ready/selector_studies.py ingredients
$PY camera_ready/selector_studies.py stacking
$PY camera_ready/sensitivity.py
$PY camera_ready/make_figures.py densities
$PY camera_ready/make_figures.py likelihood
