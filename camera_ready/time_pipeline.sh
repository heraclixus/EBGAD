#!/bin/bash
# Wall-clock time of the regenerated pipeline, one dataset at a time (run on an idle machine).
# Each entry is the full run that produces the Table 1 score: data loading, eigendecomposition,
# template, EB fit, the score bank of the run and the label-free selection.
cd "$(dirname "$0")/.."
PY=${PY:-python}
export EBGAD_SUBSPACE=keep
export OMP_NUM_THREADS=${THREADS:-16} MKL_NUM_THREADS=${THREADS:-16} OPENBLAS_NUM_THREADS=${THREADS:-16}
O=results/camera_ready/timing
mkdir -p $O
H="0.02,0.05,0.1,0.2,0.5,1,2,5,10,50,inf"
K="0,0.001,0.003,0.01,0.03,0.1,0.3,0.5,1,2,3,5,7,10,15,20"
TAUS="0.25,0.5,1,2,3,4,8,12"
LAMS="0.3,0.5,1,2,5,10,50,100,500"
OUT=$O/timing.jsonl
: > $OUT
t () {  # name kind command...
  local name=$1 kind=$2; shift 2
  local t0=$(date +%s%N)
  "$@" > $O/$name.log 2>&1
  local rc=$?
  local t1=$(date +%s%N)
  local sec=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f", (b-a)/1e9}')
  echo "{\"dataset\": \"$name\", \"kind\": \"$kind\", \"seconds\": $sec, \"exit\": $rc, \"threads\": $OMP_NUM_THREADS, \"load\": \"$(cut -d' ' -f1 /proc/loadavg)\"}" >> $OUT
  tail -1 $OUT
}
hyp () { local ds=$1 sub=$2; t $ds bank $PY neurips2026/test_dynamic_hypotheses.py --dataset $ds --horizons $H --kappas $K --modeled-subspace $sub --out $O/run_$ds.json; }
pool () { local ds=$1 prof=$2; t $ds pool $PY neurips2026/test_h12_transport_filterbank.py --dataset $ds --modeled-subspace legacy --config-profile $prof --transport-taus $TAUS --lam-penalties $LAMS --out $O/run_$ds.json; }
hyp facebook legacy
hyp weibo legacy
hyp reddit legacy
hyp yelpchi legacy
hyp elliptic legacy
hyp elliptic_plus_plus legacy
hyp t_finance legacy
pool amazon fine
pool blogcatalog fine
pool acm h14
rm -rf cache/dgraph_eigen_timing
t dgraph_eigen eigen $PY neurips2026/precompute_dgraph_eigen.py --k-values 128 --graph-types original --subset labeled --output-dir cache/dgraph_eigen_timing
t dgraph bank $PY neurips2026/test_dynamic_hypotheses_dgraph_cached.py --cache-dir cache/dgraph_eigen_timing --template-gammas 0.5,1,2 --horizons $H --kappas $K --out $O/run_dgraph.json
rm -rf cache/dgraph_eigen_timing
echo finished
