#!/bin/bash
# EB smoothing (ebgad/ebsmooth.py) on every dataset: probe seeds 0-2, relabeling gate on seed 0.
# Usage: bash ebgad/scripts/smoothing_batch.sh [results-dir]      (set PYTHON to choose the interpreter)
cd "$(dirname "$0")/../.." || exit 1
PY=${PYTHON:-python}
OUT=${1:-results/ebsmooth}
export PYTHONPATH=. OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
run() { echo "### $(date '+%H:%M:%S') $*"; $PY ebgad/experiments/ebsmooth_run.py "$@" 2>&1 | grep --line-buffered -vE "Warning|sparse_coo"; }
for seed in 0 1 2; do
  gate=""; [ "$seed" = 0 ] && gate="--permute-check"
  bands=""; [ "$seed" != 0 ] && bands="--bands 0"      # the band surprise is slow on the large graphs
  for ds in t_finance yelpchi amazon weibo reddit facebook; do run $ds --seed $seed $gate --out "$OUT/seed$seed"; done
  for ds in acm blogcatalog; do run $ds --seed $seed --pca 256 $gate --out "$OUT/seed$seed"; done
  for ds in elliptic elliptic_plus_plus; do run $ds --seed $seed $gate $bands --out "$OUT/seed$seed"; done
done
