#!/bin/bash
# Marginal-surprise scores (run_ebgad.py --cam --surprise) with probe seeds, the relabeling gate and the
# Monte Carlo null check on the fraud graphs, then one seed on the other datasets.
# Usage: bash ebgad/scripts/camouflage_batch.sh [results-dir]     (set PYTHON to choose the interpreter)
cd "$(dirname "$0")/../.." || exit 1
PY=${PYTHON:-python}
OUT=${1:-results/ebgad_surprise}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
run() { echo "### $(date '+%H:%M:%S') $*"; $PY run_ebgad.py "$@" 2>&1 | grep -v Warning; }
for ds in elliptic elliptic_plus_plus t_finance; do
  run --dataset $ds --template low_unit --cam --surprise --seed 0 --permute-check --null-check 30 --out "$OUT/seed0"
  run --dataset $ds --template low_unit --cam --surprise --seed 1 --out "$OUT/seed1"
  run --dataset $ds --template low_unit --cam --surprise --seed 2 --out "$OUT/seed2"
done
for ds in yelpchi amazon facebook weibo reddit blogcatalog acm; do
  run --dataset $ds --template low_unit --cam --surprise --seed 0 --out "$OUT/scope"
done
