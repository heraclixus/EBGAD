#!/bin/bash
# Regenerate the EB-GAD cells of Table 1 with the paper's pipeline.
# Usage: run_table1.sh <queue> [relabel_seed|-] [dataset]
#   queue A: equilibrium / profile families (test_dynamic_hypotheses.py)
#   queue B: control-energy families on candidate pools (test_h12_transport_filterbank.py)
# With a relabel seed, every dataset is run on randomly relabeled nodes (the relabeling gate).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Protocol: all computed Laplacian modes are kept in the modeled subspace (the paper's Appendix A).
export EBGAD_SUBSPACE=${EBGAD_SUBSPACE:-keep}
QUEUE=$1
SEED=${2:-}
[ "$SEED" = "-" ] && SEED=""
ONLY=${3:-}
TAG=orig
if [ -n "$SEED" ]; then export EBGAD_RELABEL_SEED=$SEED; TAG=relabel$SEED; fi
OUT=${OUT_ROOT:-results/camera_ready}
mkdir -p $OUT/logs $OUT/dumps
H="0.02,0.05,0.1,0.2,0.5,1,2,5,10,50,inf"
K="0,0.001,0.003,0.01,0.03,0.1,0.3,0.5,1,2,3,5,7,10,15,20"
TAUS="0.25,0.5,1,2,3,4,8,12"
LAMS="0.3,0.5,1,2,5,10,50,100,500"

hyp () {  # dataset subspace [truncated-k]
  local ds=$1 sub=$2 tk=${3:-}
  [ -n "$ONLY" ] && [ "$ONLY" != "$ds" ] && return
  local stem=hyp_${ds}_${TAG}
  [ -f $OUT/$stem.json ] && { echo "skip $stem"; return; }
  local extra="${HYP_EXTRA:-}"
  [ -n "$tk" ] && extra="$extra --truncated-k $tk"
  echo "[$(date +%H:%M:%S)] start $stem"
  $PY neurips2026/test_dynamic_hypotheses.py --dataset $ds --horizons $H --kappas $K \
      --modeled-subspace $sub $extra --score-dump-dir $OUT/dumps/$stem \
      --out $OUT/$stem.json > $OUT/logs/$stem.log 2>&1
  echo "[$(date +%H:%M:%S)] done  $stem (exit $?)"
}

h12 () {  # dataset subspace profile
  local ds=$1 sub=$2 prof=$3
  [ -n "$ONLY" ] && [ "$ONLY" != "$ds" ] && return
  local stem=h12_${ds}_${prof}_${TAG}
  [ -f $OUT/$stem.json ] && { echo "skip $stem"; return; }
  echo "[$(date +%H:%M:%S)] start $stem"
  $PY neurips2026/test_h12_transport_filterbank.py --dataset $ds --modeled-subspace $sub \
      --config-profile $prof --transport-taus $TAUS --lam-penalties $LAMS \
      --score-dump-dir $OUT/dumps --out $OUT/$stem.json > $OUT/logs/$stem.log 2>&1
  echo "[$(date +%H:%M:%S)] done  $stem (exit $?)"
}

if [ "$QUEUE" = "A" ]; then
  hyp facebook legacy
  hyp weibo legacy
  hyp reddit legacy
  hyp yelpchi legacy 500
  hyp elliptic legacy 300
  hyp elliptic_plus_plus legacy 300
  hyp t_finance legacy 500
  hyp amazon legacy
  hyp blogcatalog legacy
  hyp acm legacy
elif [ "$QUEUE" = "B" ]; then
  h12 amazon legacy fine
  h12 blogcatalog legacy fine
  h12 acm legacy h14
elif [ "$QUEUE" = "Breddit" ]; then
  # pool that a perturbed cutoff routes Reddit to (cutoff sensitivity study only)
  h12 reddit legacy fine
elif [ "$QUEUE" = "Ajac" ]; then
  # sensitivity: template bandwidth chosen with the Jacobian term of the residual map
  TAG=${TAG}_jac
  HYP_EXTRA="--gamma-jacobian"
  hyp facebook legacy
  hyp weibo legacy
  hyp reddit legacy
  hyp yelpchi legacy 500
  hyp elliptic legacy 300
  hyp elliptic_plus_plus legacy 300
  hyp t_finance legacy 500
  hyp amazon legacy
  hyp blogcatalog legacy
  hyp acm legacy
fi
echo "[$(date +%H:%M:%S)] queue $QUEUE $TAG finished"
