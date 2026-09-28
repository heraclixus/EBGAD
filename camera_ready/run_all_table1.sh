#!/bin/bash
# Every run behind Table 1: original node order, one relabeling for every dataset (the relabeling
# gate), four relabelings for the truncated spectra, and the two sensitivity variants.
set -u
cd "$(dirname "$0")/.."
for seed in - 123; do
  camera_ready/run_table1.sh A $seed
  camera_ready/run_table1.sh B $seed
done
for seed in 124 125 126; do
  for ds in yelpchi elliptic elliptic_plus_plus t_finance; do camera_ready/run_table1.sh A $seed $ds; done
done
camera_ready/run_table1.sh Ajac -
camera_ready/run_table1.sh Breddit -
# nullspace dropped on the full-spectrum datasets (sensitivity table): same runs into a second folder
for q in A B; do
  for ds in weibo reddit amazon blogcatalog facebook acm; do
    EBGAD_SUBSPACE=drop OUT_ROOT=results/camera_ready_dropnull camera_ready/run_table1.sh $q - $ds
  done
done
