# Reproducing the EB-GAD numbers of the NeurIPS 2026 paper

`PROTOCOL.md` states the procedure. No label enters any step before evaluation.

```bash
# from the repository root, with the datasets in place (see ../DATA.md)
camera_ready/run_all_table1.sh            # runs behind Table 1 (original and relabeled node orders)
python camera_ready/select_table1.py      # label-free selection -> results/camera_ready/table1_ebgad.json
camera_ready/run_studies.sh               # studies of the appendix
camera_ready/time_pipeline.sh             # wall-clock times (run on an idle machine)
python camera_ready/make_tables.py        # LaTeX tables -> results/camera_ready/tables/
python camera_ready/make_study_tables.py
```

DGraph uses the eigenpairs of its labeled subgraph:

```bash
python neurips2026/precompute_dgraph_eigen.py --k-values 128 --graph-types original --subset labeled
python neurips2026/test_dynamic_hypotheses_dgraph_cached.py --template-gammas 0.5,1,2 \
    --horizons 0.02,0.05,0.1,0.2,0.5,1,2,5,10,50,inf \
    --kappas 0,0.001,0.003,0.01,0.03,0.1,0.3,0.5,1,2,3,5,7,10,15,20 \
    --score-dump-dir results/camera_ready/dumps/hyp_dgraph_orig --out results/camera_ready/hyp_dgraph_orig.json
```

For a relabeled node order set `EBGAD_RELABEL_SEED` (and a separate `--output-dir` / `--cache-dir` for DGraph).

| File | Role |
|---|---|
| `run_table1.sh` | one dataset, one node order: equilibrium and profile families (queue A), candidate pools (queue B) |
| `select_table1.py` | ordered rules of Appendix B and within-family score; relabeling gate |
| `selector_studies.py` | cutoff sensitivity, ingredients of the within-family rule, supervised bound |
| `anchor_bandwidth.py` | equilibrium anchor at every bandwidth of the grid; `select_table1.py` reads it for runs that do not carry the anchor |
| `bandwidth_criteria.py` | what each label-free bandwidth criterion selects, and the equilibrium scores at each bandwidth |
| `sensitivity.py` | Table 1 with the anchor at the likelihood bandwidth, with the anchor averaged over the grid, with the nullspace dropped, and with the Jacobian term in the bandwidth likelihood |
| `uniform_center.py` | the rejected variant: anchor rule applied to every family (Facebook) |
| `make_table1_configs.py` | writes `table1_configs.json` and `fitted_configs.json` from the selection |
| `karate_example.py` | worked example on Zachary's karate club |
| `make_tables.py`, `make_study_tables.py`, `make_figures.py` | tables and figures from the results files |
| `baselines.json` | baseline numbers of the paper with their sources |
| `table1_configs.json` | configurations and EB fits selected in Table 1 |
| `fitted_configs.json` | EB-fitted configuration of each run, at the likelihood-selected bandwidth |

Bandwidths. The finite-horizon families of a run are built at the grid bandwidth with the largest
residual-space likelihood. The equilibrium anchor uses the grid bandwidth nearest to the
graph-statistic center `gamma_0`, with its own EB fit. `PROTOCOL.md` (second amendment) states when
and why this rule was adopted, and `sensitivity.py` reports the alternatives.

Environment switches: `EBGAD_ANCHOR_BANDWIDTH` (`center`, `likelihood` or `average`) selects the anchor
rule in `select_table1.py`; `EBGAD_STUDY_BANDWIDTH=anchor` runs a study of the fitted bank at the anchor
bandwidth; `EBGAD_SUBSPACE` (`keep` or `drop`) overrides the nullspace flag a script passes;
`EBGAD_RELABEL_SEED` relabels the nodes of every dataset that is loaded; `EBGAD_INDEX_TIES=1` restores the
tie-breaking of the submitted version (only to reproduce the ordering artifact documented in the paper).
