# Baselines

Every baseline is evaluated on the same graphs and evaluation nodes as EB-GAD, in the anomaly-score direction
fixed by its implementation, and none is tuned on labels. Run the scripts from the repository root.

| Runner | Methods | Needs |
|---|---|---|
| `run_pyod_baselines.py` | LOF, ECOD, DIF, IForest on the node features | `pyod` |
| `run_graph_baselines.py` | ANOMALOUS, DOMINANT, AnomalyDAE, CONAD, CoLA | `pygod` with `pyg-lib` or `torch-sparse` (`requirements-pygod.txt`) |
| `run_pygod_fullbatch.py` | the same PyGOD detectors in full batch | `pygod` only |
| `run_tam.py` | TAM | patched clone of TAM, see below |
| `run_diffgad_cpu.py`, `diffgad_pool.py` | DiffGAD with the authors' code and configurations | clone of DiffGAD |
| `run_freegad_ours.py` | FreeGAD | clone of FreeGAD |
| `run_smoothgnn_ours.py` | SmoothGNN | clone of SmoothGNN, DGL |
| `run_smoothgnn_local.py` | SmoothGNN without DGL (exact shim for the calls the authors' code makes) | clone of SmoothGNN |
| `run_prem_ours.py` | PREM | clone of PREM, DGL |
| `run_gadam_ours.py` | GADAM | clone of GADAM, DGL |
| `run_baseline_orientation_audit.py` | native against reversed score direction, diagnostic only | |
| `run_missing_baselines.py`, `run_missing_local.py`, `run_timing_fast.py`, `run_timing_missing.py` | fill-in and timing runs | |

## Third-party code

The authors' code of the baselines is not redistributed. Clone it into `third_party/`, which git ignores, or point
the environment variable at an existing clone.

| Method | Repository | Default location | Variable |
|---|---|---|---|
| TAM | https://github.com/mala-lab/TAM-master | `third_party/TAM-master` | `TAM_DIR` |
| DiffGAD | https://github.com/fortunato-all/DiffGAD | `third_party/DiffGAD` | `DIFFGAD_DIR` |
| FreeGAD | https://github.com/yunf-zhao/FreeGAD | `third_party/FreeGAD/FreeGAD` | `FREEGAD_DIR` |
| SmoothGNN | https://github.com/xydong127/SmoothGNN | `third_party/SmoothGNN` | `SMOOTHGNN_DIR` |
| PREM | https://github.com/CampanulaBells/PREM-GAD | `third_party/PREM-GAD` | `PREM_DIR` |
| GADAM | https://github.com/PasaLab/GADAM | `third_party/GADAM` | `GADAM_DIR` |

TAM needs two small changes, a sparse-product fix in the GCN layer and the removal of an unused DGL import:

```bash
git clone https://github.com/mala-lab/TAM-master third_party/TAM-master
patch -d third_party/TAM-master -p1 < baselines/patches/tam.patch
python baselines/run_tam.py --dataset disney --device cpu --trials 1
```

Two exact replacements keep the other runners within memory: PREM's diagonal of the squared normalized adjacency
is computed sparsely instead of through a dense identity, and GADAM's index sampler receives the size argument
that recent torch versions require.
