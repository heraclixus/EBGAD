# EB-GAD

Official code for the NeurIPS 2026 paper **"Graph Anomaly Detection as Dynamical Transport: Training-Free Scoring
via Empirical Bayes"**.

EB-GAD is a training-free detector of anomalous nodes in attributed graphs. It models normal node features as a
graph Ornstein-Uhlenbeck (GOU) relaxation toward a graph-filtered template, fits the graph precision of that
process by empirical Bayes on the residual field, and scores every node by the closed-form transport energy needed
to reach its observed features. No neural network is trained, and labels are used only to report metrics.

## Repository layout

| Path | Content |
|---|---|
| `ebgad/` | The EB-GAD package: spectrum and template (`prep`), empirical-Bayes fit (`fit`), transport score bank (`scores`), exact per-node null distributions (`nulls`), label-free selection (`select`, `twogroups`), hierarchical scale prior and marginal surprise (`hier`, `surprise`). |
| `run_ebgad.py` | Entry point of the package: load, preprocess, fit, score, select, report. |
| `soc/`, `bridge/` | Core of the implementation behind the paper's experiments: GOU trainer, prior optimizer and transport scores (`soc/`); graph operators, spectra and the PCA encoder (`bridge/`). The package names are historical. |
| `neurips2026/` | The scripts behind the paper's experiments: score bank, selector, ablations, prior studies, figures. Indexed in `neurips2026/README.md`. |
| `baselines/` | Runners for the baselines. Third-party code is not redistributed, see `baselines/README.md`. |
| `ebgad/ebsmooth.py`, `ebgad/edgeleg.py`, `docs/ebsmooth.md` | Matrix-free EB smoothing (no eigendecomposition, exact null) and the edge statistic. |
| `ebgad/experiments/` | Diagnostics, gates and runners for the package, indexed in its README. |
| `tools/` | Dataset converters. |
| `data_utils.py`, `eval_utils.py` | Dataset loading and metrics. |
| `DATA.md` | How to obtain the datasets. |

## Installation

Install PyTorch and PyTorch Geometric for your platform first, then the package:

```bash
git clone https://github.com/heraclixus/EBGAD && cd EBGAD
pip install -e .                 # pip install -e ".[baselines]" adds PyOD
```

`pip install -r requirements.txt` installs the dependencies without the package; the scripts then need the
repository root on the path (`export PYTHONPATH=$PWD`). Tested on CPU with Python 3.12, torch 2.3.1,
torch_geometric 2.6.1, numpy 2.3.5, scipy 1.16.0, scikit-learn 1.7.1, pygod 1.1.0 and pyamg 5.3.0.

## Quick start

Run every command from the repository root. The PyGOD datasets download themselves on first use; the others are
described in `DATA.md`.

```bash
# a 124-node graph with the two gates, a few seconds
python run_ebgad.py --dataset disney --permute-check --null-check 30 --out results/ebgad
```

`run_ebgad.py` prints the AUROC and AUPRC of every statistic of the bank, the label-free selection and the
selection-free scan, and writes `<out>/<dataset>.json` and `<out>/<dataset>_scores.npz` (every reported score per
node). `--permute-check` reruns the pipeline on a randomly relabeled copy of the graph and reports the rank
agreement of every score, which must not depend on the node order. `--null-check B` compares the closed-form null
distributions with `B` Monte Carlo draws from the fitted model. `python run_ebgad.py --help` lists the options
(template family, bandwidth and precision grids, PCA, spectrum truncation, large-graph options). On a graph of ten
thousand nodes the first run computes the full spectrum and caches it under `cache/`; with both gates it takes
several minutes.

The defaults of `run_ebgad.py` are those of the package, not the per-dataset configurations of the paper: the
numbers of the paper come from the scripts in `neurips2026/`.

## The paper's experiments

The scripts in `neurips2026/` are the ones behind the tables and figures. The two library-like entry points are
the transport score bank with its label-free selectors, and the trajectory-level scores:

```bash
python neurips2026/test_dynamic_score_bank.py --dataset weibo
python neurips2026/test_dynamic_hypotheses.py --dataset weibo --out results/hypotheses_weibo.json
```

`--dataset` takes a name or a comma-separated list. `neurips2026/README.md` lists every script with a one-line
description. Large graphs (Elliptic, T-Finance, DGraph) use a truncated eigendecomposition, cached by the
`neurips2026/precompute_*.py` scripts.

## Matrix-free EB smoothing

`ebgad/ebsmooth.py` is a variant that needs no eigendecomposition. Features are modeled as a graph-smooth field
observed through white noise, the template is the posterior mean of the field, and its bandwidth is fitted by
marginal likelihood with block conjugate gradients, stochastic Lanczos quadrature and an algebraic multigrid
preconditioner. The residual scale of every node has an exact chi-squared null. `docs/ebsmooth.md` has the details.

```bash
PYTHONPATH=. python ebgad/experiments/ebsmooth_run.py reddit --seed 0 --permute-check --out results/ebsmooth/seed0
```

`ebgad/edgeleg.py` adds a statistic that costs one pass over the edges and no solve: the similarity of a node to
each of its neighbors, standardized on the population of edges.

```bash
PYTHONPATH=. python ebgad/experiments/edge_leg_run.py reddit --out results/edge_leg
```

## Baselines

```bash
python baselines/run_pyod_baselines.py --datasets weibo --methods LOF DIF
python baselines/run_graph_baselines.py --datasets weibo --methods DOMINANT AnomalyDAE CONAD CoLA ANOMALOUS
```

Every baseline is read in the anomaly-score direction fixed by its implementation, and none is tuned on labels.
`baselines/README.md` covers TAM, DiffGAD, FreeGAD, SmoothGNN, PREM and GADAM, whose code is cloned separately.

## Citation

The BibTeX entry will be added with the camera-ready version of the paper.
