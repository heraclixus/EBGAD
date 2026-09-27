# Obtaining the datasets

Datasets are not committed (`data/` is about 1.3 GB, `fraud_data/` about 2.0 GB). All loading goes through
`data_utils.py` (`load_data(name)`), which expects the layout below. `fraud_data/DATASET_STATS.md` has verified
node, edge and feature statistics for the fraud datasets, and `fraud_data/verify_load.py` checks them after
download.

## 1. PyGOD datasets: automatic

`disney`, `weibo`, `reddit`, `books`, `enron` and the `inj_*` variants are fetched by `pygod.utils.load_data`,
which downloads from the [pygod-team/data](https://github.com/pygod-team/data) repository on first use and caches
locally. The labels of the injected graphs (`inj_cora`, `inj_amazon`, `inj_flickr`) are a bit mask (1 contextual,
2 structural, 3 both); `load_data` binarizes them.

## 2. Heterophilous datasets: automatic

`tolokers`, `questions` and `minesweeper` are downloaded by
`torch_geometric.datasets.HeterophilousGraphDataset` into `data/heterophilous/` on first use.

## 3. `.mat` files: place in `data/`

`load_mat_data` expects `data/<name>.mat` with keys `Network` or `A` (adjacency), `Attributes` or `X` (features)
and `Label` or `gnd` (labels).

| File | Size | Source |
|---|---|---|
| `ACM.mat`, `BlogCatalog.mat` | 7.7M, 8.7M | Standard attributed-graph benchmarks with injected anomalies; shipped with the [TAM](https://github.com/mala-lab/TAM-master) repository |
| `Amazon.mat`, `YelpChi.mat` | 5.9M, 10M | Single-relation versions from the TAM repository (originally [CARE-GNN](https://github.com/YingtongDou/CARE-GNN)) |
| `Amazon-all.mat`, `YelpChi-all.mat` | 103M, 105M | Multi-relation union, from `data/Amazon.zip` and `data/YelpChi.zip` of the CARE-GNN repository (key `homo`), saved with the keys above |
| `Facebook.mat` | 585K | TAM repository |
| `t_finance.mat` | 329M | Converted from the T-Finance dataset of [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) with `tools/convert_tfinance.py` |
| `Cora.mat`, `CiteSeer.mat`, `PubMed.mat`, `Flickr.mat` | 0.8M to 13M | Injected anomalies, from the [PREM](https://github.com/CampanulaBells/PREM-GAD) repository (`dataset/`) |

Direct links:

- `ACM.mat`, `Amazon.mat`, `BlogCatalog.mat`, `Facebook.mat`, `YelpChi.mat`:
  `https://raw.githubusercontent.com/mala-lab/TAM-master/main/data/<name>.mat`.
  The TAM Amazon file is the filtered U-P-U relation (10,224 nodes) and the TAM YelpChi file the filtered R-U-R
  relation (23,831 nodes); the multi-relation files have 11,944 and 45,954 nodes.
- T-Finance and T-Social: the BWGNN Google Drive folder
  `https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr` (`gdown --folder`), DGL
  `save_graphs` binaries that need `dgl` to read. `tools/convert_tfinance.py` writes `data/t_finance.mat`
  (label = argmax of the one-hot) and `tools/convert_tsocial.py` writes `fraud_data/tsocial/tsocial.npz`.

## 4. Fraud datasets: place in `fraud_data/`

- **Elliptic**: `fraud_data/elliptic/{elliptic_txs_features,elliptic_txs_classes,elliptic_txs_edgelist}.csv` from
  the [Elliptic Data Set on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
- **Elliptic++**: `fraud_data/elliptic_plus_plus/{txs_features,txs_classes,txs_edgelist}.csv` from
  [git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus) (Transactions dataset; Google Drive
  folder `https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l`). Elliptic's 166 features are
  the time step plus the 93 local and 72 aggregate columns of Elliptic++'s `txs_features.csv`, with the same
  edge list, so `tools/reconstruct_elliptic.py` rebuilds Elliptic from Elliptic++ (class 3 becomes unknown).
- **DGraph-Fin**: `fraud_data/dgraph/dgraphfin.npz` (680 MB) from [DGraph](https://dgraph.xinye.com/dataset),
  free registration required. Class counts: 1,210,092 normal, 15,509 fraud, 2,474,949 background.
  `run_ebgad.py --labeled-subgraph` uses the subgraph of the 1.2 million labeled nodes; the default is the full
  graph of 3.7 million nodes.

## 5. Derived caches

`cache/` holds precomputed eigendecompositions for the large graphs. Regenerate them with
`neurips2026/precompute_dgraph_eigen.py`, `neurips2026/precompute_dgraph_lowrank.py` and
`neurips2026/precompute_medium_eigen.py` once the raw data is in place.

## Verify

```bash
python -c "from data_utils import load_data; print(load_data('weibo'))"
python fraud_data/verify_load.py     # Elliptic, Elliptic++, DGraph
```
