# Fraud Data Dataset Statistics

This document verifies the statistics of datasets under `fraud_data/` for graph anomaly detection. These datasets are related to the [GGAD (NeurIPS'24)](https://github.com/mala-lab/GGAD) paper, which uses them in a **semi-supervised** setting (labeled normal nodes). DiffGAD typically uses an **unsupervised** setting.

---

## 1. Elliptic (`elliptic/`)

**Source:** Original [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) – Bitcoin transaction graph for illicit transaction detection.

**File structure:**
- `elliptic_txs_features.csv` – node features (no header; col 0 = txId)
- `elliptic_txs_classes.csv` – labels (txId, class: 1=illicit, 2=licit, unknown)
- `elliptic_txs_edgelist.csv` – edges (txId1, txId2)

**Verified statistics:**

| Metric        | Value    | Notes                                      |
|---------------|----------|--------------------------------------------|
| **Nodes**     | 203,769  | Matches features and classes               |
| **Edges**     | 234,355  | Transaction-to-transaction money flow      |
| **Feature dim** | 166    | 167 columns total (col 0 = txId)           |

**Comparison with GGAD:** GGAD reports Elliptic as 46,564 nodes, 73,248 edges, 93 attributes. That corresponds to a **filtered/processed** version (e.g., labeled subset or temporal split). The data in `fraud_data/elliptic/` is the **full** Elliptic dataset.

---

## 2. Elliptic++ (`elliptic_plus_plus/`)

**Source:** [Elliptic++ Transactions Dataset](https://github.com/git-disl/EllipticPlusPlus) (KDD'23) – extended Elliptic with richer features.

**File structure:**
- `txs_features.csv` – node features (header: txId, Time step, Local_feature_1..93, Aggregate_feature_1..72, + degree/BTC stats)
- `txs_classes.csv` – labels (txId, class: 1=illicit, 2=licit, 3=unknown)
- `txs_edgelist.csv` – edges (txId1, txId2)

**Verified statistics:**

| Metric        | Value    | Notes                                      |
|---------------|----------|--------------------------------------------|
| **Nodes**     | 203,769  | Same transaction set as Elliptic           |
| **Edges**     | 234,355  | Same graph structure as Elliptic           |
| **Feature dim** | 182    | 184 cols total (excl. txId, Time step)     |

**Comparison with official Elliptic++ README:** Official stats: 203,769 nodes, 234,355 edges, 183 features. The 1-feature difference may be due to whether Time step is counted as a feature.

---

## 3. DGraph (`dgraph/`)

**Source:** [DGraph-Fin](https://dgraph.xinye.com/) – large-scale financial fraud detection (NeurIPS'22).

**File structure:**
- `dgraphfin.npz` – x (features), y (labels), edge_index, edge_type, edge_timestamp, train_mask, valid_mask, test_mask
- `dgraphfinv2_node_timestamp.npy`, `dgraphfinv2_edge_timestamp.npy` – optional timestamps

**Label mapping:** 0=normal (1,210,092), 1=fraud (15,509), 2&3=background (2,474,949) → stored as -1 for GAD eval.

**Verified statistics:**

| Metric        | Value      | Notes                                  |
|---------------|------------|----------------------------------------|
| **Nodes**     | 3,700,550  | From dgraphfin.npz                     |
| **Edges**     | 4,300,999  | From dgraphfin.npz                      |
| **Feature dim** | 17       | Per Readme                              |

**Note:** GGAD reports 73M edges for DGraph – they may use a different version or processed graph. Our loader uses the standard `dgraphfin.npz` format.

---

## Summary table

| Dataset      | Nodes   | Edges   | Feature dim | Label format              |
|-------------|---------|---------|-------------|---------------------------|
| Elliptic    | 203,769 | 234,355 | 166         | 1=illicit, 2=licit, unknown |
| Elliptic++  | 203,769 | 234,355 | 182         | 1=illicit, 2=licit, 3=unknown |
| DGraph      | 3,700,550 | 4,300,999 | 17        | 0=normal, 1=fraud, -1=background |

---

## GGAD dataset reference (from README)

| Dataset   | Type                | Nodes     | Edges      | Attributes | Anomalies (Rate) |
|-----------|---------------------|-----------|------------|------------|------------------|
| Amazon    | Co-review           | 11,944    | 4,398,392  | 25         | 821 (6.9%)       |
| T-Finance | Transaction         | 39,357    | 1,222,543  | 10         | 1,803 (4.6%)     |
| Reddit    | Social Media        | 10,984    | 168,016    | 64         | 366 (3.3%)       |
| Elliptic  | Bitcoin Transaction | 46,564    | 73,248     | 93         | 4,545 (9.8%)     |
| Photo     | Co-purchase         | 7,535     | 119,043    | 745        | 698 (9.2%)       |
| DGraph    | Financial Networks  | 3,700,550 | 73,105,508 | 17         | 15,509 (1.3%)    |

Note: GGAD’s Elliptic numbers refer to a filtered version; the full Elliptic dataset in `fraud_data/elliptic/` has 203,769 nodes and 234,355 edges.
