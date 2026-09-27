"""Reconstruct the original Elliptic CSVs from the Elliptic++ transactions files.

Elliptic++ (git-disl/EllipticPlusPlus, KDD'23) extends the Kaggle Elliptic
transaction set with extra columns; the original 166 features are the Time
step plus Local_feature_1..93 and Aggregate_feature_1..72, on the same
203,769 transactions and 234,355 edges. This writes the three files that
data_utils.load_fraud_data("elliptic") expects:

    fraud_data/elliptic/elliptic_txs_features.csv   (no header; txId, 166 features)
    fraud_data/elliptic/elliptic_txs_classes.csv    (header txId,class; 1 / 2 / unknown)
    fraud_data/elliptic/elliptic_txs_edgelist.csv   (header txId1,txId2)

Usage: python tools/reconstruct_elliptic.py fraud_data/elliptic_plus_plus fraud_data/elliptic
"""
import csv
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

with open(os.path.join(src, "txs_features.csv")) as f:
    r = csv.reader(f)
    header = next(r)
    keep = [0, 1] + [i for i, h in enumerate(header)
                     if h.startswith("Local_feature_") or h.startswith("Aggregate_feature_")]
    assert len(keep) == 167, f"expected 167 columns, got {len(keep)}: {header[:5]}..."
    with open(os.path.join(dst, "elliptic_txs_features.csv"), "w", newline="") as out:
        w = csv.writer(out)
        n = 0
        for row in r:
            w.writerow([row[i] for i in keep])
            n += 1
print("features rows", n)

cls_map = {"1": "1", "2": "2", "3": "unknown"}
with open(os.path.join(src, "txs_classes.csv")) as f, \
        open(os.path.join(dst, "elliptic_txs_classes.csv"), "w", newline="") as out:
    r = csv.reader(f)
    next(r)
    w = csv.writer(out)
    w.writerow(["txId", "class"])
    counts = {}
    for row in r:
        c = cls_map.get(row[1].strip(), "unknown")
        counts[c] = counts.get(c, 0) + 1
        w.writerow([row[0], c])
print("classes", counts)

with open(os.path.join(src, "txs_edgelist.csv")) as f, \
        open(os.path.join(dst, "elliptic_txs_edgelist.csv"), "w", newline="") as out:
    r = csv.reader(f)
    next(r)
    w = csv.writer(out)
    w.writerow(["txId1", "txId2"])
    m = 0
    for row in r:
        w.writerow(row[:2])
        m += 1
print("edges", m)
