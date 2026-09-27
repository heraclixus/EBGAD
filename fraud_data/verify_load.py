#!/usr/bin/env python3
"""Verify fraud dataset loading and statistics.

Run from the repository root:
    python fraud_data/verify_load.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_data

FRAUD_DATASETS = ["elliptic", "elliptic_plus_plus", "dgraph"]


def main():
    print("=" * 60)
    print("Fraud dataset loading verification")
    print("=" * 60)

    for name in FRAUD_DATASETS:
        print(f"\n--- {name} ---")
        try:
            data = load_data(name)
            n = data.num_nodes
            d = data.x.shape[1]
            n_edges = data.edge_index.shape[1]
            n_anom = (data.y == 1).sum().item()
            n_norm = (data.y == 0).sum().item()
            n_unk = (data.y == -1).sum().item()

            print(f"  Nodes:      {n:,}")
            print(f"  Edges:      {n_edges:,}")
            print(f"  Feature dim: {d}")
            print(f"  Anomalies (y=1): {n_anom:,}")
            print(f"  Normal (y=0):    {n_norm:,}")
            print(f"  Unknown (y=-1):  {n_unk:,}")
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
