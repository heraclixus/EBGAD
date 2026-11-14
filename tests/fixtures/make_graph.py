"""Small attributed graph with planted anomalies, for tests that must run without datasets.

    python tests/fixtures/make_graph.py N SEED OUT.npz [--isolated K] [--one-way]

Four equal blocks with about 8 neighbours inside a block and 2 outside, 16 features
drawn around a block mean and smoothed once over the edges, and n // 20 anomalies
(at least 20): half take the features of a distant node, half are joined into
cliques of 10. --isolated K removes every edge of K normal nodes. --one-way stores
each edge once instead of in both directions.

The file holds x (N, 16) float32, edge_index (2, E) int64 and y (N,) int64, where
y = 1 marks a planted anomaly. Only Generator.random and Generator.standard_normal
are used, so a seed gives the same graph on any recent numpy.
"""
import argparse

import numpy as np


def make_graph(n, seed, isolated=0, one_way=False):
    rng = np.random.default_rng(seed)
    block = np.arange(n) * 4 // n
    size = n // 4
    p_in, p_out = 8.0 / (size - 1), 2.0 / (n - size)
    same = block[:, None] == block[None, :]
    adj = np.triu(rng.random((n, n)) < np.where(same, p_in, p_out), 1)

    mu = rng.standard_normal((4, 16))
    x = mu[block] + 0.7 * rng.standard_normal((n, 16))
    sym = adj | adj.T
    deg = sym.sum(1)
    x = 0.5 * x + 0.5 * (sym @ x) / np.maximum(deg, 1)[:, None]

    y = np.zeros(n, dtype=np.int64)
    order = np.argsort(rng.random(n), kind="stable")
    n_anom = max(20, n // 20)
    for i in order[: n_anom // 2]:
        cand = np.argsort(rng.random(n), kind="stable")[:50]
        far = cand[np.argmax(((x[cand] - x[i]) ** 2).sum(1))]
        x[i] = x[far]
        y[i] = 1
    clique_nodes = order[n_anom // 2: n_anom]
    for g in range(0, len(clique_nodes) - 9, 10):
        grp = np.sort(clique_nodes[g:g + 10])
        for a in range(10):
            adj[grp[a], grp[a + 1:]] = True
        y[grp] = 1

    if isolated:
        lonely = order[n_anom: n_anom + isolated]
        adj[lonely, :] = False
        adj[:, lonely] = False

    i, j = np.nonzero(adj)
    if one_way:
        edge_index = np.stack([i, j])
    else:
        edge_index = np.concatenate([np.stack([i, j]), np.stack([j, i])], axis=1)
    return x.astype(np.float32), edge_index.astype(np.int64), y


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--isolated", type=int, default=0)
    ap.add_argument("--one-way", action="store_true")
    args = ap.parse_args()
    x, edge_index, y = make_graph(args.n, args.seed, args.isolated, args.one_way)
    np.savez_compressed(args.out, x=x, edge_index=edge_index, y=y)
    print(f"{args.out}: {args.n} nodes, {edge_index.shape[1]} directed edges, {int(y.sum())} anomalies")


if __name__ == "__main__":
    main()
