"""Test: Can graph statistics predict the best discrete config?

Compute unsupervised graph statistics for all datasets,
then check correlation with the oracle-best discrete config.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, json
import numpy as np
import torch
from sklearn.decomposition import PCA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data


def compute_graph_stats(data):
    """Compute unsupervised graph statistics."""
    x = data.x.numpy()
    n, d = x.shape
    edge_index = data.edge_index.numpy()
    src, dst = edge_index[0], edge_index[1]
    n_edges = len(src)

    stats = {"n": n, "d": d, "n_edges": n_edges}

    # Edge density
    stats["edge_density"] = n_edges / (n * (n - 1))

    # Degree statistics
    degree = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    stats["mean_degree"] = float(np.mean(degree))
    stats["std_degree"] = float(np.std(degree))
    stats["cv_degree"] = float(np.std(degree) / (np.mean(degree) + 1e-8))

    # Feature homophily (mean cosine similarity over edges)
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    # Sample edges if too many
    if n_edges > 100000:
        idx = np.random.RandomState(42).choice(n_edges, 100000, replace=False)
        s, d_arr = src[idx], dst[idx]
    else:
        s, d_arr = src, dst
    cos_sim = np.sum(x_norm[s] * x_norm[d_arr], axis=1)
    stats["feature_homophily"] = float(np.mean(cos_sim))
    stats["feature_homophily_std"] = float(np.std(cos_sim))
    stats["frac_positive_sim"] = float(np.mean(cos_sim > 0))

    # Effective rank (PCA)
    if d > 1:
        pca = PCA(random_state=42)
        pca.fit(x[:min(n, 5000)])  # subsample for speed
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_rank_90 = int(np.searchsorted(cumvar, 0.9) + 1)
        eff_rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        stats["effective_rank_90"] = eff_rank_90
        stats["effective_rank_95"] = eff_rank_95
        stats["rank_ratio_90"] = eff_rank_90 / d
        stats["rank_ratio_95"] = eff_rank_95 / d
    else:
        stats["effective_rank_90"] = 1
        stats["rank_ratio_90"] = 1.0

    # Spectral gap (approximate via power iteration)
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh
    adj = coo_matrix((np.ones(n_edges), (src, dst)), shape=(n, n))
    adj = (adj + adj.T) / 2
    deg_vec = np.array(adj.sum(1)).flatten()
    L = diags(deg_vec) - adj
    try:
        if n < 50000:
            eigs = eigsh(L.tocsc(), k=min(10, n-2), which='SM',
                         return_eigenvectors=False)
            eigs = np.sort(eigs)
            stats["lambda_2"] = float(eigs[1]) if len(eigs) > 1 else 0
            stats["spectral_gap"] = float(eigs[1] / (eigs[-1] + 1e-8))
        else:
            stats["lambda_2"] = -1  # too large to compute
            stats["spectral_gap"] = -1
    except:
        stats["lambda_2"] = -1
        stats["spectral_gap"] = -1

    # Anomaly rate (for reference, not used for prediction)
    y = data.y.numpy()
    mask = (y >= 0) & (y <= 1)
    if mask.sum() > 0:
        stats["anomaly_rate"] = float(y[mask].mean())
    else:
        stats["anomaly_rate"] = -1

    return stats


def main():
    datasets = ["enron", "weibo", "reddit", "amazon", "yelpchi",
                "blogcatalog", "facebook", "acm",
                "elliptic", "elliptic_plus_plus", "t_finance"]

    # Best configs from our sweeps (oracle, for comparison)
    best_configs = {
        "enron": {"template": "high", "graph": "affinity", "gamma": 0.1, "pca": None, "score": "R"},
        "weibo": {"template": "low", "graph": "original", "gamma": 0.2, "pca": "yes", "score": "J*"},
        "reddit": {"template": "low", "graph": "original", "gamma": 1.0, "pca": None, "score": "J*"},
        "amazon": {"template": "low", "graph": "original", "gamma": 5.0, "pca": "yes", "score": "J*"},
        "yelpchi": {"template": "low", "graph": "original", "gamma": 1.0, "pca": "yes", "score": "J*"},
        "blogcatalog": {"template": "low", "graph": "original", "gamma": 0.1, "pca": "yes", "score": "J*"},
        "facebook": {"template": "affinity", "graph": "original", "gamma": 0.1, "pca": "yes", "score": "R"},
        "acm": {"template": "affinity", "graph": "original", "gamma": 0.5, "pca": "yes", "score": "C"},
        "elliptic": {"template": "affinity", "graph": "original", "gamma": 0.7, "pca": None, "score": "R"},
        "elliptic_plus_plus": {"template": "affinity", "graph": "original", "gamma": 0.5, "pca": None, "score": "R"},
        "t_finance": {"template": "affinity", "graph": "original", "gamma": 0.1, "pca": None, "score": "R"},
    }

    all_stats = {}
    for ds in datasets:
        print(f"=== {ds} ===", flush=True)
        load_name = ("YelpChi" if ds == "yelpchi" else
                     "Facebook" if ds == "facebook" else ds)
        data = load_data(load_name)
        stats = compute_graph_stats(data)
        stats["best_config"] = best_configs.get(ds, {})
        all_stats[ds] = stats

        bc = best_configs.get(ds, {})
        print(f"  n={stats['n']}  d={stats['d']}  "
              f"homophily={stats['feature_homophily']:.3f}  "
              f"rank90={stats.get('effective_rank_90','?')}  "
              f"gap={stats.get('spectral_gap','?'):.4f}  "
              f"best: tmpl={bc.get('template','?')} "
              f"pca={'yes' if bc.get('pca') else 'no'} "
              f"score={bc.get('score','?')}")

    with open("results/graph_stats_prediction.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    # Analysis: correlations
    print("\n=== ANALYSIS ===")

    # Template: affinity vs low/zero
    print("\nTemplate prediction (affinity vs low):")
    for ds, s in all_stats.items():
        bc = s["best_config"]
        uses_affinity = bc.get("template") == "affinity"
        print(f"  {ds:15s}  homophily={s['feature_homophily']:.3f}  "
              f"frac_pos={s['frac_positive_sim']:.3f}  "
              f"affinity={'YES' if uses_affinity else 'no'}")

    # PCA prediction
    print("\nPCA prediction:")
    for ds, s in all_stats.items():
        bc = s["best_config"]
        uses_pca = bc.get("pca") is not None and bc.get("pca") != None
        if isinstance(uses_pca, str):
            uses_pca = uses_pca == "yes"
        print(f"  {ds:15s}  d={s['d']:5d}  "
              f"rank90={s.get('effective_rank_90','?'):>4}  "
              f"ratio={s.get('rank_ratio_90',0):.2f}  "
              f"PCA={'YES' if uses_pca else 'no'}")

    # Score prediction (J* vs R)
    print("\nScore prediction (J* vs R):")
    for ds, s in all_stats.items():
        bc = s["best_config"]
        print(f"  {ds:15s}  homophily={s['feature_homophily']:.3f}  "
              f"anom_rate={s.get('anomaly_rate',0):.3f}  "
              f"score={bc.get('score','?')}")


if __name__ == "__main__":
    main()
