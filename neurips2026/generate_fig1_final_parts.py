"""Generate Figure 1 components that need LaTeX rendering.

1. GOUB formula green box (with proper LaTeX)
2. Prior formula green box
3. Toy graph spectrum (eigenvalues of karate club Laplacian)
4. Graph with anomalies (red/blue nodes)
5. Graph template (all blue nodes)
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
    "font.size": 11, "font.family": "serif",
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.transparent": True,
})
import networkx as nx
import os

OUT = "tex/figures/fig1_parts"
os.makedirs(OUT, exist_ok=True)

C_NORMAL = "#4a90d9"
C_ANOMALY = "#e74c3c"


def make_formulas():
    """Green boxes with properly rendered LaTeX."""
    # GOUB formula
    fig, ax = plt.subplots(figsize=(5.5, 0.9))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.9,
                 facecolor="#e8f5e9", edgecolor="#2ca02c", lw=2,
                 transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.5,
            r"$dX_t = -\alpha(t)\,\bar{Q}_{\rho}\,(X_t - \mathbf{m})\,dt"
            r" + \sqrt{2\alpha(t)}\,dW_t$",
            ha="center", va="center", fontsize=13,
            transform=ax.transAxes)
    fig.savefig(os.path.join(OUT, "goub_formula.pdf"))
    print("  Saved goub_formula.pdf")
    plt.close()

    # Prior formula
    fig, ax = plt.subplots(figsize=(4.0, 0.9))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.9,
                 facecolor="#e8f5e9", edgecolor="#2ca02c", lw=2,
                 transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.5,
            r"$x_0 \sim \mathcal{N}\!\left(\mathbf{m},\;"
            r"\bar{Q}_{\rho}^{-1}\right)$",
            ha="center", va="center", fontsize=13,
            transform=ax.transAxes)
    fig.savefig(os.path.join(OUT, "prior_formula.pdf"))
    print("  Saved prior_formula.pdf")
    plt.close()


def make_spectrum():
    """Eigenvalue spectrum of the karate club graph Laplacian."""
    G = nx.karate_club_graph()
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.bar(range(len(eigenvalues)), eigenvalues, color=C_NORMAL,
           edgecolor="white", linewidth=0.3, width=0.8)
    ax.set_xlabel(r"Eigenmode $j$")
    ax.set_ylabel(r"$\lambda_j$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT, "spectrum.pdf"))
    print("  Saved spectrum.pdf")
    plt.close()


def make_graphs():
    """Toy graphs: anomaly version and template version."""
    np.random.seed(42)
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G, seed=7, k=0.6)
    anomaly_nodes = {0, 33, 8}

    # (a) Graph with anomalies
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    ax.axis("off")
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                "-", color="#dddddd", lw=0.5, alpha=0.6)
    for node in G.nodes():
        x, y = pos[node]
        if node in anomaly_nodes:
            ax.scatter(x, y, c=C_ANOMALY, s=100, zorder=3,
                       edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(x, y, c=C_NORMAL, s=50, zorder=3,
                       edgecolors="white", linewidths=0.5)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(OUT, "graph_anomaly.pdf"))
    print("  Saved graph_anomaly.pdf")
    plt.close()

    # (b) Template graph (all blue)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    ax.axis("off")
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                "-", color="#dddddd", lw=0.5, alpha=0.6)
    for node in G.nodes():
        x, y = pos[node]
        ax.scatter(x, y, c=C_NORMAL, s=50, zorder=3,
                   edgecolors="white", linewidths=0.5)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(OUT, "graph_template.pdf"))
    print("  Saved graph_template.pdf")
    plt.close()


if __name__ == "__main__":
    print("=== Generating Figure 1 parts ===")
    make_formulas()
    make_spectrum()
    make_graphs()
    print("Done.")
