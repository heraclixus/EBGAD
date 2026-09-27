"""
Phase 2: Score Network
======================
Graph-filter augmented MLP that predicts the bridge marginal score::

    s_θ(t, x_t; G)  ≈  ∇_{x_t} log q_t(x_t)

Architecture
------------
1. **Graph-filter augmentation** (fixed, non-learnable):

       φ(x_t) = [ x_t ,  K_low x_t ,  K_high x_t ]  ∈  R^{n × 3d}

   where  K_low = exp(−τ L)  and  K_high = I − K_low.

2. **Linear projection**:  3d → hidden.
3. **Sinusoidal time embedding** → small MLP → hidden.
4. **Sum** projected features + time embedding  (broadcast over nodes).
5. **MLP body** with SiLU activations → output  ∈  R^{n × d}.

The network is applied **node-wise** with shared weights.  Graph structure
enters only through the pre-computed filter features — this is intentionally
*not* a GNN, for scalability and to isolate the bridge's contribution.

Supports two modes:
  * **Dense** (legacy):  pass ``L`` — stores full K_low/K_high as buffers.
  * **Eigenspace** (fast):  pass ``graph_eigen`` — stores eigenvectors V
    and eigenvalue vectors; forward uses V diag(f(λ)) Vᵀ @ x in place of
    dense matmuls, and avoids the O(n³) ``matrix_exp`` at init time.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge.graph_ops import heat_kernel_filters, GraphEigen, GraphEigenTruncated


# ═══════════════════════════════════════════════════════════════════════════
# Sinusoidal positional (time) embedding
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """Map a scalar time  t  to a vector of sinusoidal features.

    Follows the standard positional-encoding recipe used in DDPM / EDM.

    Parameters
    ----------
    dim : int
        Output dimension (must be even).
    max_period : float
        Controls the minimum frequency.
    """

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even."
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,)  or  scalar tensor
            Time values.

        Returns
        -------
        (B, dim)  sinusoidal features.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half = self.dim // 2
        freqs = torch.arange(half, dtype=torch.float32, device=t.device)
        freqs = freqs / (half - 1)
        freqs = (1.0 / self.max_period) ** freqs          # (half,)

        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=-1)    # (B, dim)


# ═══════════════════════════════════════════════════════════════════════════
# Score network
# ═══════════════════════════════════════════════════════════════════════════

class BridgeScoreNetwork(nn.Module):
    """Node-wise MLP score network with graph-filter feature augmentation.

    Parameters
    ----------
    d_in : int
        Raw feature dimension  *d*.
    L : (n, n) tensor, optional
        Graph Laplacian (legacy dense mode).  Mutually exclusive with
        ``graph_eigen``.
    graph_eigen : GraphEigen, optional
        Pre-computed eigendecomposition (fast eigenspace mode).
    d_hidden : int
        Hidden dimension of the MLP.
    n_hidden_layers : int
        Number of hidden layers in the MLP body (≥ 1).
    tau : float
        Heat-kernel scale for the graph filters.
    """

    def __init__(
        self,
        d_in: int,
        L: Optional[torch.Tensor] = None,
        graph_eigen: Optional[GraphEigen] = None,
        d_hidden: int = 256,
        n_hidden_layers: int = 3,
        tau: float = 1.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.tau = tau
        self._eigen_mode = False

        # ---- graph filter setup ----
        if graph_eigen is not None:
            # Eigenspace mode: store V and per-eigenvalue vectors.
            # Avoids materialising dense K_low/K_high (saves O(n²) memory)
            # and skips the O(n³) matrix_exp entirely.
            self._eigen_mode = True
            self.register_buffer("_V", graph_eigen.V)
            self.register_buffer("_kl_eig", graph_eigen.K_low_eigenvalues(tau))
            self.register_buffer("_kh_eig", graph_eigen.K_high_eigenvalues(tau))
        elif L is not None:
            K_low, K_high = heat_kernel_filters(L, tau)
            self.register_buffer("K_low", K_low)
            self.register_buffer("K_high", K_high)
        else:
            raise ValueError("Must provide either `L` or `graph_eigen`.")

        # ---- input projection: 3d → hidden ----
        self.proj = nn.Linear(3 * d_in, d_hidden)

        # ---- time embedding: scalar t → hidden ----
        self.time_embed_sin = SinusoidalTimeEmbedding(d_hidden)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # ---- MLP body ----
        layers: list[nn.Module] = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(d_hidden, d_in))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    #  Graph-filter augmentation
    # ------------------------------------------------------------------ #

    def _filter_features(self, x_t: torch.Tensor) -> torch.Tensor:
        """Compute  [x_t, K_low x_t, K_high x_t].

        In eigenspace mode this is done via  V diag(f(λ)) Vᵀ x_t  with a
        shared  Vᵀ x_t  projection, avoiding dense n×n kernel matrices.
        """
        if self._eigen_mode:
            Vt_x = self._V.T @ x_t                             # (n, d)
            x_low = self._V @ (self._kl_eig.unsqueeze(-1) * Vt_x)
            x_high = self._V @ (self._kh_eig.unsqueeze(-1) * Vt_x)
            return torch.cat([x_t, x_low, x_high], dim=-1)
        x_low = self.K_low @ x_t
        x_high = self.K_high @ x_t
        return torch.cat([x_t, x_low, x_high], dim=-1)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the bridge marginal score  s_θ(t, x_t; G).

        Parameters
        ----------
        t : scalar tensor  or  Python float
            Current time.
        x_t : (n, d)
            Node features at time *t*.

        Returns
        -------
        score : (n, d)
        """
        # 1.  Graph-filter augmentation
        phi = self._filter_features(x_t)                     # (n, 3d)

        # 2.  Project to hidden
        h = self.proj(phi)                                    # (n, hidden)

        # 3.  Time conditioning  (scalar → hidden, broadcast over nodes)
        t_tensor = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device)
        t_emb = self.time_embed_sin(t_tensor)                 # (1, hidden)
        t_emb = self.time_embed_mlp(t_emb)                    # (1, hidden)
        h = h + t_emb                                         # (n, hidden)

        # 4.  MLP → output
        return self.mlp(h)                                    # (n, d)


# ═══════════════════════════════════════════════════════════════════════════
# GNN-augmented score network
# ═══════════════════════════════════════════════════════════════════════════

_GNN_CONV_REGISTRY: dict[str, type] = {}


def _lazy_import_gnn_conv(gnn_type: str):
    """Import PyG conv layers on first use to avoid hard top-level dep."""
    if _GNN_CONV_REGISTRY:
        return _GNN_CONV_REGISTRY[gnn_type]
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    _GNN_CONV_REGISTRY["gcn"] = GCNConv
    _GNN_CONV_REGISTRY["gat"] = GATConv
    _GNN_CONV_REGISTRY["sage"] = SAGEConv
    return _GNN_CONV_REGISTRY[gnn_type]


class GNNScoreNetwork(nn.Module):
    """Score network with GNN message-passing layers.

    Same graph-filter augmentation and time conditioning as
    :class:`BridgeScoreNetwork`, but inserts GNN layers (GCN / GAT / SAGE)
    with residual connections between the input projection and the MLP head.

    The ``edge_index`` is stored as a buffer so the forward signature stays
    ``forward(t, x_t)`` — a drop-in replacement for the MLP variant.

    Parameters
    ----------
    d_in : int
        Raw feature dimension *d*.
    edge_index : (2, E) long tensor
        Graph edge list for message passing.
    L, graph_eigen : optional
        Same as :class:`BridgeScoreNetwork` — for the heat-kernel filter
        augmentation.
    d_hidden : int
        Hidden dimension.
    n_hidden_layers : int
        Number of hidden layers in the MLP body (≥ 1).
    n_gnn_layers : int
        Number of GNN message-passing layers.
    gnn_type : ``"gcn"`` | ``"gat"`` | ``"sage"``
    gnn_dropout : float
        Dropout applied inside each GNN residual block (training only).
    tau : float
        Heat-kernel scale for the graph filters.
    """

    def __init__(
        self,
        d_in: int,
        edge_index: torch.Tensor,
        L: Optional[torch.Tensor] = None,
        graph_eigen: Optional[Union[GraphEigen, GraphEigenTruncated]] = None,
        d_hidden: int = 256,
        n_hidden_layers: int = 3,
        n_gnn_layers: int = 2,
        gnn_type: str = "gcn",
        gnn_dropout: float = 0.0,
        tau: float = 1.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.tau = tau
        self.gnn_dropout = gnn_dropout
        self._eigen_mode = False

        self.register_buffer("_edge_index", edge_index)

        # ---- graph filter setup (identical to BridgeScoreNetwork) ----
        if graph_eigen is not None:
            self._eigen_mode = True
            self.register_buffer("_V", graph_eigen.V)
            self.register_buffer("_kl_eig", graph_eigen.K_low_eigenvalues(tau))
            self.register_buffer("_kh_eig", graph_eigen.K_high_eigenvalues(tau))
        elif L is not None:
            K_low, K_high = heat_kernel_filters(L, tau)
            self.register_buffer("K_low", K_low)
            self.register_buffer("K_high", K_high)
        else:
            raise ValueError("Must provide either `L` or `graph_eigen`.")

        # ---- input projection: 3d → hidden ----
        self.proj = nn.Linear(3 * d_in, d_hidden)

        # ---- time embedding: scalar t → hidden ----
        self.time_embed_sin = SinusoidalTimeEmbedding(d_hidden)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # ---- GNN layers with residual connections ----
        ConvCls = _lazy_import_gnn_conv(gnn_type)
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(n_gnn_layers):
            if gnn_type == "gat":
                conv = ConvCls(
                    d_hidden, d_hidden, heads=4,
                    concat=False, dropout=gnn_dropout,
                )
            else:
                conv = ConvCls(d_hidden, d_hidden)
            self.gnn_layers.append(conv)
            self.gnn_norms.append(nn.LayerNorm(d_hidden))

        # ---- MLP body ----
        layers: list[nn.Module] = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(d_hidden, d_in))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    #  Graph-filter augmentation (same as BridgeScoreNetwork)
    # ------------------------------------------------------------------ #

    def _filter_features(self, x_t: torch.Tensor) -> torch.Tensor:
        if self._eigen_mode:
            Vt_x = self._V.T @ x_t
            x_low = self._V @ (self._kl_eig.unsqueeze(-1) * Vt_x)
            x_high = self._V @ (self._kh_eig.unsqueeze(-1) * Vt_x)
            return torch.cat([x_t, x_low, x_high], dim=-1)
        x_low = self.K_low @ x_t
        x_high = self.K_high @ x_t
        return torch.cat([x_t, x_low, x_high], dim=-1)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        t: torch.Tensor | float,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        # 1.  Graph-filter augmentation
        phi = self._filter_features(x_t)                     # (n, 3d)

        # 2.  Project to hidden
        h = self.proj(phi)                                    # (n, hidden)

        # 3.  Time conditioning
        t_tensor = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device)
        t_emb = self.time_embed_sin(t_tensor)                 # (1, hidden)
        t_emb = self.time_embed_mlp(t_emb)                    # (1, hidden)
        h = h + t_emb                                         # (n, hidden)

        # 4.  GNN message-passing with pre-norm residual blocks
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_res = h
            h = conv(h, self._edge_index)
            h = norm(h)
            h = F.silu(h)
            if self.gnn_dropout > 0 and self.training:
                h = F.dropout(h, p=self.gnn_dropout, training=True)
            h = h + h_res

        # 5.  MLP → output
        return self.mlp(h)                                    # (n, d)
