"""
Encoders for Latent Bridge
============================
Autoencoders that map raw features into a latent space before the bridge.

* **GCN** (``encoder_type="gcn"``): GCN encoder/decoder with graph-aware
  message passing.  Good when graph smoothing helps.
* **GAT** (``encoder_type="gat"``): Graph Attention encoder/decoder.
  Adaptive edge weights let anomalous nodes avoid being smoothed.
* **GraphSAGE** (``encoder_type="sage"``): Sampling-based encoder.
  Scales to large graphs where GCN/GAT OOM.
* **MLP** (``encoder_type="mlp"``): Node-wise MLP encoder/decoder — no
  graph smoothing.  Preserves high-frequency anomaly signal.
* **PCA** (``encoder_type="pca"``): Deterministic PCA projection — no
  training randomness, fast, orthogonal latent features.

All share the same interface: ``encode(x, edge_index) → z`` and
``decode_features(z, edge_index) → x_hat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCN


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EncoderConfig:
    """Hyperparameters for the autoencoder."""
    encoder_type: str = "gcn"  # "gcn" or "mlp"
    hid_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.3
    lr: float = 0.01
    epochs: int = 300
    alpha: float = 0.8        # weight: α * feat_loss + (1-α) * struct_loss
    weight_decay: float = 0.01
    verbose: bool = True
    log_every: int = 50


# ═══════════════════════════════════════════════════════════════════════════
# GCN Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

class GraphEncoder(nn.Module):
    """GCN-based encoder:  (X, edge_index) → Z  ∈ R^{n × k}."""

    def __init__(self, in_dim: int, hid_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gcn = GCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=num_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act="relu",
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gcn(x, edge_index)


class GraphDecoder(nn.Module):
    """GCN-based feature decoder:  (Z, edge_index) → X̂  ∈ R^{n × d}."""

    def __init__(self, in_dim: int, out_dim: int, hid_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gcn = GCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=num_layers,
            out_channels=out_dim,
            dropout=dropout,
            act="relu",
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gcn(z, edge_index)


class GraphAutoencoder(nn.Module):
    """Full autoencoder: encoder + feature decoder + structure decoder.

    Structure is decoded via dot-product:  Â = σ(Z Zᵀ).
    """

    def __init__(self, in_dim: int, hid_dim: int = 64,
                 num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        import math
        enc_layers = math.floor(num_layers / 2)
        dec_layers = math.ceil(num_layers / 2)

        self.encoder = GraphEncoder(in_dim, hid_dim, enc_layers, dropout)
        self.decoder = GraphDecoder(hid_dim, in_dim, hid_dim, dec_layers, dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode_features(self, z: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)

    def decode_structure(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(self, x, edge_index, compute_structure: bool = True):
        z = self.encode(x, edge_index)
        x_hat = self.decode_features(z, edge_index)
        s_hat = self.decode_structure(z) if compute_structure else None
        return x_hat, s_hat, z


# ═══════════════════════════════════════════════════════════════════════════
# MLP Autoencoder (no graph smoothing)
# ═══════════════════════════════════════════════════════════════════════════

def _build_mlp(dims: list[int], dropout: float = 0.3) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    """Node-wise MLP autoencoder — no message passing.

    Same interface as :class:`GraphAutoencoder` (``encode``,
    ``decode_features``, ``decode_structure``), but the encoder and decoder
    are plain MLPs applied independently per node.  This preserves
    high-frequency features that GCN smoothing would destroy.

    The ``edge_index`` arguments are accepted but ignored, so this is a
    drop-in replacement in the training pipeline.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64,
                 num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        import math
        enc_layers = math.floor(num_layers / 2)
        dec_layers = math.ceil(num_layers / 2)

        enc_dims = [in_dim] + [hid_dim] * enc_layers
        dec_dims = [hid_dim] + [hid_dim] * (dec_layers - 1) + [in_dim]

        self.encoder_mlp = _build_mlp(enc_dims, dropout)
        self.decoder_mlp = _build_mlp(dec_dims, dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder_mlp(x)

    def decode_features(self, z: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder_mlp(z)

    def decode_structure(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(self, x, edge_index, compute_structure: bool = True):
        z = self.encode(x, edge_index)
        x_hat = self.decode_features(z, edge_index)
        s_hat = self.decode_structure(z) if compute_structure else None
        return x_hat, s_hat, z


# ═══════════════════════════════════════════════════════════════════════════
# GAT Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

class GATAutoencoder(nn.Module):
    """GAT-based autoencoder with attention-weighted message passing.

    Adaptive edge weights let anomalous nodes avoid being smoothed into
    their neighbors. Same interface as GraphAutoencoder.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64,
                 num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        import math
        from torch_geometric.nn import GAT
        enc_layers = math.floor(num_layers / 2)
        dec_layers = math.ceil(num_layers / 2)

        self.encoder_gat = GAT(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=max(enc_layers, 1),
            out_channels=hid_dim,
            dropout=dropout,
            heads=4,
            concat=False,
        )
        self.decoder_gat = GAT(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=max(dec_layers, 1),
            out_channels=in_dim,
            dropout=dropout,
            heads=4,
            concat=False,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder_gat(x, edge_index)

    def decode_features(self, z: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder_gat(z, edge_index)

    def decode_structure(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(self, x, edge_index, compute_structure: bool = True):
        z = self.encode(x, edge_index)
        x_hat = self.decode_features(z, edge_index)
        s_hat = self.decode_structure(z) if compute_structure else None
        return x_hat, s_hat, z


# ═══════════════════════════════════════════════════════════════════════════
# GraphSAGE Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

class SAGEAutoencoder(nn.Module):
    """GraphSAGE-based autoencoder. Scales better to large graphs."""

    def __init__(self, in_dim: int, hid_dim: int = 64,
                 num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        import math
        from torch_geometric.nn import GraphSAGE
        enc_layers = math.floor(num_layers / 2)
        dec_layers = math.ceil(num_layers / 2)

        self.encoder_sage = GraphSAGE(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=max(enc_layers, 1),
            out_channels=hid_dim,
            dropout=dropout,
        )
        self.decoder_sage = GraphSAGE(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=max(dec_layers, 1),
            out_channels=in_dim,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder_sage(x, edge_index)

    def decode_features(self, z: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder_sage(z, edge_index)

    def decode_structure(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(self, x, edge_index, compute_structure: bool = True):
        z = self.encode(x, edge_index)
        x_hat = self.decode_features(z, edge_index)
        s_hat = self.decode_structure(z) if compute_structure else None
        return x_hat, s_hat, z


# ═══════════════════════════════════════════════════════════════════════════
# PCA Encoder (deterministic, no training)
# ═══════════════════════════════════════════════════════════════════════════

class PCAEncoder(nn.Module):
    """Deterministic PCA projection — no training randomness.

    Fits PCA on the input features and projects to top-k components.
    The ``encode``/``decode_features`` interface is compatible with the
    autoencoder pipeline. ``edge_index`` is accepted but ignored.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = min(hid_dim, in_dim)
        self.register_buffer("components", torch.zeros(self.hid_dim, in_dim))
        self.register_buffer("mean", torch.zeros(in_dim))
        self.fitted = False

    def fit(self, x: torch.Tensor):
        """Fit PCA on data matrix x (n, d)."""
        self.mean.copy_(x.mean(dim=0))
        x_centered = x - self.mean
        # Use SVD for numerical stability
        U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)
        self.components.copy_(Vt[:self.hid_dim])
        self.fitted = True

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.components.T

    def decode_features(self, z: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        return z @ self.components + self.mean

    def decode_structure(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.T)

    def forward(self, x, edge_index, compute_structure: bool = True):
        z = self.encode(x, edge_index)
        x_hat = self.decode_features(z, edge_index)
        s_hat = self.decode_structure(z) if compute_structure else None
        return x_hat, s_hat, z


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

# Graphs with n > this use feature-only encoder loss (no structure) to avoid OOM.
# Structure loss requires dense n×n matrices (A, z@z.T) which don't fit for large graphs.
_LARGE_GRAPH_N_THRESHOLD = 100_000


def train_autoencoder(
    data,
    cfg: EncoderConfig,
    device: str = "cpu",
) -> GraphAutoencoder | MLPAutoencoder:
    """Train an autoencoder and return it (in eval mode).

    Parameters
    ----------
    data : PyG-like object with ``.x`` and ``.edge_index``.
    cfg : encoder configuration (``encoder_type`` selects GCN vs MLP).
    device : ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    Trained autoencoder (eval mode, on ``device``).

    For large graphs (n > 100k), structure loss is skipped to avoid OOM from
    dense n×n matrices. Only feature reconstruction is trained.
    """
    x = data.x.to(device).float()
    edge_index = data.edge_index.to(device)
    n = x.shape[0]
    in_dim = x.shape[1]

    # If alpha=1, the structure term has zero weight; avoid constructing the
    # dense n x n adjacency just to multiply its loss by zero.
    use_structure = (cfg.alpha < 1.0) and (n <= _LARGE_GRAPH_N_THRESHOLD)
    if use_structure:
        from torch_geometric.utils import to_dense_adj
        A = to_dense_adj(edge_index, max_num_nodes=n)[0]
    else:
        A = None
        if cfg.verbose:
            print(f"  Large graph (n={n:,}): using feature-only loss (no structure)")

    # PCA: deterministic, no training loop needed
    if cfg.encoder_type == "pca":
        model = PCAEncoder(in_dim=in_dim, hid_dim=cfg.hid_dim).to(device)
        model.fit(x)
        model.eval()
        if cfg.verbose:
            print(f"  PCA fitted: {in_dim} -> {model.hid_dim} components")
        return model

    _ENCODER_CLS = {
        "mlp": MLPAutoencoder,
        "gcn": GraphAutoencoder,
        "gat": GATAutoencoder,
        "sage": SAGEAutoencoder,
    }
    AutoencoderCls = _ENCODER_CLS.get(cfg.encoder_type, GraphAutoencoder)
    model = AutoencoderCls(
        in_dim=in_dim, hid_dim=cfg.hid_dim,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()
        x_hat, s_hat, z = model(x, edge_index, compute_structure=use_structure)

        feat_loss = ((x_hat - x) ** 2).mean()
        if use_structure:
            struct_loss = ((s_hat - A) ** 2).mean()
            loss = cfg.alpha * feat_loss + (1.0 - cfg.alpha) * struct_loss
        else:
            struct_loss = torch.tensor(0.0, device=x.device)
            loss = feat_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if cfg.verbose and epoch % cfg.log_every == 0:
            tag = f"{cfg.encoder_type.upper()}-AE"
            print(f"  {tag} Epoch {epoch:4d}/{cfg.epochs}  "
                  f"loss={loss.item():.5f}  "
                  f"feat={feat_loss.item():.5f}  "
                  f"struct={struct_loss.item():.5f}")

    model.eval()
    return model
