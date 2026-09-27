"""
DiffGAD Baseline (CPU-compatible)
=================================
A device-agnostic reimplementation of DiffGAD's core pipeline for
fair comparison on the same hardware (CPU).

This follows DiffGAD's architecture:
  1. GCN AutoEncoder → latent embeddings Z
  2. Unconditional diffusion model on Z
  3. Conditional diffusion model on Z (prototype-conditioned)
  4. Classifier-free guidance at inference
  5. AE reconstruction error as anomaly score

Simplified from the original DiffGAD.py to:
- Accept arbitrary data objects (not just PyGOD dataset names)
- Run on CPU or CUDA via a `device` parameter
- Use fewer trials for faster benchmarking
- Use sparse edge-level structure loss for large graphs (n > 50K) to avoid OOM
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc

import os as _os, sys as _sys
# The DiffGAD authors' modules (auto_encoder.py, diffusion_model.py) are not redistributed here: clone
# https://github.com/fortunato-all/DiffGAD into third_party/DiffGAD or point DIFFGAD_DIR at a clone.
_sys.path.insert(0, _os.environ.get("DIFFGAD_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "third_party", "DiffGAD")))
from auto_encoder import GraphAE
from diffusion_model import MLPDiffusion, Model, sample_dm_free
from bridge.diffgad_utils import extract, softmax_with_temperature, get_noises

sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_noises(timesteps=500)

# Graphs with n > this use sparse edge-level structure loss to avoid OOM
LARGE_GRAPH_NODES = 50_000


def _double_recon_loss_sparse(
    x: torch.Tensor,
    x_: torch.Tensor,
    edge_index: torch.Tensor,
    emb: torch.Tensor,
    weight: float = 0.5,
    pos_weight_a: float = 0.5,
) -> torch.Tensor:
    """Memory-efficient structure loss: edge-level MSE, aggregated per-node.

    Avoids dense n×n adjacency. For each edge (i,j), s_ij=1; s'_ij = emb[i]·emb[j].
    Per-node structure error = sqrt(sum over incident edges of (1 - s'_ij)²).
    """
    # Attribute reconstruction (same as double_recon_loss)
    diff_attr = torch.pow(x - x_, 2)
    if pos_weight_a != 0.5:
        diff_attr = torch.where(
            x > 0, diff_attr * pos_weight_a, diff_attr * (1 - pos_weight_a)
        )
    attr_error = torch.sqrt(torch.sum(diff_attr, 1) + 1e-8)

    # Edge-level structure: s'_ij = (emb[i] * emb[j]).sum()
    src, dst = edge_index[0], edge_index[1]
    s_pred = (emb[src] * emb[dst]).sum(dim=1)  # (E,)
    stru_err = torch.pow(1.0 - s_pred, 2)  # (E,)

    # Aggregate per-node: each edge contributes to both endpoints
    n = x.shape[0]
    per_node = torch.zeros(n, device=x.device, dtype=x.dtype)
    per_node.scatter_add_(0, src, stru_err)
    per_node.scatter_add_(0, dst, stru_err)
    stru_error = torch.sqrt(per_node + 1e-8)

    return weight * attr_error + (1 - weight) * stru_error


def run_diffgad_trial(
    data,
    device: str = "cpu",
    hid_dim: Optional[int] = None,
    diff_dim: Optional[int] = None,
    ae_epochs: int = 300,
    diff_epochs: int = 800,
    patience: int = 100,
    lr: float = 0.005,
    ae_lr: float = 0.05,
    ae_dropout: float = 0.3,
    ae_alpha: float = 0.8,
    proto_alpha: float = 0.01,
    weight: float = 1.0,
    sample_steps: int = 50,
    timesteps: int = 500,
    verbose: bool = False,
    return_scores: bool = False,
) -> dict:
    """Run one trial of DiffGAD and return metrics.

    Returns dict with keys: auc, ap, recall, auprc. If return_scores=True, adds scores, y.
    """
    dev = torch.device(device)

    x = data.x.to(dev).float()
    edge_index = data.edge_index.to(dev)
    y_full = data.y.long()
    mask = (y_full >= 0) & (y_full <= 1)
    n = x.shape[0]

    if hid_dim is None:
        hid_dim = max(4, 2 ** int(math.log2(x.size(1)) - 1))
    if diff_dim is None:
        diff_dim = 2 * hid_dim

    # ---- Train AE ----
    ae = GraphAE(in_dim=x.size(1), hid_dim=hid_dim, dropout=ae_dropout).to(dev)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=ae_lr, weight_decay=0.01)
    sch_ae = torch.optim.lr_scheduler.StepLR(opt_ae, 100, 0.5)

    use_sparse = n > LARGE_GRAPH_NODES
    s = None
    if not use_sparse:
        s = to_dense_adj(edge_index)[0]

    for epoch in range(ae_epochs):
        ae.train()
        opt_ae.zero_grad()
        if use_sparse:
            emb = ae.encode(x, edge_index)
            x_ = ae.attr_decoder(emb, edge_index)
            score = _double_recon_loss_sparse(
                x, x_, edge_index, emb, weight=ae_alpha, pos_weight_a=0.5
            )
        else:
            x_, s_, emb = ae(x, edge_index)
            score = ae.loss_func(x, x_, s, s_, ae_alpha)
        loss = score.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt_ae.step()
        sch_ae.step()

    ae.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # ---- Train unconditional DM ----
    with torch.no_grad():
        inputs = ae.encode(x, edge_index)

    denoise_fn = MLPDiffusion(hid_dim, diff_dim).to(dev)
    dm = Model(denoise_fn=denoise_fn, hid_dim=hid_dim).to(dev)
    opt_dm = torch.optim.Adam(dm.parameters(), lr=lr)
    sch_dm = torch.optim.lr_scheduler.StepLR(opt_dm, 100, 0.5)

    proto = torch.mean(inputs, dim=0)
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(diff_epochs):
        dm.train()
        loss_val, _, reconstructed = dm(inputs)
        loss_val = loss_val.mean()

        if epoch > 0:
            s_v = cos(proto, reconstructed)
            w = softmax_with_temperature(s_v, t=5).reshape(1, -1)
            proto = torch.mm(w, reconstructed).detach()

        opt_dm.zero_grad()
        loss_val.backward()
        nn.utils.clip_grad_norm_(dm.parameters(), 1.0)
        opt_dm.step()
        sch_dm.step()

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_dm_state = {k: v.clone() for k, v in dm.state_dict().items()}
            best_proto = proto.clone()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    dm.load_state_dict(best_dm_state)
    proto = best_proto

    # ---- Train conditional DM ----
    denoise_proto = MLPDiffusion(hid_dim, diff_dim).to(dev)
    dm_proto = Model(denoise_fn=denoise_proto, hid_dim=hid_dim).to(dev)
    opt_proto = torch.optim.Adam(dm_proto.parameters(), lr=lr)
    sch_proto = torch.optim.lr_scheduler.StepLR(opt_proto, 100, 0.5)

    best_loss = float("inf")
    patience_ctr = 0
    for epoch in range(diff_epochs):
        dm_proto.train()
        loss_val, _, _ = dm_proto(inputs, proto=proto, proto_alpha=proto_alpha)
        loss_val = loss_val.mean()
        opt_proto.zero_grad()
        loss_val.backward()
        nn.utils.clip_grad_norm_(dm_proto.parameters(), 1.0)
        opt_proto.step()
        sch_proto.step()
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_proto_state = {k: v.clone() for k, v in dm_proto.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    dm_proto.load_state_dict(best_proto_state)

    # ---- Evaluate: forward noise → denoise → decode → recon error ----
    ae.eval()
    dm.eval()
    dm_proto.eval()
    proto_net = dm_proto.denoise_fn_D
    free_net = dm.denoise_fn_D

    with torch.no_grad():
        Z_0 = ae.encode(x, edge_index)
        noise = torch.randn_like(Z_0)

        best_auc = 0.0
        best_metrics = {}

        for i in range(timesteps):
            t = torch.tensor([i] * n).long().to(dev)
            sqrt_ac = extract(sqrt_alphas_cumprod, t, Z_0.shape)
            sqrt_omc = extract(sqrt_one_minus_alphas_cumprod, t, Z_0.shape)
            Z_t = sqrt_ac * Z_0 + sqrt_omc * noise

            reconstructed = sample_dm_free(
                proto_net, free_net, Z_t, sample_steps,
                proto=proto, proto_alpha=proto_alpha, weight=weight,
            )
            if use_sparse:
                x_ = ae.attr_decoder(reconstructed, edge_index)
                score = _double_recon_loss_sparse(
                    x, x_, edge_index, reconstructed,
                    weight=ae_alpha, pos_weight_a=0.5,
                )
            else:
                x_, s_ = ae.decode(reconstructed, edge_index)
                score = ae.loss_func(x, x_, s, s_, ae_alpha)

            from eval_utils import mask_labeled
            y_eval, score_eval = mask_labeled(y_full, score)
            if y_eval is not None:
                y_np = y_eval.cpu().numpy()
                s_np = score_eval.cpu().numpy()
                auc_val = roc_auc_score(y_np, s_np)
            else:
                auc_val = 0.0
            if auc_val > best_auc:
                best_auc = auc_val
                auprc_val = 0.0
                if y_eval is not None:
                    p, r, _ = precision_recall_curve(y_np, s_np)
                    auprc_val = sk_auc(r, p)
                best_metrics = {
                    "auc": auc_val,
                    "auprc": auprc_val,
                }
                if return_scores and y_eval is not None:
                    best_metrics["scores"] = s_np.copy()
                    best_metrics["y"] = y_np.copy()

            if verbose and i % 50 == 0:
                print(f"  DiffGAD t={i}: AUC={auc_val:.4f}")

    if return_scores and "scores" not in best_metrics:
        best_metrics["scores"] = np.array([])
        best_metrics["y"] = np.array([])
    return best_metrics


def run_diffgad_benchmark(
    data,
    device: str = "cpu",
    num_trials: int = 5,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Run multiple DiffGAD trials and return aggregated results."""
    aucs, auprcs = [], []
    for trial in range(num_trials):
        if verbose:
            print(f"DiffGAD trial {trial + 1}/{num_trials}")
        r = run_diffgad_trial(data, device=device, verbose=verbose, **kwargs)
        aucs.append(r["auc"])
        auprcs.append(r["auprc"])
        if verbose:
            print(f"  → AUC={r['auc']:.4f}")

    return {
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "auprc_mean": np.mean(auprcs),
        "auprc_std": np.std(auprcs),
    }
