"""
SOC-GAD — Full Pipeline
========================
End-to-end pipeline for graph anomaly detection via the SOC bridge:

1. Load dataset
2. Train SOC bridge score network (× num_trials)
3. Compute per-node anomaly scores
4. Evaluate metrics (AUC, AP, Recall@k, AUPRC)
5. Report mean ± std over trials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm
from sklearn.metrics import auc as sk_auc, precision_recall_curve

import torch_geometric.data.storage as _pyg_storage
import torch_geometric.data.data as _pyg_data
import torch_geometric.data.batch as _pyg_batch

torch.serialization.add_safe_globals([
    _pyg_storage.GlobalStorage,
    _pyg_storage.NodeStorage,
    _pyg_storage.EdgeStorage,
    _pyg_data.Data,
    _pyg_batch.Batch,
])

from data_utils import load_data
from eval_utils import mask_labeled
from pygod.metric.metric import (
    eval_roc_auc,
    eval_average_precision,
    eval_recall_at_k,
)

from soc.soc_trainer import SOCTrainer, SOCTrainerConfig
from soc.soc_anomaly import compute_anomaly_scores


@dataclass
class SOCGADConfig:
    """Full pipeline configuration."""

    dataset: str = "books"

    # ---- graph prior ----
    kappa: float = 1.0
    kappa2: float = 5.0
    nu: float = 1.0
    laplacian_variant: str = "sym"
    precision_family: str = "single_scale"
    eta: float = 0.5
    modeled_subspace: str = "legacy"
    nullspace_tol: float = 1e-8

    # ---- SOC-specific ----
    rho: float = 1.0
    lam_penalty: float = 10.0
    template_type: str = "zero"
    gamma: float = 1.0
    template_nu: float = 1.0
    pi: float = 0.5
    graph_type: str = "original"

    # ---- prior type ----
    prior_type: str = "matern"
    sigma_low: float = 1.0
    sigma_high: float = 1.0
    prior_tau: float = 1.0

    # ---- reference diffusion ----
    alpha: float = 1.0
    T: float = 1.0

    # ---- score network ----
    d_hidden: int = 256
    n_hidden_layers: int = 3
    filter_tau: float = 1.0
    parameterization: str = "score"
    score_network_type: str = "mlp"
    gnn_type: str = "gcn"
    n_gnn_layers: int = 2
    gnn_dropout: float = 0.0

    # ---- training ----
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 500
    patience: int = 50
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    grad_clip: float = 1.0

    # ---- scoring ----
    score_K: int = 10
    score_method: str = "magnitude"

    # ---- evaluation ----
    num_trials: int = 20

    # ---- numerical ----
    jitter: float = 1e-6
    target_clip: float = 10.0
    normalize_features: bool = True
    normalize_mode: str = "zscore"
    recon_features: bool = False
    n_samples_per_step: int = 1
    time_weighting: str = "uniform"

    # ---- encoder ----
    use_encoder: bool = False
    encoder_type: str = "gcn"
    encoder_hid_dim: int = 64
    encoder_num_layers: int = 4
    encoder_dropout: float = 0.3
    encoder_lr: float = 0.01
    encoder_epochs: int = 300
    encoder_alpha: float = 0.8
    encoder_weight_decay: float = 0.01

    # ---- prior mean ----
    prior_mean_mode: str = "zero"  # "stationary" = delta=x-m, "zero" = delta=x-(I-Phi)m

    # ---- eigendecomposition ----
    truncated_k: Optional[int] = None

    # ---- subgraph ----
    labeled_only: bool = False

    # ---- feature preprocessing ----
    pca_features: bool = False
    pca_n_components: int = 8

    # ---- checkpointing ----
    save_dir: Optional[str] = None

    # ---- misc ----
    verbose: bool = True
    log_every: int = 50

    def to_trainer_config(self) -> SOCTrainerConfig:
        return SOCTrainerConfig(
            kappa=self.kappa, kappa2=self.kappa2, nu=self.nu,
            laplacian_variant=self.laplacian_variant,
            precision_family=self.precision_family,
            eta=self.eta,
            modeled_subspace=self.modeled_subspace,
            nullspace_tol=self.nullspace_tol,
            rho=self.rho, lam_penalty=self.lam_penalty,
            template_type=self.template_type, gamma=self.gamma,
            template_nu=self.template_nu, pi=self.pi,
            graph_type=self.graph_type,
            prior_type=self.prior_type,
            sigma_low=self.sigma_low, sigma_high=self.sigma_high,
            prior_tau=self.prior_tau,
            alpha=self.alpha, T=self.T,
            d_hidden=self.d_hidden, n_hidden_layers=self.n_hidden_layers,
            filter_tau=self.filter_tau,
            parameterization=self.parameterization,
            score_network_type=self.score_network_type,
            gnn_type=self.gnn_type,
            n_gnn_layers=self.n_gnn_layers,
            gnn_dropout=self.gnn_dropout,
            lr=self.lr, weight_decay=self.weight_decay,
            epochs=self.epochs, patience=self.patience,
            scheduler_step=self.scheduler_step,
            scheduler_gamma=self.scheduler_gamma,
            grad_clip=self.grad_clip,
            jitter=self.jitter, target_clip=self.target_clip,
            normalize_features=self.normalize_features,
            normalize_mode=self.normalize_mode,
            prior_mean_mode=self.prior_mean_mode,
            recon_features=self.recon_features,
            n_samples_per_step=self.n_samples_per_step,
            time_weighting=self.time_weighting,
            use_encoder=self.use_encoder,
            encoder_type=self.encoder_type,
            encoder_hid_dim=self.encoder_hid_dim,
            encoder_num_layers=self.encoder_num_layers,
            encoder_dropout=self.encoder_dropout,
            encoder_lr=self.encoder_lr,
            encoder_epochs=self.encoder_epochs,
            encoder_alpha=self.encoder_alpha,
            encoder_weight_decay=self.encoder_weight_decay,
            save_dir=self.save_dir,
            pca_features=self.pca_features,
            pca_n_components=self.pca_n_components,
            truncated_k=self.truncated_k,
            verbose=self.verbose, log_every=self.log_every,
        )


@dataclass
class TrialResult:
    auc: float
    ap: float
    recall: float
    auprc: float


def evaluate_single_trial(
    data, cfg: SOCGADConfig, device: str = "cpu",
) -> TrialResult:
    # Extract labeled subgraph if requested (drops background y=-1 nodes)
    eval_data = data
    if cfg.labeled_only:
        from data_utils import extract_labeled_subgraph
        eval_data, _mask = extract_labeled_subgraph(data)

    trainer_cfg = cfg.to_trainer_config()
    trainer = SOCTrainer(trainer_cfg)
    trainer.train(eval_data, device=device)

    x_test = eval_data.x.to(device).float()
    scores = compute_anomaly_scores(
        trainer, x_test, K=cfg.score_K, method=cfg.score_method,
    )
    scores_cpu = scores.cpu().detach()
    y, scores_cpu = mask_labeled(eval_data.y, scores_cpu)
    if y is None:
        return TrialResult(auc=0.0, ap=0.0, recall=0.0, auprc=0.0)

    auc_val = eval_roc_auc(y, scores_cpu)
    ap_val = eval_average_precision(y, scores_cpu)
    rec_val = eval_recall_at_k(y, scores_cpu, y.sum().item())
    p, r, _ = precision_recall_curve(y.numpy(), scores_cpu.numpy())
    auprc_val = sk_auc(r, p)

    return TrialResult(auc=auc_val, ap=ap_val, recall=rec_val, auprc=auprc_val)


@dataclass
class PipelineResult:
    auc_mean: float
    auc_std: float
    auc_max: float
    ap_mean: float
    ap_std: float
    ap_max: float
    recall_mean: float
    recall_std: float
    recall_max: float
    auprc_mean: float
    auprc_std: float
    auprc_max: float
    per_trial: List[TrialResult] = field(default_factory=list)


def run_pipeline(
    cfg: SOCGADConfig,
    device: str = "cpu",
    data=None,
) -> PipelineResult:
    if data is None:
        data = load_data(cfg.dataset)

    trials: List[TrialResult] = []
    iterator = range(cfg.num_trials)
    if cfg.verbose:
        print(f"=== SOC-GAD: {cfg.dataset} | {cfg.num_trials} trials | "
              f"rho={cfg.rho} lam={cfg.lam_penalty} "
              f"template={cfg.template_type} ===")
        iterator = tqdm.tqdm(iterator, desc="Trials")

    for _ in iterator:
        result = evaluate_single_trial(data, cfg, device=device)
        trials.append(result)
        if cfg.verbose:
            tqdm.tqdm.write(
                f"  AUC={result.auc:.4f}  AP={result.ap:.4f}  "
                f"Rec={result.recall:.4f}  AUPRC={result.auprc:.4f}"
            )

    aucs = torch.tensor([t.auc for t in trials])
    aps = torch.tensor([t.ap for t in trials])
    recs = torch.tensor([t.recall for t in trials])
    auprcs = torch.tensor([t.auprc for t in trials])

    result = PipelineResult(
        auc_mean=aucs.mean().item(), auc_std=aucs.std().item(),
        auc_max=aucs.max().item(),
        ap_mean=aps.mean().item(), ap_std=aps.std().item(),
        ap_max=aps.max().item(),
        recall_mean=recs.mean().item(), recall_std=recs.std().item(),
        recall_max=recs.max().item(),
        auprc_mean=auprcs.mean().item(), auprc_std=auprcs.std().item(),
        auprc_max=auprcs.max().item(),
        per_trial=trials,
    )

    if cfg.verbose:
        print(
            f"\nFinal AUC: {result.auc_mean:.4f}±{result.auc_std:.4f} "
            f"({result.auc_max:.4f})\t"
            f"Final AP: {result.ap_mean:.4f}±{result.ap_std:.4f}\t"
            f"Final AUPRC: {result.auprc_mean:.4f}±{result.auprc_std:.4f}"
        )

    return result
