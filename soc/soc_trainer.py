"""
SOC Bridge Trainer
===================
Training loop for the SOC bridge score-matching model.

Each training step:

1. x_T  ← data node features
2. x_0  ~ p_prior(· | G)      (Matérn or two-band)
3. t    ~ Uniform(ε, T − ε)
4. Compute SOC bridge statistics (a_t, b_t, c_t, Σ'_t)
5. x_t  ~ N(μ_{t,λ}, Σ'_{t,λ})
6. target = −(Σ'_t)^{−1}(x_t − μ_t)  or  ε-prediction
7. loss  = ‖s_θ(t, x_t; G) − target‖²
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from bridge.graph_ops import (
    compute_laplacian,
    shifted_laplacian_precision,
    GraphEigen,
    GraphEigenTruncated,
)
from bridge.score_network import BridgeScoreNetwork
from soc.soc_bridge import (
    compute_Q_rho_eigenvalues,
    compute_prior_base_eigenvalues,
    compute_template,
    compute_local_affinity,
    affinity_to_weights,
    compute_affinity_refined_laplacian,
    compute_affinity_refined_laplacian_sparse,
    apply_modeled_subspace,
    soc_bridge_statistics,
    soc_bridge_sample,
    soc_score_target,
    soc_epsilon_target,
    LearnableRho,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SOCTrainerConfig:
    """All hyperparameters for the SOC bridge trainer."""

    # ---- graph prior ----
    kappa: float = 1.0
    kappa2: float = 5.0
    nu: float = 1.0
    laplacian_variant: str = "sym"
    precision_family: str = "single_scale"  # "single_scale" or "two_scale"
    eta: float = 0.5
    modeled_subspace: str = "legacy"  # "legacy" or "drop_nullspace"
    nullspace_tol: float = 1e-8

    # ---- SOC-specific ----
    rho: float = 1.0              # geometry blending [0, 1] (init for learnable)
    rho_mode: str = "fixed"       # "fixed", "learned_scalar", "learned_diag", "learned_mask"
    rho_mask_tau_start: float = 5.0   # mask mode: initial temperature
    rho_mask_tau_end: float = 0.5     # mask mode: final temperature
    lam_penalty: float = 10.0     # terminal penalty λ
    template_type: str = "zero"   # "zero", "low", "high", "mix", "affinity"
    gamma: float = 1.0            # smoother scale for template
    template_nu: float = 1.0      # smoother order for template
    pi: float = 0.5               # mixing weight for "mix" template
    graph_type: str = "original"  # "original" or "affinity" (affinity-refined L̃)

    # ---- prior type ----
    prior_type: str = "matern"    # "matern", "two_band", or "normal"
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
    parameterization: str = "score"   # "score" or "epsilon"
    score_network_type: str = "mlp"   # "mlp" or "gnn"
    gnn_type: str = "gcn"             # "gcn", "gat", or "sage" (only when gnn)
    n_gnn_layers: int = 2
    gnn_dropout: float = 0.0

    # ---- optimisation ----
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 500
    patience: int = 50
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    grad_clip: float = 1.0

    # ---- numerical ----
    jitter: float = 1e-6
    target_clip: float = 10.0
    normalize_features: bool = True
    n_samples_per_step: int = 1
    time_weighting: str = "uniform"

    # ---- prior mean ----
    prior_mean_mode: str = "zero"  # "stationary" or "zero"

    # ---- encoder ----
    use_encoder: bool = False
    recon_features: bool = False
    encoder_type: str = "gcn"     # "gcn" or "mlp"
    encoder_hid_dim: int = 64
    encoder_num_layers: int = 4
    encoder_dropout: float = 0.3
    encoder_lr: float = 0.01
    encoder_epochs: int = 300
    encoder_alpha: float = 0.8
    encoder_weight_decay: float = 0.01

    # ---- checkpointing ----
    save_dir: Optional[str] = None

    # ---- memory (large graphs) ----
    truncated_k: Optional[int] = None  # None = auto-scale by n; else explicit k for GraphEigenTruncated

    # ---- feature preprocessing ----
    pca_features: bool = False     # Apply PCA after z-score normalization
    pca_n_components: int = 8      # Number of PCA components to keep
    normalize_mode: str = "zscore"  # "zscore", "minmax", "robust"

    # ---- P16 auxiliary directional loss ----
    aux_directional_weight: float = 0.0   # Weight for L_aux = -E[<s_θ,v>·𝟙[‖s_θ‖>τ]]
    aux_directional_tau: float = 0.5      # Threshold for high-magnitude nodes
    aux_directional_warmup_epochs: int = 0  # Start aux loss only after N epochs (0=from start)

    # ---- misc ----
    verbose: bool = True
    log_every: int = 10


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════

class SOCTrainer:
    """SOC bridge score-matching trainer.

    Usage::

        cfg = SOCTrainerConfig(kappa=1.0, rho=0.8, lam_penalty=10.0, ...)
        trainer = SOCTrainer(cfg)
        score_net = trainer.train(data, device='cpu')
    """

    def __init__(self, config: SOCTrainerConfig):
        self.cfg = config

        self.graph_eigen: Optional[GraphEigen] = None
        self.lam_Q_rho: Optional[torch.Tensor] = None
        self.lam_Q_base: Optional[torch.Tensor] = None
        self.rho_module: Optional[LearnableRho] = None
        self.V: Optional[torch.Tensor] = None
        self.V_template: Optional[torch.Tensor] = None
        self.template: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.Q_eigen: Optional[tuple] = None
        self.score_net: Optional[BridgeScoreNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.autoencoder: Optional[Any] = None
        self.edge_index: Optional[torch.Tensor] = None

        self.device: torch.device = torch.device("cpu")
        self.x_T: Optional[torch.Tensor] = None
        self.d_in: int = 0
        self.loss_history: list[float] = []
        self._epoch: int = 0

        self.feat_mean: Optional[torch.Tensor] = None
        self.feat_std: Optional[torch.Tensor] = None
        self.lam_L_model: Optional[torch.Tensor] = None
        self.modeled_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def _setup(self, data: Any, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        cfg = self.cfg

        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device)
        self.edge_index = edge_index
        n = x.shape[0]

        # Optional encoder
        if cfg.use_encoder:
            from bridge.encoder import (
                GraphAutoencoder, train_autoencoder, EncoderConfig,
            )
            enc_cfg = EncoderConfig(
                encoder_type=cfg.encoder_type,
                hid_dim=cfg.encoder_hid_dim,
                num_layers=cfg.encoder_num_layers,
                dropout=cfg.encoder_dropout,
                lr=cfg.encoder_lr,
                epochs=cfg.encoder_epochs,
                alpha=cfg.encoder_alpha,
                weight_decay=cfg.encoder_weight_decay,
                verbose=cfg.verbose,
                log_every=cfg.log_every,
            )
            self.autoencoder = train_autoencoder(data, enc_cfg, device=device)
            with torch.no_grad():
                z = self.autoencoder.encode(x, edge_index)
                if cfg.recon_features:
                    x_hat = self.autoencoder.decode_features(z, edge_index)
                    recon_err = ((x_hat - x) ** 2).sum(dim=-1, keepdim=True)
                    x = torch.cat([z, recon_err], dim=-1)
                else:
                    x = z

        self.d_in = x.shape[1]

        # Feature normalisation
        if cfg.normalize_features:
            mode = getattr(cfg, "normalize_mode", "zscore")
            if mode == "minmax":
                self.feat_mean = x.min(dim=0, keepdim=True).values
                self.feat_std = (x.max(dim=0, keepdim=True).values - self.feat_mean).clamp(min=1e-8)
            elif mode == "robust":
                self.feat_mean = x.median(dim=0, keepdim=True).values
                q25 = torch.quantile(x, 0.25, dim=0, keepdim=True)
                q75 = torch.quantile(x, 0.75, dim=0, keepdim=True)
                self.feat_std = (q75 - q25).clamp(min=1e-8)
            else:  # zscore
                self.feat_mean = x.mean(dim=0, keepdim=True)
                self.feat_std = x.std(dim=0, keepdim=True).clamp(min=1e-8)
            x = (x - self.feat_mean) / self.feat_std
        else:
            self.feat_mean = torch.zeros(1, self.d_in, device=self.device)
            self.feat_std = torch.ones(1, self.d_in, device=self.device)

        # PCA: reduce to orthogonal components (removes multicollinearity)
        self.pca_components = None
        if cfg.pca_features:
            k = min(cfg.pca_n_components, x.shape[1])
            U, S, Vt = torch.linalg.svd(x - x.mean(dim=0, keepdim=True), full_matrices=False)
            self.pca_components = Vt[:k].T  # (d_in, k)
            x = x @ self.pca_components     # (n, k)
            self.d_in = x.shape[1]

        self.x_T = x

        # Graph eigendecomposition (one-time) on original graph
        # For large graphs (n >= 20K): use GraphEigenTruncated to avoid dense n×n
        # matrices that cause GPU segfault. LOBPCG uses sparse Laplacian only.
        # truncated_k: V matrix is (n,k) — main memory cost. Auto-scale for GPU.
        _LARGE_GRAPH_THRESHOLD = 20_000
        if cfg.truncated_k is not None:
            _truncated_k = cfg.truncated_k
        else:
            # Scale k by n to keep V under ~2GB: n*k*4bytes <= 2e9 => k <= 5e8/n
            if n >= 1_000_000:
                _truncated_k = 128   # 3.7M*128*4 ≈ 1.9 GB
            elif n >= 500_000:
                _truncated_k = 200
            elif n >= 100_000:
                _truncated_k = 300   # elliptic ~200K
            else:
                _truncated_k = 500
        lap_var = cfg.laplacian_variant if cfg.laplacian_variant != "rw" else "sym"
        if cfg.laplacian_variant == "rw" and n >= _LARGE_GRAPH_THRESHOLD:
            import warnings
            warnings.warn(
                f"GraphEigenTruncated does not support laplacian_variant='rw' "
                f"(n={n}); using 'sym' instead.",
                UserWarning,
            )

        # Affinity-refined Laplacian: use sparse path for n >= 8K to avoid dense
        # linalg.eigh failures (Intel MKL "Parameter 8 illegal" on Amazon ~10K nodes).
        _AFFINITY_DENSE_THRESHOLD = 8_000
        if cfg.graph_type == "affinity" and n >= _AFFINITY_DENSE_THRESHOLD:
            # Sparse affinity path: no dense n×n, use LOBPCG on sparse L̃
            L_sparse = compute_affinity_refined_laplacian_sparse(
                edge_index, x, n, variant=lap_var,
            )
            self.graph_eigen = GraphEigenTruncated.from_sparse_laplacian(
                L_sparse, k=_truncated_k,
            )
            lam_L, V = self.graph_eigen.lam_L, self.graph_eigen.V
        elif cfg.graph_type == "affinity" and n < _AFFINITY_DENSE_THRESHOLD:
            # Dense affinity path (small graphs only)
            L_refined = compute_affinity_refined_laplacian(
                edge_index, x, n, variant=cfg.laplacian_variant,
            )
            lam_L, V = torch.linalg.eigh(L_refined)
            self.graph_eigen = GraphEigen.from_eigenpairs(lam_L, V, L=L_refined)
        elif n >= _LARGE_GRAPH_THRESHOLD:
            self.graph_eigen = GraphEigenTruncated(
                edge_index, n, k=_truncated_k, variant=lap_var,
            )
            lam_L, V = self.graph_eigen.lam_L, self.graph_eigen.V
        else:
            self.graph_eigen = GraphEigen(edge_index, n, variant=cfg.laplacian_variant)
            lam_L, V = self.graph_eigen.lam_L, self.graph_eigen.V

        self.V_template = V
        lam_L_model, V_model, modeled_mask = apply_modeled_subspace(
            lam_L,
            V,
            mode=cfg.modeled_subspace,
            tol=cfg.nullspace_tol,
        )
        self.lam_L_model = lam_L_model
        self.modeled_mask = modeled_mask
        self.V = V_model

        # Base spectral precision (before rho blending) on the modeled subspace.
        self.lam_Q_base = compute_prior_base_eigenvalues(
            lam_L_model,
            kappa=cfg.kappa,
            nu=cfg.nu,
            precision_family=cfg.precision_family,
            eta=cfg.eta,
            kappa2=cfg.kappa2,
        )

        # Q_ρ eigenvalues — fixed or via learnable module
        if cfg.rho_mode == "fixed":
            self.lam_Q_rho = compute_Q_rho_eigenvalues(
                lam_L_model,
                rho=cfg.rho,
                kappa=cfg.kappa,
                nu=cfg.nu,
                precision_family=cfg.precision_family,
                eta=cfg.eta,
                kappa2=cfg.kappa2,
            )
            self.rho_module = None
        else:
            k = lam_L_model.shape[0]
            self.rho_module = LearnableRho(
                k=k,
                mode=cfg.rho_mode,
                init_rho=cfg.rho,
                tau_start=cfg.rho_mask_tau_start,
                tau_end=cfg.rho_mask_tau_end,
            ).to(self.device)
            self.lam_Q_rho = self._get_lam_Q_rho().detach()

        # Node affinity weights (for all affinity-variant templates)
        node_weights = None
        if cfg.template_type.startswith("affinity"):
            aff = compute_local_affinity(x, edge_index, n)
            node_weights = affinity_to_weights(aff)

        # Template
        self.template = compute_template(
            x, self.V_template, lam_L,
            template_type=cfg.template_type,
            gamma=cfg.gamma, nu=cfg.template_nu, pi=cfg.pi,
            node_weights=node_weights,
        )

        # Legacy Q for prior sampling (always from original graph)
        # Skip dense Q for GraphEigenTruncated (L is sparse; dense would OOM)
        if n < _LARGE_GRAPH_THRESHOLD or not hasattr(self.graph_eigen, "k"):
            self.Q = shifted_laplacian_precision(
                self.graph_eigen.L, kappa=cfg.kappa, nu=cfg.nu,
            )
        else:
            self.Q = None
        self.Q_eigen = self.graph_eigen.bridge_eigen_cache(cfg.kappa, cfg.nu)

        # Score network
        if cfg.score_network_type == "gnn":
            from bridge.score_network import GNNScoreNetwork
            self.score_net = GNNScoreNetwork(
                d_in=self.d_in,
                edge_index=edge_index,
                graph_eigen=self.graph_eigen,
                d_hidden=cfg.d_hidden,
                n_hidden_layers=cfg.n_hidden_layers,
                n_gnn_layers=cfg.n_gnn_layers,
                gnn_type=cfg.gnn_type,
                gnn_dropout=cfg.gnn_dropout,
                tau=cfg.filter_tau,
            ).to(self.device)
        else:
            self.score_net = BridgeScoreNetwork(
                d_in=self.d_in,
                graph_eigen=self.graph_eigen,
                d_hidden=cfg.d_hidden,
                n_hidden_layers=cfg.n_hidden_layers,
                tau=cfg.filter_tau,
            ).to(self.device)

        # Optimiser — include learnable rho parameters if applicable
        param_groups = [
            {"params": self.score_net.parameters()},
        ]
        if self.rho_module is not None:
            param_groups.append({
                "params": self.rho_module.parameters(),
                "lr": cfg.lr * 0.1,
            })
        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma,
        )

    # ------------------------------------------------------------------ #
    #  Normalisation
    # ------------------------------------------------------------------ #

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_encoder and self.autoencoder is not None:
            with torch.no_grad():
                z = self.autoencoder.encode(x, self.edge_index)
                if self.cfg.recon_features:
                    x_hat = self.autoencoder.decode_features(z, self.edge_index)
                    recon_err = ((x_hat - x) ** 2).sum(dim=-1, keepdim=True)
                    x = torch.cat([z, recon_err], dim=-1)
                else:
                    x = z
        x = (x - self.feat_mean) / self.feat_std
        if self.pca_components is not None:
            x = x @ self.pca_components
        return x

    # ------------------------------------------------------------------ #
    #  Learnable rho
    # ------------------------------------------------------------------ #

    def _get_lam_Q_rho(self) -> torch.Tensor:
        """Compute current Q_rho eigenvalues (differentiable for learnable modes)."""
        if self.rho_module is None:
            return self.lam_Q_rho
        progress = self._epoch / max(self.cfg.epochs, 1)
        return self.rho_module(self.lam_Q_base, progress=progress)

    # ------------------------------------------------------------------ #
    #  Prior sampling
    # ------------------------------------------------------------------ #

    def _sample_prior(self) -> torch.Tensor:
        cfg = self.cfg
        ge = self.graph_eigen
        n = self.x_T.shape[0]
        if cfg.prior_type == "normal":
            return torch.randn(n, self.d_in, device=self.device)
        elif cfg.prior_type == "matern":
            return ge.sample_matern_prior(self.d_in, cfg.kappa, cfg.nu)
        elif cfg.prior_type == "two_band":
            return ge.sample_two_band_prior(
                self.d_in, tau=cfg.prior_tau,
                sigma_low=cfg.sigma_low, sigma_high=cfg.sigma_high,
            )
        else:
            raise ValueError(f"Unknown prior_type: {cfg.prior_type!r}")

    # ------------------------------------------------------------------ #
    #  Training step
    # ------------------------------------------------------------------ #

    def _sample_time(self) -> float:
        cfg = self.cfg
        eps_t = 1e-3 * cfg.T
        if cfg.time_weighting == "importance":
            import math
            u = torch.rand(1, device=self.device).item()
            t = cfg.T * (math.acos(1.0 - 2.0 * u) / math.pi)
            return max(eps_t, min(t, cfg.T - eps_t))
        else:
            return eps_t + (cfg.T - 2.0 * eps_t) * torch.rand(
                1, device=self.device).item()

    def train_step(self) -> float:
        cfg = self.cfg
        self.score_net.train()
        x_T = self.x_T
        m = self.template

        lam_Q = self._get_lam_Q_rho()

        total_loss = torch.tensor(0.0, device=self.device)
        for _ in range(cfg.n_samples_per_step):
            x_0 = self._sample_prior()
            t = self._sample_time()

            stats = soc_bridge_statistics(
                lam_Q, self.V,
                alpha=cfg.alpha, t=t, T=cfg.T,
                lam_penalty=cfg.lam_penalty,
                jitter=cfg.jitter,
            )
            x_t, mu_t = soc_bridge_sample(stats, x_0, x_T, m)

            pred = self.score_net(t, x_t.detach())

            if cfg.parameterization == "epsilon":
                target = soc_epsilon_target(stats, x_t, mu_t)
            else:
                target = soc_score_target(stats, x_t, mu_t)

            if cfg.target_clip > 0:
                target = target.clamp(-cfg.target_clip, cfg.target_clip)
            target = target.detach()

            if cfg.time_weighting == "importance":
                import math
                w = math.sin(math.pi * t / cfg.T)
            else:
                w = 1.0

            total_loss = total_loss + w * ((pred - target) ** 2).mean()

        loss = total_loss / cfg.n_samples_per_step

        # P16b/P16e: Auxiliary directional loss (optional, with optional warmup)
        aux_weight = getattr(cfg, "aux_directional_weight", 0.0)
        aux_warmup = getattr(cfg, "aux_directional_warmup_epochs", 0)
        if aux_weight > 0 and self._epoch >= aux_warmup:
            v = self.feat_std.squeeze() / self.feat_std.squeeze().norm().clamp(min=1e-8)
            tau = getattr(cfg, "aux_directional_tau", 0.5)
            # pred: (n, d), v: (d,). For high-magnitude nodes, encourage alignment
            pred_norm = pred.norm(dim=-1)
            mask = (pred_norm > tau).float()
            if mask.sum() > 0:
                align = (pred * v).sum(dim=-1)
                L_aux = -(align * mask).sum() / mask.sum().clamp(min=1)
                loss = loss + aux_weight * L_aux

        self.optimizer.zero_grad()
        loss.backward()
        all_params = list(self.score_net.parameters())
        if self.rho_module is not None:
            all_params += list(self.rho_module.parameters())
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------ #
    #  Full training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        data: Any,
        device: str | torch.device = "cpu",
    ) -> BridgeScoreNetwork:
        cfg = self.cfg
        self._setup(data, device)

        best_loss = float("inf")
        self.best_loss = float("inf")
        patience_counter = 0
        best_state: Dict[str, Any] = {}
        self.loss_history = []

        for epoch in range(cfg.epochs):
            self._epoch = epoch
            loss = self.train_step()
            self.loss_history.append(loss)
            self.scheduler.step()

            if cfg.verbose and (epoch + 1) % cfg.log_every == 0:
                rho_str = ""
                if self.rho_module is not None:
                    rho_val = self.rho_module.get_rho(epoch / max(cfg.epochs, 1))
                    if rho_val.dim() == 0:
                        rho_str = f"  rho={rho_val.item():.3f}"
                    else:
                        rho_str = f"  rho=[{rho_val.min():.3f},{rho_val.mean():.3f},{rho_val.max():.3f}]"
                print(f"Epoch {epoch + 1:4d}/{cfg.epochs}  loss={loss:.6f}{rho_str}")

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                best_state = {
                    "epoch": epoch,
                    "score_net": copy.deepcopy(self.score_net.state_dict()),
                    "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                    "loss": best_loss,
                }
                if self.rho_module is not None:
                    best_state["rho_module"] = copy.deepcopy(
                        self.rho_module.state_dict()
                    )
                if cfg.save_dir is not None:
                    os.makedirs(cfg.save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(cfg.save_dir, "best.pt"))
            else:
                patience_counter += 1
                if cfg.patience > 0 and patience_counter >= cfg.patience:
                    if cfg.verbose:
                        print(f"Early stopping at epoch {epoch + 1} "
                              f"(best_loss={best_loss:.6f})")
                    break

        if best_state:
            self.score_net.load_state_dict(best_state["score_net"])
            if self.rho_module is not None and "rho_module" in best_state:
                self.rho_module.load_state_dict(best_state["rho_module"])

        # Freeze learned rho into self.lam_Q_rho for scoring
        if self.rho_module is not None:
            self.lam_Q_rho = self._get_lam_Q_rho().detach()

        self.best_loss = best_loss

        if cfg.verbose:
            print(f"Training complete.  Best loss={best_loss:.6f} "
                  f"at epoch {best_state.get('epoch', -1) + 1}")

        return self.score_net

    # ------------------------------------------------------------------ #
    #  Checkpoint loading
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.score_net.load_state_dict(ckpt["score_net"])
        if self.optimizer is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
