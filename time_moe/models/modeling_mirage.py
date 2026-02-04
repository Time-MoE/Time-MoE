# time_moe/models/modeling_mirage.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from .modeling_time_moe import TimeMoePreTrainedModel, TimeMoeModel


@dataclass
class MiragePretrainOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_gen: Optional[torch.FloatTensor] = None
    loss_rank: Optional[torch.FloatTensor] = None
    loss_align: Optional[torch.FloatTensor] = None
    z: Optional[torch.FloatTensor] = None
    u: Optional[torch.FloatTensor] = None
    bin_ids: Optional[torch.LongTensor] = None


def _pool_snapshot(hidden_states: torch.Tensor, pooling: str, loss_masks: Optional[torch.Tensor] = None):
    # hidden_states: [N, L, H]
    if pooling == "last":
        return hidden_states[:, -1, :]
    if pooling == "mean":
        if loss_masks is None:
            return hidden_states.mean(dim=1)
        w = loss_masks.unsqueeze(-1)  # [N,L,1]
        return (hidden_states * w).sum(dim=1) / (w.sum(dim=1).clamp_min(1e-6))
    raise ValueError(f"Unknown pooling={pooling}")


def _assign_quantile_bins(z_detached: torch.Tensor, M: int) -> torch.LongTensor:
    """
    z_detached: [N] (no grad)
    Returns bin ids in [0, M-1] with approx equal counts.
    """
    N = z_detached.numel()
    order = torch.argsort(z_detached)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(N, device=z_detached.device)
    bin_ids = (ranks * M) // max(N, 1)
    bin_ids = torch.clamp(bin_ids, max=M - 1)
    return bin_ids.long()


def _pairwise_soft_rank_loss(
    z: torch.Tensor,            # [N]
    t: torch.Tensor,            # [N]
    entity_ids: torch.Tensor,   # [N]
    beta: torch.Tensor,         # scalar
    use_pair_weight: bool = True,
) -> torch.Tensor:
    """
    Noise-aware weak-order loss:
      p_ij = sigmoid(beta * (t_j - t_i))
      q_ij = sigmoid(z_j - z_i)
      L = BCEWithLogits(z_j-z_i, p_ij) with optional w_ij=2|p_ij-0.5|
    """
    device = z.device
    uniq = torch.unique(entity_ids)
    losses = []

    for eid in uniq:
        idx = (entity_ids == eid).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        ze = z[idx]  # [K]
        te = t[idx]  # [K]
        K = ze.numel()

        # all unordered pairs (i<j)
        ij = torch.triu_indices(K, K, offset=1, device=device)
        i, j = ij[0], ij[1]

        dt = te[j] - te[i]                          # [P]
        p = torch.sigmoid(beta * dt)                # soft target
        logit = ze[j] - ze[i]                       # predicted logit

        per = F.binary_cross_entropy_with_logits(logit, p, reduction="none")
        if use_pair_weight:
            w = 2.0 * torch.abs(p - 0.5)
            per = per * w

        losses.append(per.mean())

    if len(losses) == 0:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


class TimeMoeForMirage(TimeMoePreTrainedModel):
    """
    MIRAGE pretraining model:
      - backbone: TimeMoeModel (decoder-only, MoE optional, supports stage-conditioned routing)
      - generative head: next-step regression (Huber) on snapshot window
      - progression head: stage z + embedding u
      - losses: L = L_gen + λ_rank L_rank + λ_align L_align
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = TimeMoeModel(config)

        H = config.hidden_size
        F_in = config.input_size

        # --- generative head (1-step) ---
        self.pred_head = nn.Linear(H, F_in)

        # --- progression heads ---
        proj_dim = int(getattr(config, "mirage_proj_dim", 128))
        self.stage_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, 1),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, proj_dim),
        )

        # --- beta (fixed or learnable) ---
        beta0 = float(getattr(config, "mirage_rank_beta", 0.02))
        learn_beta = bool(getattr(config, "mirage_learn_beta", False))
        self.learn_beta = learn_beta
        if learn_beta:
            # softplus(log_beta) > 0
            self.log_beta = nn.Parameter(torch.log(torch.tensor(beta0)))
        else:
            self.register_buffer("beta_const", torch.tensor(beta0), persistent=False)

        # --- prototypes ---
        M = int(getattr(config, "mirage_num_prototypes", 10))
        self.M = M
        proto_m = float(getattr(config, "mirage_proto_momentum", 0.95))
        self.proto_m = proto_m
        self.tau = float(getattr(config, "mirage_tau", 0.1))
        self.update_prototypes = bool(getattr(config, "mirage_update_prototypes", True))

        proto = torch.randn(M, proj_dim)
        proto = F.normalize(proto, dim=-1)
        self.register_buffer("prototypes", proto, persistent=True)

        # weights
        self.lambda_rank = float(getattr(config, "mirage_lambda_rank", 0.1))
        self.lambda_align = float(getattr(config, "mirage_lambda_align", 0.05))
        self.pooling = str(getattr(config, "mirage_pooling", "last"))
        self.use_pair_weight = bool(getattr(config, "mirage_use_pair_weight", True))

        self.post_init()

    def _get_beta(self) -> torch.Tensor:
        if self.learn_beta:
            return F.softplus(self.log_beta) + 1e-6
        return self.beta_const

    @torch.no_grad()
    def _ema_update_prototypes(self, u: torch.Tensor, bin_ids: torch.Tensor):
        # u: [N,d] unit vectors; bin_ids: [N] in [0,M-1]
        for m in range(self.M):
            mask = (bin_ids == m)
            if not torch.any(mask):
                continue
            mean_u = u[mask].mean(dim=0)
            mean_u = F.normalize(mean_u, dim=-1)
            self.prototypes[m] = F.normalize(self.proto_m * self.prototypes[m] + (1.0 - self.proto_m) * mean_u, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,                 # [N,L,F]
        labels: Optional[torch.Tensor] = None,    # [N,L,F]
        loss_masks: Optional[torch.Tensor] = None,# [N,L]
        entity_ids: Optional[torch.Tensor] = None,# [N]
        timestamps: Optional[torch.Tensor] = None,# [N]
        **kwargs: Dict[str, Any],
    ) -> MiragePretrainOutput:

        out = self.model(input_ids=input_ids, **kwargs)
        hidden = out.last_hidden_state            # [N,L,H]

        # -----------------
        # Generative loss
        # -----------------
        pred = self.pred_head(hidden)             # [N,L,F]
        loss_gen = None
        if labels is not None:
            per = F.smooth_l1_loss(pred, labels, reduction="none")  # [N,L,F]
            per = per.mean(dim=-1)                                  # [N,L]
            if loss_masks is not None:
                per = per * loss_masks
                denom = loss_masks.sum().clamp_min(1e-6)
            else:
                denom = per.numel()
            loss_gen = per.sum() / denom
        else:
            loss_gen = torch.zeros((), device=hidden.device)

        # -----------------
        # Stage + embedding
        # -----------------
        h = _pool_snapshot(hidden, self.pooling, loss_masks=loss_masks)   # [N,H]
        z = self.stage_head(h).squeeze(-1)                                # [N]
        u = F.normalize(self.proj_head(h), dim=-1)                        # [N,d]

        # -----------------
        # Rank loss
        # -----------------
        loss_rank = torch.zeros((), device=hidden.device)
        beta = self._get_beta()
        if (entity_ids is not None) and (timestamps is not None):
            loss_rank = _pairwise_soft_rank_loss(
                z=z,
                t=timestamps,
                entity_ids=entity_ids,
                beta=beta,
                use_pair_weight=self.use_pair_weight,
            )

        # -----------------
        # Prototype alignment
        # -----------------
        loss_align = torch.zeros((), device=hidden.device)
        bin_ids = None
        if self.M > 1:
            z_det = z.detach()
            bin_ids = _assign_quantile_bins(z_det, self.M)  # [N]
            logits = (u @ self.prototypes.t()) / max(self.tau, 1e-6)  # [N,M]
            loss_align = F.cross_entropy(logits, bin_ids)

            if self.training and self.update_prototypes:
                self._ema_update_prototypes(u.detach(), bin_ids)

        # -----------------
        # Total
        # -----------------
        loss = loss_gen + self.lambda_rank * loss_rank + self.lambda_align * loss_align

        return MiragePretrainOutput(
            loss=loss,
            loss_gen=loss_gen,
            loss_rank=loss_rank,
            loss_align=loss_align,
            z=z,
            u=u,
            bin_ids=bin_ids,
        )
