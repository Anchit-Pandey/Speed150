"""
losses.py — Loss functions and metrics for CS3T-UNet
Paper: INFOCOM 2024 — Kang et al.

[PAPER §IV-A]:
  Loss function: MSE
  Metric: NMSE = 10 log10 E[||H - H_hat||²_F / ||H||²_F]   Eq.(10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# NMSE METRIC  [PAPER Eq. (10)]
# NMSE = 10 log10( ||H - H_hat||²_F / ||H||²_F )
# ─────────────────────────────────────────────
def nmse_loss(pred: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-10) -> torch.Tensor:
    """
    Computes mean NMSE in linear scale (not dB).
    pred, target: (B, 2L, Nf, Nt)
    Returns: scalar (mean over batch)
    """
    # Flatten spatial and channel dims per sample
    diff  = (pred - target).reshape(pred.shape[0], -1)       # (B, *)
    ref   = target.reshape(target.shape[0], -1)               # (B, *)
    nmse  = (diff.norm(dim=1) ** 2) / (ref.norm(dim=1) ** 2 + eps)
    return nmse.mean()


def nmse_db(pred: torch.Tensor, target: torch.Tensor,
            eps: float = 1e-10) -> float:
    """Returns NMSE in dB (detached scalar for logging)."""
    with torch.no_grad():
        val = nmse_loss(pred, target, eps)
        return 10.0 * torch.log10(val + eps).item()


# ─────────────────────────────────────────────
# MSE LOSS  [PAPER §IV-A training objective]
# ─────────────────────────────────────────────
class MSELoss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


# ─────────────────────────────────────────────
# COMPOSITE LOSS  (MSE + weighted NMSE)
# [PAPER]: trains with MSE; we optionally add NMSE term
# ─────────────────────────────────────────────
class CompositeLoss(nn.Module):
    def __init__(self, nmse_weight: float = 0.0):
        """
        nmse_weight=0.0 → pure MSE (paper default)
        nmse_weight>0   → MSE + nmse_weight * NMSE_linear
        """
        super().__init__()
        self.nmse_w = nmse_weight

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        if self.nmse_w > 0:
            return mse + self.nmse_w * nmse_loss(pred, target)
        return mse


# ─────────────────────────────────────────────
# MAE (optional additional metric)
# ─────────────────────────────────────────────
def mae_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return F.l1_loss(pred, target).item()
