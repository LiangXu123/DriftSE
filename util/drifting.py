"""
Drifting field computation (Algorithm 2 from the paper, Sec A.1).
Implements the core V computation for training drifting models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def cosine_cdist(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes pairwise cosine distance (1 - cosine_similarity) between two sets of vectors.
    Useful for sequence-aware chunk-wise trajectory matching.
    """
    x1_norm = x1 / x1.norm(dim=1, keepdim=True).clamp(min=eps)
    x2_norm = x2 / x2.norm(dim=1, keepdim=True).clamp(min=eps)
    sim = torch.mm(x1_norm, x2_norm.t())
    return (1.0 - sim).clamp(min=0.0)


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True,
    return_components: bool = False,
    chunksize: int = 1,
) -> torch.Tensor:
    """
    Compute the drifting field V (Algorithm 2 from paper, Page 12).

    This is the EXACT implementation from the paper's pseudocode.

    Args:
        x: Generated samples in feature space, shape (N, D)
        y_pos: Positive (real data) samples, shape (N_pos, D)
        y_neg: Negative (generated) samples, shape (N_neg, D)
        temperature: Temperature for softmax (smaller = sharper)
        mask_self: Whether to mask self-distances (when y_neg == x)

    Returns:
        V: Drifting field, shape (N, D)
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device

    # 1. Compute pairwise distances
    if chunksize > 1:
        dist_pos = cosine_cdist(x, y_pos)  # (N, N_pos)
        dist_neg = cosine_cdist(x, y_neg)  # (N, N_neg)
    else:
        dist_pos = torch.cdist(x, y_pos, p=2)  # (N, N_pos)
        dist_neg = torch.cdist(x, y_neg, p=2)  # (N, N_neg)

    # 2. Mask self-distances (when y_neg contains x)
    if mask_self:
        if N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask
        else:
            dist_neg = dist_neg.masked_fill(dist_neg < 1e-5, 1e6)

    # 3. Compute logits
    logit_pos = -dist_pos / temperature  # (N, N_pos)
    logit_neg = -dist_neg / temperature  # (N, N_neg)

    # 4. Concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

    # 5. Normalize along BOTH dimensions (key insight from paper)
    A_row = torch.softmax(logit, dim=1)   # softmax over y (columns)
    A_col = torch.softmax(logit, dim=0)   # softmax over x (rows)
    A = torch.sqrt(A_row * A_col + 1e-12)         # geometric mean

    # 6. Split back to pos and neg
    A_pos = A[:, :N_pos]  # (N, N_pos)
    A_neg = A[:, N_pos:]  # (N, N_neg)

    # 7. Compute weights (cross-weighting from paper)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)

    # 8. Compute drift
    drift_pos = torch.mm(W_pos, y_pos)  # (N, D)
    drift_neg = torch.mm(W_neg, y_neg)  # (N, D)
    
    V = drift_pos - drift_neg

    if return_components:
        return V, drift_pos, drift_neg
    
    return V


def compute_V_paired(
    x: torch.Tensor,      # (N, D) - Current generated/noisy frames
    y_pos: torch.Tensor,  # (N, D) - EXACT clean counterparts (must be same shape as x)
    y_neg: torch.Tensor,  # (N_neg, D) - Other generated samples (negatives)
    temperature: float,
    mask_self: bool = True,
    return_components: bool = False,
    chunksize: int = 1,
) -> torch.Tensor:
    """
    Hybrid Drift Field: 1-to-1 Attraction + Batch Repulsion.
    Solves temporal blurring in speech dereverberation.
    """
    drift_pos = y_pos  # (N, D)
    drift_neg = x      # (N, D)

    V = drift_pos - drift_neg

    if return_components:
        return V, drift_pos, drift_neg
        
    return V