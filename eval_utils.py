"""
Shared evaluation utilities for GAD pipelines.

Handles datasets with unlabeled nodes (y=-1) by evaluating only on labeled nodes (y in {0,1}).
"""

import torch
import numpy as np


def mask_labeled(y, scores):
    """Filter to labeled nodes (y in {0,1}) for evaluation.

    Returns (y_binary, scores_masked) or (None, None) if no labeled nodes.
    y_binary is 0/1 (0=normal, 1=anomaly).
    """
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores).float()
    y = y.long()
    mask = (y >= 0) & (y <= 1)
    n_labeled = mask.sum().item()
    if n_labeled == 0:
        return None, None
    y_binary = (y[mask] > 0).long()
    scores_masked = scores[mask]
    return y_binary, scores_masked
