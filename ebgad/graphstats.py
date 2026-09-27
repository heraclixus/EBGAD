"""Graph statistics behind the default template-bandwidth grid: feature homophily and edge density."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

TEMPLATE_GAMMAS = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]


def compute_graph_stats(data) -> Tuple[float, float]:
    edge_index = data.edge_index.cpu().numpy()
    x = data.x.float()
    n = data.x.shape[0]
    density = edge_index.shape[1] // 2 / max(n, 1)
    n_sample = min(100000, edge_index.shape[1])
    rng = np.random.RandomState(42)
    idx = rng.choice(edge_index.shape[1], n_sample, replace=False)
    cos = torch.nn.functional.cosine_similarity(
        x[edge_index[0, idx]], x[edge_index[1, idx]], dim=1,
    )
    return float(cos.mean()), float(density)


def density_adjusted_template_gammas(h: float, density: float) -> List[float]:
    density_factor = np.sqrt(max(density, 10.0) / 10.0)
    gamma_center = np.clip((0.5 + h) / density_factor, 0.05, 10.0)
    max_gamma = gamma_center * 2.0
    candidates = [g for g in TEMPLATE_GAMMAS if g <= max_gamma]
    if not candidates:
        return [TEMPLATE_GAMMAS[0]]
    if len(candidates) > 3:
        candidates.sort(key=lambda g: abs(np.log(g) - np.log(gamma_center)))
        candidates = sorted(candidates[:3])
    return candidates
