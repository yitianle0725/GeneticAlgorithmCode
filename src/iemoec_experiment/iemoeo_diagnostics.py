"""IE-MOEO 环境选择的辅助诊断，不消耗 FE，也不使用算法自身的随机流。"""
from __future__ import annotations

import numpy as np

from .normalization import ObjectiveNormalization


def selection_pair_audit(problem, candidates, selected, seed: int) -> dict:
    """固定阈值描述候选保留，不把某对未保留自动认定为选择错误。"""
    count = len(candidates)
    if count < 2:
        left = right = np.array([], dtype=int)
    elif count * (count - 1) // 2 <= 1024:
        left, right = np.triu_indices(count, k=1)
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(count, size=1024)
        right = rng.integers(count - 1, size=1024)
        right += right >= left
    X = (candidates.get("X") - problem.xl) / (problem.xu - problem.xl)
    F = ObjectiveNormalization.from_objectives(candidates.get("F")).apply(candidates.get("F"))
    x_distance = np.sqrt(np.mean((X[left] - X[right]) ** 2, axis=1))
    f_distance = np.sqrt(np.mean((F[left] - F[right]) ** 2, axis=1))
    relevant = (f_distance <= 0.02) & (x_distance >= 0.2)
    selected_keys = {np.asarray(ind.X, dtype=np.float64).tobytes() for ind in selected}
    retained = np.asarray([np.asarray(ind.X, dtype=np.float64).tobytes() in selected_keys for ind in candidates])
    return {
        "audited_pair_count": len(left),
        "near_f_far_x_pair_count": int(np.sum(relevant)),
        "near_f_far_x_both_retained": int(np.sum(relevant & retained[left] & retained[right])),
        "near_f_far_x_endpoints_retained": int(np.sum(retained[left[relevant]])) + int(np.sum(retained[right[relevant]])),
    }
