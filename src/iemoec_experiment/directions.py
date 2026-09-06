from __future__ import annotations

import numpy as np


def reference_direction_subset(
    reference_directions: np.ndarray,
    n_objectives: int,
    island_count: int,
) -> np.ndarray:
    """以轴向覆盖为起点，用 max-min 角距离选择稳定方向子集。"""
    directions = np.asarray(reference_directions, dtype=float)
    if island_count < n_objectives:
        raise ValueError("island_count 不能小于目标数")
    if island_count > len(directions):
        raise ValueError("island_count 不能超过公共参考方向数")

    unit = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True),
        1e-12,
    )
    selected: list[int] = []
    for objective in range(n_objectives):
        axis_index = int(np.argmax(unit[:, objective]))
        if axis_index not in selected:
            selected.append(axis_index)

    while len(selected) < island_count:
        available = np.setdiff1d(
            np.arange(len(unit)),
            np.asarray(selected, dtype=int),
            assume_unique=False,
        )
        similarities = unit[available] @ unit[selected].T
        nearest_angle = np.min(
            np.arccos(np.clip(similarities, -1.0, 1.0)),
            axis=1,
        )
        selected.append(int(available[int(np.argmax(nearest_angle))]))

    return directions[np.asarray(selected, dtype=int)].copy()


def direction_objective(weight: np.ndarray, epsilon: float = 1e-12) -> int | None:
    """轴向方向返回对应目标，其余方向返回 None。"""
    values = np.asarray(weight, dtype=float)
    objective = int(np.argmax(values))
    mask = np.ones(len(values), dtype=bool)
    mask[objective] = False
    if values[objective] > 0 and np.all(values[mask] <= epsilon):
        return objective
    return None
