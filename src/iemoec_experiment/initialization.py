from __future__ import annotations

import hashlib

import numpy as np
from pymoo.operators.sampling.rnd import FloatRandomSampling


_INITIAL_X_CACHE: dict[tuple, np.ndarray] = {}


def shared_initial_decisions(problem, population_size: int, seed: int) -> np.ndarray:
    """返回算法间共享且可复现的初始决策向量。"""
    key = (
        problem.__class__.__module__,
        problem.__class__.__name__,
        problem.n_var,
        problem.n_obj,
        population_size,
        seed,
    )
    if key not in _INITIAL_X_CACHE:
        population = FloatRandomSampling().do(
            problem,
            population_size,
            random_state=np.random.default_rng(seed),
        )
        _INITIAL_X_CACHE[key] = np.asarray(population.get("X"), dtype=float)
    return _INITIAL_X_CACHE[key].copy()


def initialization_hash(X: np.ndarray) -> str:
    """生成便于审计公共初始化一致性的稳定摘要。"""
    values = np.ascontiguousarray(X, dtype=np.float64)
    return hashlib.sha256(values.tobytes()).hexdigest()
