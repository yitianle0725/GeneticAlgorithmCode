from __future__ import annotations

import numpy as np
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


_REFERENCE_DATA_CACHE: dict[
    tuple,
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}

METRIC_SCHEMA_VERSION = 3


def reference_data(
    problem,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回进程内共享的参考 PF、ideal point 和 nadir point。"""
    cache_key = (
        problem.__class__.__module__, problem.__class__.__name__,
        problem.n_var, problem.n_obj, n_points,
    )
    if cache_key in _REFERENCE_DATA_CACHE:
        return _REFERENCE_DATA_CACHE[cache_key]
    if problem.n_obj <= 5:
        ref_dirs = get_reference_directions(
            "energy", problem.n_obj, n_points=n_points, seed=1
        )
    else:
        # 高维 energy 优化本身很昂贵；固定 Dirichlet 方向可复现且对所有算法共享。
        ref_dirs = np.random.default_rng(1).dirichlet(
            np.ones(problem.n_obj), size=n_points
        )
    pf = problem.pareto_front(ref_dirs=ref_dirs)
    if pf is None or len(pf) == 0:
        raise ValueError(f"{problem.__class__.__name__} 无法生成参考 Pareto front")
    ref_pf = np.asarray(pf, dtype=float)
    ideal = np.min(ref_pf, axis=0)
    nadir = np.max(ref_pf, axis=0)
    data = (ref_pf, ideal, nadir)
    _REFERENCE_DATA_CACHE[cache_key] = data
    return data


def normalize(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    return (np.asarray(F, dtype=float) - ideal) / np.maximum(nadir - ideal, 1e-12)


def nondominated(F: np.ndarray) -> np.ndarray:
    if len(F) == 0:
        return np.empty((0, F.shape[1] if F.ndim == 2 else 0))
    indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return np.asarray(F)[indices]


class MetricSuite:
    def __init__(
        self,
        problem,
        n_reference_points: int = 1000,
        hv_samples: int = 20000,
        direction_directions: np.ndarray | None = None,
    ):
        self.ref_pf, self.ideal, self.nadir = reference_data(
            problem,
            n_reference_points,
        )
        self.igd_plus = IGDPlus(self.ref_pf)
        self.gd_plus = GDPlus(self.ref_pf)
        self.spacing = SpacingIndicator()
        self.direction_directions = (
            None
            if direction_directions is None
            else np.asarray(direction_directions, dtype=float)
        )
        self.hv_method = "exact" if problem.n_obj <= 5 else "monte_carlo"
        self.hv = HV(ref_point=np.full(problem.n_obj, 1.1)) if problem.n_obj <= 5 else None
        self.hv_ref = 1.1
        self.hv_samples = np.random.default_rng(20260903).uniform(
            0.0, self.hv_ref, size=(hv_samples, problem.n_obj)
        ) if problem.n_obj > 5 else None

    def _calculate_hv(self, normalized_nd: np.ndarray) -> float:
        points = np.maximum(normalized_nd, 0.0)
        if self.hv is not None:
            return float(self.hv(points))
        dominated_count = 0
        assert self.hv_samples is not None
        for start in range(0, len(self.hv_samples), 1000):
            samples = self.hv_samples[start:start + 1000]
            dominated = np.any(
                np.all(points[:, None, :] <= samples[None, :, :], axis=2), axis=0
            )
            dominated_count += int(np.sum(dominated))
        box_volume = self.hv_ref ** points.shape[1]
        return float(box_volume * dominated_count / len(self.hv_samples))

    def calculate(self, F: np.ndarray, include_hv: bool = True) -> dict[str, float | int]:
        F = np.asarray(F, dtype=float)
        nd = nondominated(F)
        normalized_nd = normalize(nd, self.ideal, self.nadir)
        result: dict[str, float | int] = {
            "igd_plus": float(self.igd_plus(nd)),
            "gd_plus": float(self.gd_plus(nd)),
            "spacing": (
                float(self.spacing(normalized_nd)) if len(normalized_nd) >= 2 else 0.0
            ),
            "onvg": int(len(nd)),
            "nd_ratio": float(len(nd) / max(1, len(F))),
        }
        if self.direction_directions is not None:
            points = np.maximum(normalized_nd, 0.0)
            points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
            directions = self.direction_directions / np.maximum(
                np.linalg.norm(self.direction_directions, axis=1, keepdims=True),
                1e-12,
            )
            assigned = np.argmax(points @ directions.T, axis=1)
            result["direction_occupancy"] = float(
                len(np.unique(assigned)) / len(directions)
            )
        if include_hv:
            result["hv"] = self._calculate_hv(normalized_nd)
        return result
