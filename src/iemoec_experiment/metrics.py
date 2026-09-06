from __future__ import annotations

import numpy as np
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


_REFERENCE_DATA_CACHE: dict[
    tuple,
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}

METRIC_SCHEMA_VERSION = 4


def _deterministic_reference_directions(
    n_objectives: int,
    n_points: int,
) -> np.ndarray:
    """生成精确数量、低内存且跨进程一致的参考方向。"""
    return np.random.default_rng(1).dirichlet(
        np.ones(n_objectives),
        size=n_points,
    )


def _downsample_front(front: np.ndarray, n_points: int) -> np.ndarray:
    """按稳定字典序均匀抽取，避免问题实现返回远超请求数量的点。"""
    values = np.asarray(front, dtype=float)
    if len(values) <= n_points:
        return values
    order = np.lexsort(values.T[::-1])
    positions = np.linspace(0, len(order) - 1, n_points, dtype=int)
    return values[order[positions]]


def _wfg_reference_front(problem, n_points: int) -> np.ndarray:
    """从 pymoo 的 WFG Pareto set 生成有界、可复现的参考前沿。"""
    multiplier = 4 if problem.__class__.__name__.lower() == "wfg2" else 1
    sample_count = n_points * multiplier
    positions = problem._rand_optimal_position(
        sample_count,
        random_state=np.random.default_rng(1),
    )
    pareto_set = problem._positional_to_optimal(positions)
    front = np.asarray(
        problem.evaluate(pareto_set, return_values_of=["F"]),
        dtype=float,
    )
    if multiplier > 1:
        indices = NonDominatedSorting(
            method="efficient_non_dominated_sort"
        ).do(front, only_non_dominated_front=True)
        front = front[indices]
    return _downsample_front(front, n_points)


def _make_reference_front(
    problem,
    n_points: int,
    ref_dirs: np.ndarray,
) -> np.ndarray:
    module = problem.__class__.__module__.lower()
    name = problem.__class__.__name__.lower()
    if "wfg" in module:
        return _wfg_reference_front(problem, n_points)
    if name == "dtlz7":
        # pymoo 的 M=3 路径依赖可下载数据文件，M>3 路径又不接受
        # ref_dirs。这里复用其解析式和固定随机种子，避免网络与 API 分支。
        rng = np.random.default_rng(42)
        first = rng.random((n_points * 20, problem.n_obj - 1))
        last = 2 * problem.n_obj - np.sum(
            first * (1 + np.sin(3 * np.pi * first)),
            axis=1,
        )
        front = np.column_stack([first, last])
        return _downsample_front(front[last >= 0], n_points)
    return np.asarray(problem.pareto_front(ref_dirs=ref_dirs), dtype=float)


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
    ref_dirs = _deterministic_reference_directions(problem.n_obj, n_points)
    pf = _make_reference_front(problem, n_points, ref_dirs)
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
        problem_module = problem.__class__.__module__.lower()
        problem_name = problem.__class__.__name__.lower()
        if "wfg" in problem_module:
            self.reference_front_method = "pymoo_pareto_set_seed_1"
        elif problem_name == "dtlz7":
            self.reference_front_method = "pymoo_dtlz7_formula_seed_42"
        else:
            self.reference_front_method = "pymoo_pf_dirichlet_seed_1"
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

    @staticmethod
    def _weighted_batches(
        values: np.ndarray,
        calculate,
        batch_size: int = 128,
    ) -> float:
        total = 0.0
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            total += float(calculate(batch)) * len(batch)
        return total / len(values)

    def _calculate_igd_plus(self, F: np.ndarray) -> float:
        return self._weighted_batches(
            self.ref_pf,
            lambda reference_batch: IGDPlus(reference_batch)(F),
        )

    def _calculate_gd_plus(self, F: np.ndarray) -> float:
        return self._weighted_batches(F, self.gd_plus)

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
            "igd_plus": self._calculate_igd_plus(nd),
            "gd_plus": self._calculate_gd_plus(nd),
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
