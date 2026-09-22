from __future__ import annotations

from pathlib import Path

import numpy as np
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .constraints import FEASIBILITY_TOLERANCE, total_constraint_violation
from .problems import is_constrained_problem_name


_REFERENCE_DATA_CACHE: dict[
    tuple,
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}

METRIC_SCHEMA_VERSION = 5
CONSTRAINED_METRIC_SCHEMA_VERSION = 6


def metric_schema_version_for_problem(problem_name: str) -> int:
    """约束指标独立使用 schema 6，旧 formal 继续保持 schema 5。"""
    if is_constrained_problem_name(problem_name):
        return CONSTRAINED_METRIC_SCHEMA_VERSION
    return METRIC_SCHEMA_VERSION


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
    problem_id = getattr(problem, "_iemoec_problem_id", name)
    if problem_id.startswith("dascmop"):
        path = (
            Path(__file__).resolve().parents[2]
            / "references"
            / "dascmop"
            / f"{problem_id}.pf"
        )
        if not path.exists():
            raise ValueError(f"缺少 DAS-CMOP 参考前沿文件: {path}")
        return _downsample_front(np.loadtxt(path), n_points)
    if problem_id == "c2dtlz2":
        # C2-DTLZ2 的可行 PF 是多个分离区域。高维时直接使用 n_points
        # 个 Dirichlet 方向会留下过少可行点，因此先确定性过采样再筛选。
        sample_count = max(n_points, n_points * 4 * problem.n_obj)
        random_count = max(0, sample_count - problem.n_obj)
        constrained_directions = np.vstack([
            np.eye(problem.n_obj),
            _deterministic_reference_directions(
                problem.n_obj,
                random_count,
            ),
        ])
        front = problem.pareto_front(ref_dirs=constrained_directions)
        if len(front) <= n_points:
            return front
        axis_front = front[:problem.n_obj]
        remaining = _downsample_front(
            front[problem.n_obj:],
            n_points - len(axis_front),
        )
        return np.vstack([axis_front, remaining])
    if problem_id in ("c1dtlz1", "c1dtlz3", "c3dtlz4"):
        # 显式加入坐标轴，确保 constrained schema 的 ideal/nadir 不因
        # 随机方向没有采到边界而系统性偏移。
        random_count = max(0, n_points - problem.n_obj)
        constrained_directions = np.vstack([
            np.eye(problem.n_obj),
            _deterministic_reference_directions(
                problem.n_obj,
                random_count,
            ),
        ])
        return np.asarray(
            problem.pareto_front(ref_dirs=constrained_directions),
            dtype=float,
        )
    if "wfg" in module:
        return _wfg_reference_front(problem, n_points)
    if name in ("dtlz5", "dtlz6"):
        # DTLZ5/6 共用同一条退化 Pareto 曲线。pymoo 的 M=3 分支
        # 依赖外部数据文件，因此直接复用其解析式以保持离线可运行。
        theta_1 = np.linspace(0.0, np.pi / 2.0, n_points)
        theta = np.column_stack(
            [theta_1]
            + [np.full(n_points, np.pi / 4.0)] * (problem.n_obj - 2)
        )
        cosine = np.cos(theta)
        sine = np.sin(theta)
        front = np.zeros((n_points, problem.n_obj))
        for objective in range(problem.n_obj):
            front[:, objective] = np.prod(
                cosine[:, : problem.n_obj - 1 - objective],
                axis=1,
            )
            if objective > 0:
                front[:, objective] *= sine[:, problem.n_obj - 1 - objective]
        return front
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
        getattr(problem, "_iemoec_problem_id", None),
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
        hv_reference_point: float = 1.1,
    ):
        problem_module = problem.__class__.__module__.lower()
        problem_name = problem.__class__.__name__.lower()
        problem_id = getattr(problem, "_iemoec_problem_id", problem_name)
        self.constrained = bool(problem.has_constraints())
        if problem_id.startswith("dascmop"):
            self.reference_front_method = "bundled_pymoo_data_pf"
        elif "wfg" in problem_module:
            self.reference_front_method = "pymoo_pareto_set_seed_1"
        elif problem_id == "c2dtlz2":
            self.reference_front_method = (
                "pymoo_constrained_pf_oversampled_dirichlet_axes_seed_1"
            )
        elif problem_id in ("c1dtlz1", "c1dtlz3", "c3dtlz4"):
            self.reference_front_method = (
                "pymoo_constrained_pf_dirichlet_axes_seed_1"
            )
        elif problem_name in ("dtlz5", "dtlz6"):
            self.reference_front_method = "pymoo_dtlz5_dtlz6_formula"
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
        if hv_reference_point <= 0:
            raise ValueError("HV reference point must be positive")
        self.hv_method = "exact" if problem.n_obj <= 5 else "monte_carlo"
        self.hv_ref = float(hv_reference_point)
        self.hv = (
            HV(ref_point=np.full(problem.n_obj, self.hv_ref))
            if problem.n_obj <= 5
            else None
        )
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

    def _hv_values(self, normalized_nd: np.ndarray) -> dict[str, float | int]:
        return {
            "hv_reference_point": self.hv_ref,
            "hv_eligible_solution_count": int(
                np.sum(np.all(np.maximum(normalized_nd, 0.0) < self.hv_ref, axis=1))
            ),
            "hv": self._calculate_hv(normalized_nd),
        }

    def calculate_hv(
        self,
        F: np.ndarray,
        CV: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        values = np.asarray(F, dtype=float)
        if self.constrained:
            if CV is None:
                raise ValueError("约束问题计算 HV 时必须提供 CV")
            violation = total_constraint_violation(CV, len(values))
            values = values[violation <= FEASIBILITY_TOLERANCE]
            if not len(values):
                return {
                    "hv_reference_point": self.hv_ref,
                    "hv_eligible_solution_count": 0,
                    "hv": 0.0,
                }
        nd = nondominated(values)
        normalized_nd = normalize(nd, self.ideal, self.nadir)
        return self._hv_values(normalized_nd)

    def calculate(
        self,
        F: np.ndarray,
        include_hv: bool = True,
        CV: np.ndarray | None = None,
    ) -> dict[str, float | int | bool | None]:
        F = np.asarray(F, dtype=float)
        constraint_result: dict[str, float | int | bool] = {}
        if self.constrained:
            if CV is None:
                raise ValueError("约束问题计算指标时必须提供 CV")
            violation = total_constraint_violation(CV, len(F))
            feasible = violation <= FEASIBILITY_TOLERANCE
            constraint_result = {
                "has_feasible": bool(np.any(feasible)),
                "feasible_count": int(np.sum(feasible)),
                "feasible_ratio": float(np.mean(feasible)) if len(feasible) else 0.0,
                "min_cv": float(np.min(violation)) if len(violation) else 0.0,
                "mean_cv": float(np.mean(violation)) if len(violation) else 0.0,
            }
            F = F[feasible]
            if not len(F):
                result: dict[str, float | int | bool | None] = {
                    "igd_plus": None,
                    "gd_plus": None,
                    "spacing": None,
                    "onvg": 0,
                    "nd_ratio": 0.0,
                    "direction_occupancy": 0.0,
                    "feasible_direction_coverage": 0.0,
                    **constraint_result,
                }
                if include_hv:
                    result.update({
                        "hv_reference_point": self.hv_ref,
                        "hv_eligible_solution_count": 0,
                        "hv": 0.0,
                    })
                return result
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
            if self.constrained:
                result["feasible_direction_coverage"] = result[
                    "direction_occupancy"
                ]
        if include_hv:
            result.update(self._hv_values(normalized_nd))
        result.update(constraint_result)
        return result

    def calculate_population(
        self,
        population,
        include_hv: bool = True,
    ) -> dict[str, float | int | bool | None]:
        """直接从 Population 计算指标，约束问题自动读取 CV。"""
        CV = population.get("CV") if self.constrained else None
        return self.calculate(
            population.get("F"),
            include_hv=include_hv,
            CV=CV,
        )
