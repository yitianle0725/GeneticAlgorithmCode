from __future__ import annotations

import numpy as np
from pymoo.core.population import Population


FEASIBILITY_TOLERANCE = 1e-12


def total_constraint_violation(values, population_size: int) -> np.ndarray:
    """把 pymoo 的 CV 数据统一成一维非负总违反量。"""
    if population_size == 0:
        return np.empty(0, dtype=float)
    if values is None:
        return np.zeros(population_size, dtype=float)
    return np.maximum(
        np.asarray(values, dtype=float).reshape(population_size, -1).sum(axis=1),
        0.0,
    )


def constraint_violation(population: Population) -> np.ndarray:
    """返回每个个体的一维非负总约束违反量。"""
    return total_constraint_violation(population.get("CV"), len(population))


def feasible_mask(population: Population) -> np.ndarray:
    """使用统一容差判断种群可行性。"""
    return constraint_violation(population) <= FEASIBILITY_TOLERANCE


def feasibility_first_order(
    population: Population,
    feasible_scores: np.ndarray,
) -> np.ndarray:
    """Deb feasibility-first 稳定排序。

    可行解按调用方提供的目标分数排序；不可行解只按总约束违反量排序。
    最后的原始下标确保完全相同的键保持输入顺序。
    """
    scores = np.asarray(feasible_scores, dtype=float)
    if len(scores) != len(population):
        raise ValueError("feasible_scores 长度必须与种群一致")
    violation = constraint_violation(population)
    feasible = violation <= FEASIBILITY_TOLERANCE
    primary = (~feasible).astype(int)
    secondary = np.where(feasible, scores, violation)
    original_order = np.arange(len(population))
    return np.lexsort((original_order, secondary, primary))


def best_quality_key(
    population: Population,
    feasible_scores: np.ndarray,
) -> tuple[int, float]:
    """返回可直接进行字典序比较的最佳 feasibility-first 质量键。"""
    order = feasibility_first_order(population, feasible_scores)
    best = int(order[0])
    violation = constraint_violation(population)[best]
    if violation <= FEASIBILITY_TOLERANCE:
        return 0, float(np.asarray(feasible_scores, dtype=float)[best])
    return 1, float(violation)


def constraint_statistics(population: Population) -> dict[str, float | int | bool]:
    """生成不依赖目标指标的约束诊断。"""
    violation = constraint_violation(population)
    feasible = violation <= FEASIBILITY_TOLERANCE
    return {
        "has_feasible": bool(np.any(feasible)),
        "feasible_count": int(np.sum(feasible)),
        "feasible_ratio": float(np.mean(feasible)) if len(feasible) else 0.0,
        "min_cv": float(np.min(violation)) if len(violation) else 0.0,
        "mean_cv": float(np.mean(violation)) if len(violation) else 0.0,
    }


def feasible_population(population: Population) -> Population:
    """返回可行子种群；没有可行解时返回空种群。"""
    return population[np.flatnonzero(feasible_mask(population))]
