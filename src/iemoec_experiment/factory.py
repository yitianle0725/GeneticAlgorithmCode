from __future__ import annotations

import numpy as np

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from .config import ExperimentCase, default_ref_partitions


def reference_directions(case: ExperimentCase):
    partitions = case.ref_partitions or default_ref_partitions(case.n_obj)
    return get_reference_directions("das-dennis", case.n_obj, n_partitions=partitions)


def make_operators(initial_X=None):
    return {
        "sampling": initial_X.copy() if initial_X is not None else FloatRandomSampling(),
        "crossover": SBX(prob=1.0, eta=30),
        "mutation": PM(prob=1.0, prob_var=None, eta=20),
    }


def make_baseline(case: ExperimentCase, initial_X=None):
    """构造共享种群规模、参考方向和变异参数的 pymoo baseline。"""
    algorithm = case.normalized_algorithm
    if algorithm == "IEMOEC":
        raise ValueError("IEMOEC 由自定义 runner 创建")
    ref_dirs = reference_directions(case)
    pop_size = len(ref_dirs)
    operators = make_operators(initial_X)
    if algorithm == "NSGA2":
        return NSGA2(pop_size=pop_size, **operators), pop_size, ref_dirs
    if algorithm == "NSGA3":
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, **operators), pop_size, ref_dirs
    if algorithm == "MOEAD":
        return (
            MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=min(20, pop_size),
                prob_neighbor_mating=0.9,
                decomposition=Tchebicheff(),
                **operators,
            ),
            pop_size,
            ref_dirs,
        )
    if algorithm == "MOEADPBI":
        return (
            MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=min(20, pop_size),
                prob_neighbor_mating=0.9,
                decomposition=PBI(theta=5.0),
                **operators,
            ),
            pop_size,
            ref_dirs,
        )
    if algorithm == "RVEA":
        from pymoo.algorithms.moo.rvea import RVEA

        return (
            RVEA(ref_dirs=ref_dirs, pop_size=pop_size, **operators),
            pop_size,
            ref_dirs,
        )
    if algorithm == "AGEMOEA2":
        try:
            from pymoo.algorithms.moo.age2 import AGEMOEA2
        except Exception as exc:
            raise RuntimeError(
                "AGE-MOEA2 需要可选依赖 numba；请先执行 pip install numba"
            ) from exc
        return AGEMOEA2(pop_size=pop_size, **operators), pop_size, ref_dirs
    if algorithm == "AGEMOEA2STABLE":
        try:
            from pymoo.algorithms.moo.age2 import AGEMOEA2, AGEMOEA2Survival
        except Exception as exc:
            raise RuntimeError(
                "AGE-MOEA2 需要可选依赖 numba；请先执行 pip install numba"
            ) from exc

        class StableAGEMOEA2Survival(AGEMOEA2Survival):
            @staticmethod
            def pairwise_distances(front, p):
                values = np.asarray(front, dtype=float)
                norms = np.sum(np.maximum(values, 0.0) ** p, axis=1) ** (1.0 / p)
                if np.all(norms > 0.0):
                    return AGEMOEA2Survival.pairwise_distances(values, p)

                projected = np.zeros_like(values)
                valid = norms > 0.0
                projected[valid] = values[valid] / norms[valid, None]
                distances = np.zeros((len(values), len(values)), dtype=float)
                for row in range(len(values) - 1):
                    for column in range(row + 1, len(values)):
                        if 0.95 < p < 1.05:
                            distance = np.linalg.norm(
                                projected[row] - projected[column]
                            )
                        else:
                            midpoint = 0.5 * (
                                projected[row] + projected[column]
                            )
                            midpoint_norm = (
                                np.sum(np.maximum(midpoint, 0.0) ** p)
                                ** (1.0 / p)
                            )
                            if midpoint_norm > 0.0:
                                midpoint = midpoint / midpoint_norm
                            distance = (
                                np.linalg.norm(projected[row] - midpoint)
                                + np.linalg.norm(projected[column] - midpoint)
                            )
                        distances[row, column] = distance
                        distances[column, row] = distance
                return distances

        stable = AGEMOEA2(pop_size=pop_size, **operators)
        stable.survival = StableAGEMOEA2Survival()
        return stable, pop_size, ref_dirs
    raise ValueError(f"未知 baseline: {algorithm}")
