from __future__ import annotations

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from .config import ExperimentCase, default_ref_partitions


def reference_directions(case: ExperimentCase):
    partitions = case.ref_partitions or default_ref_partitions(case.n_obj)
    return get_reference_directions("das-dennis", case.n_obj, n_partitions=partitions)


def make_operators():
    return {
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=1.0, eta=30),
        "mutation": PM(prob=1.0, prob_var=None, eta=20),
    }


def make_baseline(case: ExperimentCase):
    """构造共享种群规模、参考方向和变异参数的 pymoo baseline。"""
    algorithm = case.normalized_algorithm
    if algorithm == "IEMOEC":
        raise ValueError("IEMOEC 由自定义 runner 创建")
    ref_dirs = reference_directions(case)
    pop_size = len(ref_dirs)
    operators = make_operators()
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
    raise ValueError(f"未知 baseline: {algorithm}")
