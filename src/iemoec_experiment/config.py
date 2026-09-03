from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_ALGORITHMS = ("NSGA2", "NSGA3", "MOEAD", "IEMOEC")
SUPPORTED_OBJECTIVES = (2, 3, 5, 8, 10, 15)


@dataclass(frozen=True)
class IEMOECConfig:
    """IEMOEC 的可审计配置；默认值刻意限制重组和岛内计算量。"""

    origin_ratio: float = 0.2
    min_origin: int = 20
    island_population: int = 20
    islands_per_objective: int = 2
    inner_generations_early: int = 3
    inner_generations_late: int = 5
    switch_ratio: float = 0.4
    aggregation_epsilon: float = 1e-3
    expansion_mutation_probability: float = 0.15
    partners_per_elite: int = 2
    recombination_budget_ratio: float = 1.0
    outer_survival: str = "nsga3"
    use_crowding: bool = False
    enable_recombination: bool = True

    def validate(self) -> None:
        if not 0 < self.origin_ratio <= 1:
            raise ValueError("origin_ratio 必须在 (0, 1] 内")
        if self.min_origin < 2 or self.island_population < 2:
            raise ValueError("种群大小必须至少为 2")
        if self.islands_per_objective not in (1, 2):
            raise ValueError("islands_per_objective 仅支持 1 或 2")
        if self.outer_survival not in ("nsga3", "rank"):
            raise ValueError("outer_survival 仅支持 nsga3 或 rank")
        if self.partners_per_elite < 1:
            raise ValueError("partners_per_elite 必须为正整数")


@dataclass(frozen=True)
class ExperimentCase:
    algorithm: str
    problem: str
    n_obj: int
    seed: int
    max_fes: int
    output_root: str = "results/default"
    ref_partitions: int | None = None
    n_var: int | None = None
    history_points: int = 20
    history_hv: bool = False
    reference_points: int = 1000
    high_dim_hv_samples: int = 20000
    iemoec: IEMOECConfig = field(default_factory=IEMOECConfig)

    def validate(self) -> None:
        if self.algorithm.upper() not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"不支持算法 {self.algorithm}; 可选 {SUPPORTED_ALGORITHMS}")
        if self.n_obj not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"不支持目标数 {self.n_obj}; 可选 {SUPPORTED_OBJECTIVES}")
        if self.seed < 0 or self.max_fes <= 0:
            raise ValueError("seed 必须非负，max_fes 必须为正整数")
        if self.history_points < 1 or self.reference_points < 10 or self.high_dim_hv_samples < 1000:
            raise ValueError("history_points >= 1、reference_points >= 10 且 high_dim_hv_samples >= 1000")
        self.iemoec.validate()

    @property
    def normalized_algorithm(self) -> str:
        return self.algorithm.upper()

    @property
    def normalized_problem(self) -> str:
        return self.problem.lower().replace("-", "")

    @property
    def output_dir(self) -> Path:
        return (
            Path(self.output_root)
            / self.normalized_problem.upper()
            / f"M{self.n_obj}"
            / self.normalized_algorithm
            / f"seed_{self.seed:03d}"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["algorithm"] = self.normalized_algorithm
        data["problem"] = self.normalized_problem
        return data


def default_ref_partitions(n_obj: int) -> int:
    mapping = {2: 99, 3: 12, 5: 6, 8: 3, 10: 3, 15: 2}
    try:
        return mapping[n_obj]
    except KeyError as exc:
        raise ValueError(f"没有 M={n_obj} 的参考方向划分") from exc
