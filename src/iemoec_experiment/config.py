from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_ALGORITHMS = (
    "NSGA2", "NSGA3", "MOEAD", "RVEA", "AGEMOEA2", "IEMOEC"
)
DEFAULT_ALGORITHMS = ("NSGA2", "NSGA3", "MOEAD", "IEMOEC")
SUPPORTED_OBJECTIVES = (2, 3, 5, 8, 10, 15)
ALGORITHM_LABELS = {
    "NSGA2": "NSGA-II",
    "NSGA3": "NSGA-III",
    "MOEAD": "MOEA/D-TCH",
    "RVEA": "RVEA",
    "AGEMOEA2": "AGE-MOEA2",
    "IEMOEC": "IEMOEC",
}

IEMOEC_SCHEMA_VERSIONS = {"v0": 0, "s1": 1, "candidate": 2}
BASELINE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class IEMOECConfig:
    """IEMOEC 的可审计配置；默认值刻意限制重组和岛内计算量。"""

    variant: str = "s1"
    initialization_mode: str = "legacy_origin"
    normalization_mode: str = "legacy"
    origin_ratio: float = 0.2
    min_origin: int = 20
    island_population: int = 20
    islands_per_objective: int = 2
    inner_generations_early: int = 1
    inner_generations_late: int = 1
    switch_ratio: float = 0.4
    aggregation_epsilon: float = 1e-3
    island_initialization: str = "multi_ancestor"
    direction_neighbor_ancestors: int = 4
    diverse_ancestors: int = 2
    island_direction_mode: str = "axis_random"
    island_count_multiplier: int = 2
    expansion_mutation_probability: float = 0.15
    partners_per_elite: int = 2
    recombination_budget_ratio: float = 1.0
    late_recombination_budget_ratio: float | None = 0.25
    outer_survival: str = "nsga3"
    use_crowding: bool = False
    enable_recombination: bool = True
    retain_island_state: bool = False
    fixed_island_definitions: bool = False
    fe_scheduler: str = "legacy"
    outer_batch_ratio: float = 1.0
    local_fe_ratio: float = 0.75
    recombination_fe_ratio: float = 0.25
    pairing_strategy: str = "farthest_weight"

    @classmethod
    def for_variant(cls, variant: str, **overrides) -> IEMOECConfig:
        """构造可复现的 V0、S1 或候选结构配置。"""
        profiles = {
            "v0": {
                "initialization_mode": "legacy_origin",
                "normalization_mode": "legacy",
                "island_initialization": "single_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "fe_scheduler": "legacy",
                "pairing_strategy": "farthest_weight",
            },
            "s1": {
                "initialization_mode": "legacy_origin",
                "normalization_mode": "legacy",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "fe_scheduler": "legacy",
                "pairing_strategy": "farthest_weight",
            },
            "candidate": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "reference_subset",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "pairing_strategy": "farthest_weight",
            },
        }
        if variant not in profiles:
            raise ValueError(f"未知 IEMOEC variant: {variant}")
        values = {**profiles[variant], **overrides, "variant": variant}
        return cls(**values)

    @property
    def algorithm_schema_version(self) -> int:
        return IEMOEC_SCHEMA_VERSIONS[self.variant]

    def validate(self) -> None:
        if self.variant not in IEMOEC_SCHEMA_VERSIONS:
            raise ValueError("variant 仅支持 v0、s1 或 candidate")
        if self.initialization_mode not in ("legacy_origin", "shared_population"):
            raise ValueError(
                "initialization_mode 仅支持 legacy_origin 或 shared_population"
            )
        if self.normalization_mode not in ("legacy", "global"):
            raise ValueError("normalization_mode 仅支持 legacy 或 global")
        if not 0 < self.origin_ratio <= 1:
            raise ValueError("origin_ratio 必须在 (0, 1] 内")
        if self.min_origin < 2 or self.island_population < 2:
            raise ValueError("种群大小必须至少为 2")
        if self.inner_generations_early < 1 or self.inner_generations_late < 1:
            raise ValueError("岛内演化代数必须至少为 1")
        if self.islands_per_objective not in (1, 2):
            raise ValueError("islands_per_objective 仅支持 1 或 2")
        if self.island_initialization not in ("single_ancestor", "multi_ancestor"):
            raise ValueError(
                "island_initialization 仅支持 single_ancestor 或 multi_ancestor"
            )
        if self.direction_neighbor_ancestors < 0 or self.diverse_ancestors < 0:
            raise ValueError("多祖先数量不能为负数")
        if self.island_direction_mode not in ("axis_random", "reference_subset"):
            raise ValueError(
                "island_direction_mode 仅支持 axis_random 或 reference_subset"
            )
        if self.island_count_multiplier not in (2, 4):
            raise ValueError("island_count_multiplier 仅支持 2 或 4")
        if self.outer_survival not in ("nsga3", "rank", "rank_crowding"):
            raise ValueError(
                "outer_survival 仅支持 nsga3、rank 或 rank_crowding"
            )
        if self.partners_per_elite < 1:
            raise ValueError("partners_per_elite 必须为正整数")
        if not 0 <= self.recombination_budget_ratio <= 1:
            raise ValueError("recombination_budget_ratio 必须在 [0, 1] 内")
        if (
            self.late_recombination_budget_ratio is not None
            and not 0 <= self.late_recombination_budget_ratio <= 1
        ):
            raise ValueError("late_recombination_budget_ratio 必须在 [0, 1] 内")
        if self.fe_scheduler not in ("legacy", "fixed_batch"):
            raise ValueError("fe_scheduler 仅支持 legacy 或 fixed_batch")
        if self.outer_batch_ratio <= 0:
            raise ValueError("outer_batch_ratio 必须大于 0")
        if not 0 <= self.local_fe_ratio <= 1:
            raise ValueError("local_fe_ratio 必须在 [0, 1] 内")
        if not 0 <= self.recombination_fe_ratio <= 1:
            raise ValueError("recombination_fe_ratio 必须在 [0, 1] 内")
        if not math.isclose(
            self.local_fe_ratio + self.recombination_fe_ratio,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("local_fe_ratio + recombination_fe_ratio 必须等于 1")
        strategies = {
            "farthest_weight", "nearest_weight", "random",
            "farthest_decision", "none",
        }
        if self.pairing_strategy not in strategies:
            raise ValueError(f"pairing_strategy 必须属于 {sorted(strategies)}")
        if self.variant in ("v0", "s1"):
            expected_initialization = (
                "single_ancestor" if self.variant == "v0" else "multi_ancestor"
            )
            if self.island_initialization != expected_initialization:
                raise ValueError(
                    f"{self.variant} 必须使用 {expected_initialization}；"
                    "请改用对应 variant"
                )
            if (
                self.initialization_mode != "legacy_origin"
                or self.normalization_mode != "legacy"
                or self.island_direction_mode != "axis_random"
                or self.fe_scheduler != "legacy"
                or self.pairing_strategy != "farthest_weight"
            ):
                raise ValueError("v0/s1 的兼容路径不允许启用 candidate 机制")
        if self.variant == "candidate" and self.retain_island_state:
            raise ValueError("candidate 当前仅支持周期性重建岛，不保留岛状态")
        if (
            self.fe_scheduler == "fixed_batch"
            and self.pairing_strategy == "none"
            and self.recombination_fe_ratio > 0
        ):
            raise ValueError("pairing_strategy=none 时 recombination_fe_ratio 必须为 0")


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

    @property
    def algorithm_schema_version(self) -> int:
        if self.normalized_algorithm == "IEMOEC":
            return self.iemoec.algorithm_schema_version
        return BASELINE_SCHEMA_VERSION

    @property
    def algorithm_variant(self) -> str:
        if self.normalized_algorithm == "IEMOEC":
            return self.iemoec.variant
        return "baseline"

    @property
    def algorithm_label(self) -> str:
        if self.normalized_algorithm != "IEMOEC":
            return ALGORITHM_LABELS[self.normalized_algorithm]
        return {
            "rank": "IEMOEC-Rank",
            "rank_crowding": "IEMOEC-CD",
            "nsga3": "IEMOEC-RD",
        }[self.iemoec.outer_survival]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["algorithm"] = self.normalized_algorithm
        data["problem"] = self.normalized_problem
        data["algorithm_variant"] = self.algorithm_variant
        data["algorithm_schema_version"] = self.algorithm_schema_version
        data["algorithm_label"] = self.algorithm_label
        return data


def default_ref_partitions(n_obj: int) -> int:
    mapping = {2: 99, 3: 12, 5: 6, 8: 3, 10: 3, 15: 2}
    try:
        return mapping[n_obj]
    except KeyError as exc:
        raise ValueError(f"没有 M={n_obj} 的参考方向划分") from exc
