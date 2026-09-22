from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_ALGORITHMS = (
    "NSGA2", "NSGA3", "MOEAD", "MOEADPBI", "RVEA", "CTAEA",
    "AGEMOEA2", "AGEMOEA2STABLE", "IEMOEC",
)
DEFAULT_ALGORITHMS = ("NSGA2", "NSGA3", "MOEAD", "IEMOEC")
SUPPORTED_OBJECTIVES = (2, 3, 5, 8, 10, 15)
ALGORITHM_LABELS = {
    "NSGA2": "NSGA-II",
    "NSGA3": "NSGA-III",
    "MOEAD": "MOEA/D-TCH",
    "MOEADPBI": "MOEA/D-PBI",
    "RVEA": "RVEA",
    "CTAEA": "C-TAEA",
    "AGEMOEA2": "AGE-MOEA2",
    "AGEMOEA2STABLE": "AGE-MOEA2-Stable",
    "IEMOEC": "IEMOEC",
}

IEMOEC_SCHEMA_VERSIONS = {
    "v0": 0,
    "s1": 1,
    "candidate": 2,
    "s2": 3,
    "s2_no_isolation": 4,
    "s3_memory": 5,
    "s3_hybrid": 6,
    "s3_elite": 7,
    "s3": 8,
    "principle": 9,
    "s4": 10,
    "s3_elite_constrained": 11,
}
S3_VARIANTS = (
    "s3_memory", "s3_hybrid", "s3_elite", "s3",
    "s3_elite_constrained",
)
S3_CONFIG_FIELDS = {
    "direction_memory",
    "direction_memory_capacity",
    "isolated_fe_ratio",
    "shared_fe_ratio",
    "protect_direction_elites",
    "adaptive_source_budget",
    "source_budget_smoothing",
}
BASELINE_SCHEMA_VERSION = 2
STABLE_AGEMOEA2_SCHEMA_VERSION = 3


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
    local_evolution_mode: str = "island"
    direction_memory: bool = False
    direction_memory_capacity: int | None = None
    isolated_fe_ratio: float = 0.75
    shared_fe_ratio: float = 0.0
    protect_direction_elites: bool = False
    adaptive_source_budget: bool = False
    source_budget_smoothing: float = 0.8
    principle_min_generations: int = 3
    principle_stagnation_generations: int = 3
    principle_probe_radius: float = 0.01
    principle_tolerance: float = 1e-3

    @classmethod
    def for_variant(cls, variant: str, **overrides) -> IEMOECConfig:
        """构造可复现的 V0、S1 或候选结构配置。"""
        profiles = {
            "principle": {
                "initialization_mode": "legacy_origin",
                "normalization_mode": "legacy",
                "min_origin": 2,
                "island_initialization": "single_ancestor",
                "outer_survival": "rank",
                "pairing_strategy": "random",
            },
            "s4": {
                "initialization_mode": "legacy_origin",
                "normalization_mode": "legacy",
                "min_origin": 2,
                "island_initialization": "single_ancestor",
                "outer_survival": "rank",
                "pairing_strategy": "random",
            },
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
            "s2": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "local_fe_ratio": 0.75,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
            },
            "s2_no_isolation": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "local_fe_ratio": 0.75,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
                "local_evolution_mode": "shared",
            },
            "s3_memory": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "isolated_fe_ratio": 0.75,
                "shared_fe_ratio": 0.0,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
                "local_evolution_mode": "island",
                "direction_memory": True,
            },
            "s3_hybrid": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "isolated_fe_ratio": 0.50,
                "shared_fe_ratio": 0.25,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
                "local_evolution_mode": "hybrid",
                "direction_memory": True,
            },
            "s3_elite": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "isolated_fe_ratio": 0.50,
                "shared_fe_ratio": 0.25,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
                "local_evolution_mode": "hybrid",
                "direction_memory": True,
                "protect_direction_elites": True,
            },
            "s3_elite_constrained": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "isolated_fe_ratio": 0.50,
                "shared_fe_ratio": 0.25,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
                "local_evolution_mode": "hybrid",
                "direction_memory": True,
                "protect_direction_elites": True,
            },
            "s3": {
                "initialization_mode": "shared_population",
                "normalization_mode": "global",
                "island_initialization": "multi_ancestor",
                "island_direction_mode": "axis_random",
                "island_count_multiplier": 2,
                "diverse_ancestors": 1,
                "fe_scheduler": "fixed_batch",
                "outer_batch_ratio": 1.0,
                "isolated_fe_ratio": 0.50,
                "shared_fe_ratio": 0.25,
                "recombination_fe_ratio": 0.25,
                "outer_survival": "nsga3",
                "pairing_strategy": "farthest_weight",
                "local_evolution_mode": "hybrid",
                "direction_memory": True,
                "protect_direction_elites": True,
                "adaptive_source_budget": True,
            },
        }
        if variant not in profiles:
            raise ValueError(f"未知 IEMOEC variant: {variant}")
        values = {**profiles[variant], **overrides, "variant": variant}
        return cls(**values)

    @property
    def algorithm_schema_version(self) -> int:
        return IEMOEC_SCHEMA_VERSIONS[self.variant]

    @property
    def uses_candidate_architecture(self) -> bool:
        return self.variant in ("candidate", "s2", "s2_no_isolation")

    @property
    def uses_s3_architecture(self) -> bool:
        return self.variant in S3_VARIANTS

    def validate(self) -> None:
        if self.variant in ("principle", "s4"):
            if not 0 < self.origin_ratio <= 1 or self.island_population < 2:
                raise ValueError("principle requires origin_ratio in (0,1] and island_population >= 2")
            if self.principle_min_generations < 1 or self.principle_stagnation_generations < 1:
                raise ValueError("principle generation thresholds must be positive")
            if not 0 < self.principle_probe_radius < 1 or not math.isfinite(self.principle_tolerance) or self.principle_tolerance < 0:
                raise ValueError("invalid principle probe radius or tolerance")
            if self.outer_survival != "rank" or self.use_crowding or not self.enable_recombination:
                raise ValueError("principle requires rank-only survival, no CD, and recombination")
            if self.island_initialization != "single_ancestor" or self.direction_memory or self.protect_direction_elites or self.adaptive_source_budget or self.shared_fe_ratio != 0:
                raise ValueError("principle forbids multi-ancestor pools, shared breeding and direction archives")
            return
        if self.variant not in IEMOEC_SCHEMA_VERSIONS:
            raise ValueError(
                f"variant 必须属于 {sorted(IEMOEC_SCHEMA_VERSIONS)}"
            )
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
        if not self.uses_s3_architecture and not math.isclose(
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
        if self.local_evolution_mode not in ("island", "shared", "hybrid"):
            raise ValueError(
                "local_evolution_mode 仅支持 island、shared 或 hybrid"
            )
        if self.variant == "s2_no_isolation" and self.local_evolution_mode != "shared":
            raise ValueError("s2_no_isolation 必须使用 shared 局部演化")
        if (
            self.variant not in ("s2_no_isolation", *S3_VARIANTS)
            and self.local_evolution_mode != "island"
        ):
            raise ValueError("当前 variant 不允许共享或混合局部演化")
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
        if self.uses_candidate_architecture and self.retain_island_state:
            raise ValueError("candidate/s2 仅支持周期性重建岛，不保留岛状态")
        if (
            self.fe_scheduler == "fixed_batch"
            and self.pairing_strategy == "none"
            and self.recombination_fe_ratio > 0
        ):
            raise ValueError("pairing_strategy=none 时 recombination_fe_ratio 必须为 0")
        if self.direction_memory_capacity is not None and self.direction_memory_capacity < 2:
            raise ValueError("direction_memory_capacity 必须至少为 2")
        if not 0 <= self.isolated_fe_ratio <= 1:
            raise ValueError("isolated_fe_ratio 必须在 [0, 1] 内")
        if not 0 <= self.shared_fe_ratio <= 1:
            raise ValueError("shared_fe_ratio 必须在 [0, 1] 内")
        if not 0 <= self.source_budget_smoothing < 1:
            raise ValueError("source_budget_smoothing 必须在 [0, 1) 内")
        if self.uses_s3_architecture:
            if not self.direction_memory:
                raise ValueError("S3 必须启用持久方向记忆")
            if self.initialization_mode != "shared_population":
                raise ValueError("S3 必须使用 shared_population 初始化")
            if self.normalization_mode != "global" or self.fe_scheduler != "fixed_batch":
                raise ValueError("S3 必须使用 global 归一化和 fixed_batch 调度")
            if not math.isclose(
                self.isolated_fe_ratio
                + self.shared_fe_ratio
                + self.recombination_fe_ratio,
                1.0,
                abs_tol=1e-12,
            ):
                raise ValueError("S3 三类来源 FE 比例之和必须等于 1")
            if not self.enable_recombination and self.recombination_fe_ratio > 0:
                raise ValueError("关闭重组时 recombination_fe_ratio 必须为 0")
            expected_mode = "island" if self.variant == "s3_memory" else "hybrid"
            if self.local_evolution_mode != expected_mode:
                raise ValueError(
                    f"{self.variant} 必须使用 {expected_mode} 局部演化模式"
                )
            if self.variant == "s3_memory" and (
                self.shared_fe_ratio != 0
                or self.protect_direction_elites
                or self.adaptive_source_budget
            ):
                raise ValueError("s3_memory 只能启用持久方向记忆")
            if self.variant == "s3_hybrid" and (
                self.shared_fe_ratio <= 0
                or self.protect_direction_elites
                or self.adaptive_source_budget
            ):
                raise ValueError("s3_hybrid 只增加共享后代")
            if self.variant in ("s3_elite", "s3_elite_constrained") and (
                not self.protect_direction_elites
                or self.adaptive_source_budget
            ):
                raise ValueError("s3_elite 必须启用方向精英保护且关闭自适应预算")
            if self.variant == "s3" and (
                not self.protect_direction_elites
                or not self.adaptive_source_budget
            ):
                raise ValueError("s3 必须启用方向精英保护和自适应预算")


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
    timing_only: bool = False
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
        from .problems import is_constrained_problem_name

        constrained = is_constrained_problem_name(self.normalized_problem)
        if constrained and self.normalized_algorithm in ("MOEAD", "MOEADPBI"):
            raise ValueError(
                "pymoo 的 MOEA/D 不支持约束问题；请使用明确命名的约束版本"
            )
        if self.normalized_algorithm == "IEMOEC":
            constrained_variant = self.iemoec.variant == "s3_elite_constrained"
            if constrained and not constrained_variant:
                raise ValueError(
                    "约束问题必须使用 IEMOEC variant=s3_elite_constrained"
                )
            if constrained_variant and not constrained:
                raise ValueError("s3_elite_constrained 只能用于约束问题")

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
        if self.normalized_algorithm == "AGEMOEA2STABLE":
            return STABLE_AGEMOEA2_SCHEMA_VERSION
        return BASELINE_SCHEMA_VERSION

    @property
    def algorithm_variant(self) -> str:
        if self.normalized_algorithm == "IEMOEC":
            return self.iemoec.variant
        if self.normalized_algorithm == "AGEMOEA2STABLE":
            return "stable_zero_norm_guard"
        return "baseline"

    @property
    def algorithm_label(self) -> str:
        if self.normalized_algorithm != "IEMOEC":
            return ALGORITHM_LABELS[self.normalized_algorithm]
        if self.iemoec.variant == "s3_elite_constrained":
            return "IEMOEC-C"
        if self.iemoec.variant in ("principle", "s4"):
            return "IEMOEC-Principle" if self.iemoec.variant == "principle" else "IEMOEC-S4"
        base = {
            "rank": "IEMOEC-Rank",
            "rank_crowding": "IEMOEC-CD",
            "nsga3": "IEMOEC-RD",
        }[self.iemoec.outer_survival]
        suffixes = {
            "s2_no_isolation": "-NoIsolation",
            "s3_memory": "-S3-Memory",
            "s3_hybrid": "-S3-Hybrid",
            "s3_elite": "-S3-Elite",
            "s3": "-S3-Full",
        }
        return base + suffixes.get(self.iemoec.variant, "")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        principle_fields = [name for name in data["iemoec"] if name.startswith("principle_")]
        if self.iemoec.variant not in ("principle", "s4"):
            for name in principle_fields:
                data["iemoec"].pop(name)
        else:
            used = {"variant", "origin_ratio", "island_population", *principle_fields}
            data["iemoec"] = {name: value for name, value in data["iemoec"].items() if name in used}
        if not self.iemoec.uses_s3_architecture:
            for field_name in S3_CONFIG_FIELDS:
                data["iemoec"].pop(field_name, None)
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
