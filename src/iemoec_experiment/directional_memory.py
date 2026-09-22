from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.core.population import Population

from .constraints import FEASIBILITY_TOLERANCE, constraint_violation
from .normalization import ObjectiveNormalization


@dataclass(frozen=True)
class DirectionMemoryStatistics:
    """一次方向记忆更新的轻量诊断。"""

    turnover: float
    stagnant_ratio: float


class DirectionalMemory:
    """保存跨外循环的方向微种群。

    每个方向先保留标量化最优解，再从该方向的较优候选中选择决策空间
    差异较大的个体。不同方向持有独立的 Individual 副本，避免后续算子
    就地修改时相互污染。
    """

    def __init__(
        self,
        problem,
        weights: list[np.ndarray],
        capacity: int,
        stagnation_window: int = 5,
        constraint_aware: bool = False,
    ) -> None:
        if capacity < 2:
            raise ValueError("方向记忆容量必须至少为 2")
        if not weights:
            raise ValueError("方向记忆至少需要一个方向")
        self.problem = problem
        self.weights = [np.asarray(weight, dtype=float).copy() for weight in weights]
        self.capacity = int(capacity)
        self.stagnation_window = int(stagnation_window)
        self.constraint_aware = bool(constraint_aware)
        self._populations = [Population.empty() for _ in self.weights]
        self._stagnation = np.zeros(len(self.weights), dtype=int)

    @staticmethod
    def _key(individual) -> bytes:
        return np.ascontiguousarray(
            individual.get("X"),
            dtype=np.float64,
        ).tobytes()

    @staticmethod
    def _copy_individual(individual):
        """复制会被算法修改的个体状态，避免昂贵的递归深拷贝。

        pymoo 个体的 ``config`` 只保存约束计算规则，可以安全共享；决策、
        目标等数组以及 ``data`` 和 ``evaluated`` 则必须独立。
        """
        copied = individual.copy(deep=False)
        for name, value in individual.__dict__.items():
            if isinstance(value, np.ndarray):
                copied.__dict__[name] = value.copy()
            elif name == "data":
                copied.__dict__[name] = {
                    key: item.copy() if isinstance(item, np.ndarray) else item
                    for key, item in value.items()
                }
            elif name == "evaluated":
                copied.__dict__[name] = value.copy()
        return copied

    @classmethod
    def _copy_population(cls, population: Population) -> Population:
        if not len(population):
            return Population.empty()
        return Population.create(
            *(cls._copy_individual(individual) for individual in population)
        )

    def _deduplicate(self, population: Population) -> Population:
        selected_indices = []
        seen: set[bytes] = set()
        for index, key in enumerate(self._population_keys(population)):
            if key in seen:
                continue
            seen.add(key)
            selected_indices.append(index)
        if not selected_indices:
            return Population.empty()
        return population[np.asarray(selected_indices, dtype=int)]

    @staticmethod
    def _population_keys(population: Population) -> list[bytes]:
        """一次提取种群决策键，避免逐个 Individual 重复查询属性。"""
        decisions = np.asarray(population.get("X"), dtype=np.float64)
        return [np.ascontiguousarray(row).tobytes() for row in decisions]

    def _normalized_decisions(self, population: Population) -> np.ndarray:
        decisions = np.asarray(population.get("X"), dtype=float)
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        return (decisions - lower) / np.maximum(upper - lower, 1e-12)

    def _select_from_cached_candidates(
        self,
        previous: Population,
        previous_keys: list[bytes],
        candidates: Population,
        candidate_keys: list[bytes],
        normalized_candidate_objectives: np.ndarray,
        candidate_decisions: np.ndarray,
        candidate_violation: np.ndarray,
        weight: np.ndarray,
        normalization: ObjectiveNormalization,
    ) -> tuple[
        Population,
        tuple[int, float],
        tuple[int, float] | None,
    ]:
        """从缓存候选数据更新一个方向，并返回其最优标量值。

        逻辑顺序与 ``deduplicate(merge(previous, candidates))`` 完全一致：
        先放旧记忆，再依次放尚未出现的候选。保持这个顺序可确保标量值
        相同时，稳定排序仍选择与旧实现相同的个体。
        """
        previous_key_set = set(previous_keys)
        new_candidate_indices = np.asarray(
            [
                index
                for index, key in enumerate(candidate_keys)
                if key not in previous_key_set
            ],
            dtype=int,
        )

        previous_count = len(previous)
        if previous_count:
            normalized_objectives = np.concatenate(
                [
                    normalization.apply(previous.get("F")),
                    normalized_candidate_objectives[new_candidate_indices],
                ],
                axis=0,
            )
            decisions = np.concatenate(
                [
                    self._normalized_decisions(previous),
                    candidate_decisions[new_candidate_indices],
                ],
                axis=0,
            )
        else:
            normalized_objectives = normalized_candidate_objectives
            decisions = candidate_decisions

        scores = np.max(
            np.asarray(weight, dtype=float) * np.abs(normalized_objectives),
            axis=1,
        )
        if self.constraint_aware:
            if previous_count:
                violation = np.concatenate([
                    constraint_violation(previous),
                    candidate_violation[new_candidate_indices],
                ])
            else:
                violation = candidate_violation
            feasible = violation <= FEASIBILITY_TOLERANCE
            ranking_scores = np.where(feasible, scores, violation)
            ordered = np.lexsort((
                np.arange(len(scores)),
                ranking_scores,
                (~feasible).astype(int),
            ))
            quality_keys = [
                (0, float(scores[index]))
                if feasible[index]
                else (1, float(violation[index]))
                for index in range(len(scores))
            ]
        else:
            ordered = np.argsort(scores, kind="stable")
            quality_keys = [(0, float(score)) for score in scores]
        previous_best = (
            min(quality_keys[:previous_count])
            if previous_count
            else None
        )
        if self.constraint_aware:
            # 第一版约束策略严格采用 feasibility-first。方向记忆容量有限时，
            # 可行解按原方向标量值竞争；不可行解之间只比较 CV，不允许
            # 决策空间距离把较差的不可行解提升到较好解之前。
            selected = ordered[:self.capacity]
        elif len(scores) <= self.capacity:
            selected = np.arange(len(scores), dtype=int)
        else:
            shortlist_size = min(len(ordered), self.capacity * 3)
            shortlist = ordered[:shortlist_size]
            chosen = [int(ordered[0])]
            while len(chosen) < self.capacity:
                available = np.asarray(
                    [index for index in shortlist if int(index) not in chosen],
                    dtype=int,
                )
                if len(available) == 0:
                    available = np.asarray(
                        [index for index in ordered if int(index) not in chosen],
                        dtype=int,
                    )
                distances = np.linalg.norm(
                    decisions[available, None, :]
                    - decisions[np.asarray(chosen, dtype=int)][None, :, :],
                    axis=2,
                )
                min_distances = np.min(distances, axis=1)
                chosen.append(int(available[int(np.argmax(min_distances))]))
            selected = np.asarray(chosen, dtype=int)

        selected_individuals = []
        for index in selected:
            if index < previous_count:
                selected_individuals.append(previous[int(index)])
            else:
                candidate_index = new_candidate_indices[int(index) - previous_count]
                selected_individuals.append(candidates[int(candidate_index)])
        updated = Population.create(
            *(self._copy_individual(individual) for individual in selected_individuals)
        )
        updated_best = min(quality_keys[int(index)] for index in selected)
        return updated, updated_best, previous_best

    def update(
        self,
        candidates: Population,
        normalization: ObjectiveNormalization,
    ) -> DirectionMemoryStatistics:
        """用统一候选集更新所有方向，并返回周转和停滞统计。"""
        if not len(candidates):
            raise ValueError("不能用空候选集更新方向记忆")

        unique_candidates = self._deduplicate(candidates)
        candidate_keys = self._population_keys(unique_candidates)
        normalized_candidate_objectives = normalization.apply(
            unique_candidates.get("F")
        )
        candidate_decisions = self._normalized_decisions(unique_candidates)
        candidate_violation = constraint_violation(unique_candidates)

        turnovers = []
        for direction_id, weight in enumerate(self.weights):
            previous = self._populations[direction_id]
            previous_keys = self._population_keys(previous)
            previous_key_set = set(previous_keys)

            updated, updated_best, previous_best = self._select_from_cached_candidates(
                previous,
                previous_keys,
                unique_candidates,
                candidate_keys,
                normalized_candidate_objectives,
                candidate_decisions,
                candidate_violation,
                weight,
                normalization,
            )
            improved = previous_best is None
            if previous_best is not None:
                improved = (
                    updated_best[0] < previous_best[0]
                    or (
                        updated_best[0] == previous_best[0]
                        and updated_best[1] < previous_best[1] - 1e-12
                    )
                )
            if improved:
                self._stagnation[direction_id] = 0
            else:
                self._stagnation[direction_id] += 1

            introduced = sum(
                key not in previous_key_set
                for key in self._population_keys(updated)
            )
            turnovers.append(introduced / max(1, len(updated)))
            self._populations[direction_id] = updated

        stagnant = self._stagnation >= self.stagnation_window
        return DirectionMemoryStatistics(
            turnover=float(np.mean(turnovers)),
            stagnant_ratio=float(np.mean(stagnant)),
        )

    def parent_pools(self) -> list[Population]:
        """返回各方向的独立副本，供方向内交叉使用。"""
        return [self._copy_population(population) for population in self._populations]

    def representatives(
        self,
        normalization: ObjectiveNormalization,
    ) -> Population:
        """返回每个方向的标量化最优代表。"""
        selected = []
        for population, weight in zip(self._populations, self.weights):
            scores = normalization.tchebycheff(population.get("F"), weight)
            if self.constraint_aware:
                violation = constraint_violation(population)
                feasible = violation <= FEASIBILITY_TOLERANCE
                order = np.lexsort((
                    np.arange(len(population)),
                    np.where(feasible, scores, violation),
                    (~feasible).astype(int),
                ))
                best = int(order[0])
            else:
                best = int(np.argmin(scores))
            selected.append(
                self._copy_individual(population[best])
            )
        return Population.create(*selected)

    def combined_population(self) -> Population:
        """返回全部方向记忆的决策去重副本。"""
        selected = []
        seen: set[bytes] = set()
        for population in self._populations:
            for individual, key in zip(
                population,
                self._population_keys(population),
            ):
                if key in seen:
                    continue
                seen.add(key)
                selected.append(self._copy_individual(individual))
        if not selected:
            return Population.empty()
        return Population.create(*selected)
