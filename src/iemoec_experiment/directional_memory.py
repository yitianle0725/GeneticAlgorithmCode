from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.core.population import Population

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
    ) -> None:
        if capacity < 2:
            raise ValueError("方向记忆容量必须至少为 2")
        if not weights:
            raise ValueError("方向记忆至少需要一个方向")
        self.problem = problem
        self.weights = [np.asarray(weight, dtype=float).copy() for weight in weights]
        self.capacity = int(capacity)
        self.stagnation_window = int(stagnation_window)
        self._populations = [Population.empty() for _ in self.weights]
        self._stagnation = np.zeros(len(self.weights), dtype=int)

    @staticmethod
    def _key(individual) -> bytes:
        return np.ascontiguousarray(
            individual.get("X"),
            dtype=np.float64,
        ).tobytes()

    @staticmethod
    def _copy_population(population: Population) -> Population:
        if not len(population):
            return Population.empty()
        return Population.create(*(individual.copy() for individual in population))

    def _deduplicate(self, population: Population) -> Population:
        selected = []
        seen: set[bytes] = set()
        for individual in population:
            key = self._key(individual)
            if key in seen:
                continue
            seen.add(key)
            selected.append(individual)
        if not selected:
            return Population.empty()
        return Population.create(*selected)

    def _normalized_decisions(self, population: Population) -> np.ndarray:
        decisions = np.asarray(population.get("X"), dtype=float)
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        return (decisions - lower) / np.maximum(upper - lower, 1e-12)

    def _select_for_direction(
        self,
        population: Population,
        weight: np.ndarray,
        normalization: ObjectiveNormalization,
    ) -> Population:
        unique = self._deduplicate(population)
        if len(unique) <= self.capacity:
            return self._copy_population(unique)

        scores = normalization.tchebycheff(unique.get("F"), weight)
        ordered = np.argsort(scores, kind="stable")
        shortlist_size = min(len(unique), max(self.capacity * 3, self.capacity))
        shortlist = ordered[:shortlist_size]
        normalized_x = self._normalized_decisions(unique)

        selected = [int(ordered[0])]
        while len(selected) < self.capacity:
            available = np.asarray(
                [index for index in shortlist if int(index) not in selected],
                dtype=int,
            )
            if len(available) == 0:
                available = np.asarray(
                    [index for index in ordered if int(index) not in selected],
                    dtype=int,
                )
            distances = np.linalg.norm(
                normalized_x[available, None, :]
                - normalized_x[np.asarray(selected, dtype=int)][None, :, :],
                axis=2,
            )
            min_distances = np.min(distances, axis=1)
            selected.append(int(available[int(np.argmax(min_distances))]))

        chosen = unique[np.asarray(selected, dtype=int)]
        return self._copy_population(chosen)

    def update(
        self,
        candidates: Population,
        normalization: ObjectiveNormalization,
    ) -> DirectionMemoryStatistics:
        """用统一候选集更新所有方向，并返回周转和停滞统计。"""
        if not len(candidates):
            raise ValueError("不能用空候选集更新方向记忆")

        turnovers = []
        for direction_id, weight in enumerate(self.weights):
            previous = self._populations[direction_id]
            previous_keys = {self._key(individual) for individual in previous}
            combined = candidates
            if len(previous):
                combined = Population.merge(previous, candidates)

            previous_best = None
            if len(previous):
                previous_best = float(np.min(normalization.tchebycheff(
                    previous.get("F"),
                    weight,
                )))
            updated = self._select_for_direction(combined, weight, normalization)
            updated_best = float(np.min(normalization.tchebycheff(
                updated.get("F"),
                weight,
            )))
            if previous_best is None or updated_best < previous_best - 1e-12:
                self._stagnation[direction_id] = 0
            else:
                self._stagnation[direction_id] += 1

            introduced = sum(
                self._key(individual) not in previous_keys
                for individual in updated
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
            selected.append(population[int(np.argmin(scores))].copy())
        return Population.create(*selected)

    def combined_population(self) -> Population:
        """返回全部方向记忆的决策去重副本。"""
        merged = Population.empty()
        for population in self._populations:
            copied = self._copy_population(population)
            merged = copied if not len(merged) else Population.merge(merged, copied)
        return self._copy_population(self._deduplicate(merged))

