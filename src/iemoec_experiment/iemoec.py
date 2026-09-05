from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from .config import ExperimentCase
from .factory import reference_directions


def _merge(*populations: Population) -> Population:
    available = [pop for pop in populations if pop is not None and len(pop)]
    if not available:
        return Population.empty()
    result = available[0]
    for pop in available[1:]:
        result = Population.merge(result, pop)
    return result


class IEMOECRunner:
    """IEMOEC 自定义核心；评价、算子、排序和 survival 均复用 pymoo。

    与旧实现相比，这个版本没有早停和 PF 后处理，跨岛组合受固定 FE 预算约束。
    """

    def __init__(
        self,
        problem,
        case: ExperimentCase,
        on_checkpoint: Callable[[int, Population], None] | None = None,
        on_outer_selection: Callable[[int, Population], dict | None] | None = None,
    ):
        self.problem = problem
        self.case = case
        self.config = case.iemoec
        self.rng = np.random.default_rng(case.seed)
        self.evaluator = Evaluator()
        self.ref_dirs = reference_directions(case)
        self.pop_size = len(self.ref_dirs)
        self.n_origin = min(
            self.pop_size,
            max(self.config.min_origin, math.ceil(self.pop_size * self.config.origin_ratio)),
        )
        self.sampling = FloatRandomSampling()
        self.sbx = SBX(prob=1.0, eta=30)
        self.mutation = PM(prob=1.0, prob_var=None, eta=20)
        self.expansion_mutation = PM(
            prob=1.0,
            prob_var=self.config.expansion_mutation_probability,
            eta=20,
        )
        self.on_checkpoint = on_checkpoint
        self.on_outer_selection = on_outer_selection
        self.origin = Population.empty()
        self.candidate_pool = Population.empty()
        self.outer_records: list[dict] = []

    @property
    def n_eval(self) -> int:
        return int(self.evaluator.n_eval)

    @property
    def remaining(self) -> int:
        return max(0, self.case.max_fes - self.n_eval)

    def _evaluate(self, pop: Population) -> Population:
        if len(pop) > self.remaining:
            pop = pop[: self.remaining]
        if not len(pop):
            return pop
        self.evaluator.eval(self.problem, pop)
        if self.on_checkpoint is not None:
            # 公共 checkpoint 始终使用最近一次完成全局筛选后的 archive。
            # 初始化阶段尚无 archive，才使用刚完成评价的起源种群。
            checkpoint_population = self.candidate_pool
            if len(checkpoint_population) == 0:
                checkpoint_population = pop
            self.on_checkpoint(self.n_eval, checkpoint_population)
        return pop

    def _mate(self, parent_pool: Population, pairs: np.ndarray, budget: int) -> Population:
        if budget <= 0 or len(pairs) == 0:
            return Population.empty()
        children = self.sbx.do(
            self.problem, parent_pool, parents=pairs, random_state=self.rng
        )
        children = self.mutation.do(
            self.problem, children, inplace=True, random_state=self.rng
        )
        return children[:budget]

    def _front_truncate(self, pop: Population, n_survive: int, weight=None) -> Population:
        if len(pop) <= n_survive:
            return pop
        if self.config.use_crowding:
            return RankAndCrowding().do(
                self.problem,
                pop,
                n_survive=n_survive,
                random_state=self.rng,
            )
        fronts = NonDominatedSorting().do(pop.get("F"))
        selected: list[int] = []
        for front in fronts:
            remaining = n_survive - len(selected)
            if remaining <= 0:
                break
            if len(front) <= remaining:
                selected.extend(front.tolist())
                continue
            if weight is None:
                chosen = self.rng.choice(front, size=remaining, replace=False)
            else:
                F = pop[front].get("F")
                ideal = np.min(pop.get("F"), axis=0)
                score = np.max(weight * np.abs(F - ideal), axis=1)
                chosen = front[np.argsort(score)[:remaining]]
            selected.extend(np.asarray(chosen, dtype=int).tolist())
            break
        return pop[np.asarray(selected, dtype=int)]

    def _outer_select(self, pop: Population, n_survive: int) -> Population:
        n_survive = min(n_survive, len(pop))
        if self.config.outer_survival == "nsga3":
            from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival

            return ReferenceDirectionSurvival(self.ref_dirs).do(
                self.problem,
                pop,
                n_survive=n_survive,
                random_state=self.rng,
            )
        return self._front_truncate(pop, n_survive)

    def _initialize(self) -> None:
        n = min(self.n_origin, self.remaining)
        self.origin = self.sampling.do(self.problem, n, random_state=self.rng)
        self.origin = self._evaluate(self.origin)
        self.candidate_pool = self.origin

    def _island_definitions(self):
        """创建一次岛身份，使状态消融不受权重重采样干扰。"""
        specs = []
        for m in range(self.problem.n_obj):
            weight = np.full(self.problem.n_obj, self.config.aggregation_epsilon)
            weight[m] = 1.0
            specs.append((weight, m))
        if self.config.islands_per_objective == 2:
            weights = self.rng.dirichlet(np.ones(self.problem.n_obj), self.problem.n_obj)
            for weight in weights:
                specs.append((weight, None))
        return specs

    def _ancestor_index(self, weight: np.ndarray, objective: int | None) -> int:
        scores = self._direction_scores(self.origin, weight, objective)
        return int(np.argmin(scores))

    @staticmethod
    def _direction_scores(
        population: Population,
        weight: np.ndarray,
        objective: int | None,
    ) -> np.ndarray:
        F = population.get("F")
        if objective is not None:
            return F[:, objective]
        ideal = np.min(F, axis=0)
        return np.max(weight * np.abs(F - ideal), axis=1)

    def _decision_space_points(self, population: Population) -> np.ndarray:
        X = np.asarray(population.get("X"), dtype=float)
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        return (X - lower) / np.maximum(upper - lower, 1e-12)

    def _multi_ancestor_island(
        self,
        weight: np.ndarray,
        objective: int | None,
    ) -> Population:
        """从已评价的全局候选池周期性建立一个多祖先岛。"""
        source = self.candidate_pool
        target = min(self.config.island_population, len(source))
        if target == 0:
            return Population.empty()

        scores = self._direction_scores(source, weight, objective)
        ranked = np.argsort(scores, kind="stable")
        selected = [int(ranked[0])]

        neighbor_count = min(
            self.config.direction_neighbor_ancestors,
            target - len(selected),
        )
        selected.extend(int(index) for index in ranked[1:1 + neighbor_count])

        normalized_x = self._decision_space_points(source)
        diverse_count = min(
            self.config.diverse_ancestors,
            target - len(selected),
        )
        for _ in range(diverse_count):
            available = np.setdiff1d(
                np.arange(len(source)),
                np.asarray(selected, dtype=int),
                assume_unique=False,
            )
            distances = np.linalg.norm(
                normalized_x[available, None, :] - normalized_x[selected][None, :, :],
                axis=2,
            )
            min_distances = np.min(distances, axis=1)
            selected.append(int(available[int(np.argmax(min_distances))]))

        remaining_slots = target - len(selected)
        if remaining_slots > 0:
            direction_pool_size = min(len(source), max(target * 2, target))
            direction_candidates = np.setdiff1d(
                ranked[:direction_pool_size],
                np.asarray(selected, dtype=int),
                assume_unique=False,
            )
            chosen_count = min(remaining_slots, len(direction_candidates))
            if chosen_count:
                chosen = self.rng.choice(
                    direction_candidates,
                    size=chosen_count,
                    replace=False,
                )
                selected.extend(int(index) for index in np.atleast_1d(chosen))

        remaining_slots = target - len(selected)
        if remaining_slots > 0:
            available = np.setdiff1d(
                np.arange(len(source)),
                np.asarray(selected, dtype=int),
                assume_unique=False,
            )
            chosen = self.rng.choice(available, size=remaining_slots, replace=False)
            selected.extend(int(index) for index in np.atleast_1d(chosen))

        return Population.create(*(source[index].copy() for index in selected))

    def _expand_island(self, ancestor) -> Population:
        target = self.config.island_population
        n_children = min(target - 1, self.remaining)
        island = Population.create(ancestor.copy())
        if n_children <= 0:
            return island
        repeated = Population.new(
            "X", np.repeat(ancestor.get("X")[None, :], n_children, axis=0)
        )
        children = self.expansion_mutation.do(
            self.problem, repeated, inplace=True, random_state=self.rng
        )
        children = self._evaluate(children)
        return _merge(island, children)

    def _create_islands(self, definitions) -> tuple[list[Population], list[np.ndarray]]:
        islands = []
        weights = []
        for weight, objective in definitions:
            if self.config.island_initialization == "multi_ancestor":
                island = self._multi_ancestor_island(weight, objective)
            else:
                ancestor_idx = self._ancestor_index(weight, objective)
                island = self._expand_island(self.origin[ancestor_idx])
            islands.append(island)
            weights.append(weight)
            if self.remaining <= 0:
                break
        return islands, weights

    def _evolve_island(self, island: Population, weight, phase: str) -> Population:
        n = len(island)
        if n < 2 or self.remaining <= 0:
            return island
        order = self.rng.permutation(n)
        if n % 2:
            order = np.append(order, order[0])
        pairs = order.reshape(-1, 2)
        offspring = self._mate(island, pairs, min(n, self.remaining))
        offspring = self._evaluate(offspring)
        merged = _merge(island, offspring)
        if phase == "aggregation":
            ideal = np.min(merged.get("F"), axis=0)
            scores = np.max(weight * np.abs(merged.get("F") - ideal), axis=1)
            return merged[np.argsort(scores)[:n]]
        return self._front_truncate(merged, n, weight=weight)

    def _recombine(
        self,
        islands: list[Population],
        weights: list[np.ndarray],
        budget_ratio: float,
    ) -> Population:
        if not self.config.enable_recombination or self.remaining <= 0:
            return Population.empty()
        elites = []
        for island_id, island in enumerate(islands):
            if not len(island):
                continue
            ideal = np.min(island.get("F"), axis=0)
            score = np.max(
                weights[island_id] * np.abs(island.get("F") - ideal), axis=1
            )
            elites.append(island[int(np.argmin(score))])
        if len(elites) < 2:
            return Population.empty()
        elite_pop = Population.create(*elites)
        pair_set = set()
        for i in range(len(elites)):
            candidates = np.delete(np.arange(len(elites)), i)
            count = min(self.config.partners_per_elite, len(candidates))
            wi = weights[i] / max(np.linalg.norm(weights[i]), 1e-12)
            similarity = []
            for j in candidates:
                wj = weights[int(j)] / max(np.linalg.norm(weights[int(j)]), 1e-12)
                similarity.append(float(np.dot(wi, wj)))
            for j in candidates[np.argsort(similarity)[:count]]:
                pair_set.add(tuple(sorted((i, int(j)))))
        pair_list = list(pair_set)
        self.rng.shuffle(pair_list)
        requested = math.ceil(self.pop_size * budget_ratio)
        budget = min(requested, self.remaining)
        pairs = np.asarray(pair_list, dtype=int)
        # SBX 每组产生两个后代；只创建覆盖预算所需的 mating 数。
        pairs = pairs[: math.ceil(budget / 2)]
        offspring = self._mate(elite_pop, pairs, budget)
        return self._evaluate(offspring)

    def run(self) -> tuple[Population, int]:
        self._initialize()
        island_definitions = None
        if self.config.retain_island_state or self.config.fixed_island_definitions:
            island_definitions = self._island_definitions()
        islands: list[Population] = []
        weights: list[np.ndarray] = []
        outer = 0
        while self.remaining > 0 and len(self.origin):
            fe_start = self.n_eval
            island_source_population_size = len(self.candidate_pool)
            progress = self.n_eval / self.case.max_fes
            phase = "aggregation" if progress < self.config.switch_ratio else "pareto"
            generations = (
                self.config.inner_generations_early
                if phase == "aggregation"
                else self.config.inner_generations_late
            )
            reuse_islands = self.config.retain_island_state and bool(islands)
            if not reuse_islands:
                definitions = island_definitions or self._island_definitions()
                islands, weights = self._create_islands(definitions)
            fe_after_expansion = self.n_eval
            for _ in range(generations):
                for i in range(len(islands)):
                    islands[i] = self._evolve_island(islands[i], weights[i], phase)
                    if self.remaining <= 0:
                        break
                if self.remaining <= 0:
                    break
            fe_after_island_evolution = self.n_eval
            recombination_ratio = self.config.recombination_budget_ratio
            if (
                phase == "pareto"
                and self.config.late_recombination_budget_ratio is not None
            ):
                recombination_ratio = self.config.late_recombination_budget_ratio
            recombined = self._recombine(islands, weights, recombination_ratio)
            fe_after_recombination = self.n_eval
            island_pool = _merge(*islands)
            merged_pool = _merge(self.candidate_pool, self.origin, island_pool, recombined)
            # 这是无额外评价的环境选择，不是旧版的专属 PF 变异扩展。
            self.candidate_pool = self._outer_select(merged_pool, self.pop_size)
            self.origin = self._outer_select(self.candidate_pool, self.n_origin)
            outer += 1

            record = {
                "outer_iteration": outer,
                "phase": phase,
                "fe_start": fe_start,
                "fe_end": self.n_eval,
                "expansion_fes": fe_after_expansion - fe_start,
                "island_evolution_fes": (
                    fe_after_island_evolution - fe_after_expansion
                ),
                "island_fes": fe_after_island_evolution - fe_start,
                "recombination_fes": (
                    fe_after_recombination - fe_after_island_evolution
                ),
                "recombination_offspring": len(recombined),
                "recombination_budget_ratio": recombination_ratio,
                "island_initialization": self.config.island_initialization,
                "island_source_population_size": island_source_population_size,
                "origin_population_size": len(self.origin),
                "candidate_population_size": len(self.candidate_pool),
                "island_state_reused": reuse_islands,
            }
            if self.on_outer_selection is not None:
                event_metrics = self.on_outer_selection(self.n_eval, self.origin)
                if event_metrics:
                    for name in (
                        "igd_plus", "gd_plus", "hv", "spacing", "onvg", "nd_ratio"
                    ):
                        if name in event_metrics:
                            record[name] = event_metrics[name]
            self.outer_records.append(record)

        final = self._outer_select(self.candidate_pool, self.pop_size)
        return final, outer
