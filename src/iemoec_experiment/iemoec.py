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
from .directions import direction_objective, reference_direction_subset
from .factory import reference_directions
from .normalization import ObjectiveNormalization


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
        initial_X: np.ndarray | None = None,
        on_checkpoint: Callable[[int, Population], None] | None = None,
        on_outer_selection: Callable[[int, Population], dict | None] | None = None,
    ):
        self.problem = problem
        self.case = case
        self.config = case.iemoec
        self.rng = np.random.default_rng(case.seed)
        self.initial_X = None if initial_X is None else np.asarray(initial_X, dtype=float)
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
        self.evaluated_X: dict[bytes, np.ndarray] = {}
        self.global_selection_count = 0

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

    def _front_truncate(
        self,
        pop: Population,
        n_survive: int,
        weight=None,
        normalization: ObjectiveNormalization | None = None,
    ) -> Population:
        if len(pop) <= n_survive:
            return pop
        if self.config.use_crowding and self.config.variant in ("v0", "s1"):
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
                if normalization is None:
                    ideal = np.min(pop.get("F"), axis=0)
                    score = np.max(weight * np.abs(F - ideal), axis=1)
                else:
                    score = normalization.tchebycheff(F, weight)
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
        if self.config.outer_survival == "rank_crowding":
            return RankAndCrowding().do(
                self.problem,
                pop,
                n_survive=n_survive,
                random_state=self.rng,
            )
        return self._front_truncate(pop, n_survive)

    def _initialize(self) -> None:
        n = min(self.n_origin, self.remaining)
        if self.initial_X is None:
            self.origin = self.sampling.do(self.problem, n, random_state=self.rng)
        else:
            self.origin = Population.new("X", self.initial_X[:n].copy())
        self.origin = self._evaluate(self.origin)
        self.candidate_pool = self.origin

    def _island_definitions(self):
        """创建一次岛身份，使状态消融不受权重重采样干扰。"""
        if self.config.island_direction_mode == "reference_subset":
            island_count = self.problem.n_obj * self.config.island_count_multiplier
            weights = reference_direction_subset(
                self.ref_dirs,
                self.problem.n_obj,
                island_count,
            )
            return [(weight, direction_objective(weight)) for weight in weights]

        specs = []
        for m in range(self.problem.n_obj):
            weight = np.full(self.problem.n_obj, self.config.aggregation_epsilon)
            weight[m] = 1.0
            specs.append((weight, m))
        if self.config.variant in ("v0", "s1"):
            island_count = self.problem.n_obj * self.config.islands_per_objective
        else:
            island_count = self.problem.n_obj * self.config.island_count_multiplier
        random_count = island_count - self.problem.n_obj
        if random_count > 0:
            weights = self.rng.dirichlet(
                np.ones(self.problem.n_obj),
                random_count,
            )
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
        normalization: ObjectiveNormalization | None = None,
    ) -> np.ndarray:
        F = population.get("F")
        if objective is not None:
            if normalization is None:
                return F[:, objective]
            return normalization.apply(F)[:, objective]
        if normalization is not None:
            return normalization.tchebycheff(F, weight)
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
        normalization: ObjectiveNormalization | None = None,
    ) -> Population:
        """从已评价的全局候选池周期性建立一个多祖先岛。"""
        source = self.candidate_pool
        target = min(self.config.island_population, len(source))
        if target == 0:
            return Population.empty()

        scores = self._direction_scores(source, weight, objective, normalization)
        ranked_all = np.argsort(scores, kind="stable")
        if self.config.variant == "candidate":
            anchor_index = self._ancestor_index_from_population(
                self.origin,
                source,
                weight,
                objective,
                normalization,
            )
            selected = [anchor_index]
            seen_keys = {self._x_key(source[anchor_index].get("X"))}
            ranked_candidates = []
            for index in ranked_all:
                key = self._x_key(source[int(index)].get("X"))
                if key not in seen_keys:
                    ranked_candidates.append(int(index))
                    seen_keys.add(key)
            ranked_candidates = np.asarray(ranked_candidates, dtype=int)
            target = min(target, 1 + len(ranked_candidates))
        else:
            selected = [int(ranked_all[0])]
            ranked_candidates = ranked_all[1:]

        neighbor_count = min(
            self.config.direction_neighbor_ancestors,
            target - len(selected),
        )
        selected.extend(int(index) for index in ranked_candidates[:neighbor_count])

        normalized_x = self._decision_space_points(source)
        diverse_count = min(
            self.config.diverse_ancestors,
            target - len(selected),
        )
        for _ in range(diverse_count):
            available = np.setdiff1d(
                ranked_candidates,
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
                ranked_candidates[:direction_pool_size],
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
                ranked_candidates,
                np.asarray(selected, dtype=int),
                assume_unique=False,
            )
            chosen = self.rng.choice(available, size=remaining_slots, replace=False)
            selected.extend(int(index) for index in np.atleast_1d(chosen))

        return Population.create(*(source[index].copy() for index in selected))

    def _ancestor_index_from_population(
        self,
        anchor_population: Population,
        source_population: Population,
        weight: np.ndarray,
        objective: int | None,
        normalization: ObjectiveNormalization | None,
    ) -> int:
        scores = self._direction_scores(
            anchor_population,
            weight,
            objective,
            normalization,
        )
        anchor = anchor_population[int(np.argmin(scores))]
        anchor_key = self._x_key(anchor.get("X"))
        for index, individual in enumerate(source_population):
            if self._x_key(individual.get("X")) == anchor_key:
                return index
        raise RuntimeError("origin anchor 不在 candidate pool 中")

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

    def _create_islands(
        self,
        definitions,
        normalization: ObjectiveNormalization | None = None,
    ) -> tuple[list[Population], list[np.ndarray]]:
        islands = []
        weights = []
        for weight, objective in definitions:
            if self.config.island_initialization == "multi_ancestor":
                island = self._multi_ancestor_island(
                    weight,
                    objective,
                    normalization,
                )
            else:
                ancestor_idx = self._ancestor_index(weight, objective)
                island = self._expand_island(self.origin[ancestor_idx])
            islands.append(island)
            weights.append(weight)
            if self.remaining <= 0:
                break
        return islands, weights

    def _evolve_island(
        self,
        island: Population,
        weight,
        phase: str,
    ) -> tuple[Population, Population]:
        n = len(island)
        if n < 2 or self.remaining <= 0:
            return island, Population.empty()
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
            return merged[np.argsort(scores)[:n]], offspring
        return self._front_truncate(merged, n, weight=weight), offspring

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

    @staticmethod
    def _x_key(x: np.ndarray) -> bytes:
        return np.ascontiguousarray(x, dtype=np.float64).tobytes()

    def _evaluate_unique(self, population: Population) -> Population:
        """仅评价此前未出现的决策向量。"""
        selected = []
        pending_keys: set[bytes] = set()
        for individual in population:
            key = self._x_key(individual.get("X"))
            if key in self.evaluated_X or key in pending_keys:
                continue
            selected.append(individual)
            pending_keys.add(key)
            if len(selected) >= self.remaining:
                break
        if not selected:
            return Population.empty()
        new_population = Population.create(*selected)
        new_population = self._evaluate(new_population)
        for individual in new_population:
            key = self._x_key(individual.get("X"))
            self.evaluated_X[key] = np.asarray(individual.get("F"), dtype=float).copy()
        return new_population

    def _initialize_candidate(self) -> None:
        if self.initial_X is None:
            raise ValueError("candidate 需要公共 initial_X")
        if self.remaining < self.pop_size:
            raise ValueError("candidate 的 MaxFEs 不能小于公共初始种群 N")
        initial = Population.new("X", self.initial_X.copy())
        initial.set("provenance", np.full(len(initial), "initial", dtype=object))
        self.candidate_pool = self._evaluate_unique(initial)
        self.origin = self._update_origin(self.candidate_pool, Population.empty())

    @staticmethod
    def _mean_pairwise_distance(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
        upper = distances[np.triu_indices(len(values), k=1)]
        return float(np.mean(upper)) if len(upper) else 0.0

    def _founder_statistics(
        self,
        islands: list[Population],
        normalization: ObjectiveNormalization,
    ) -> dict[str, float]:
        if not islands:
            return {
                "founders_per_island": 0.0,
                "unique_founder_ratio": 0.0,
                "mean_founder_distance_x": 0.0,
                "mean_founder_distance_f": 0.0,
            }
        unique_ratios = []
        x_distances = []
        f_distances = []
        for island in islands:
            keys = [self._x_key(individual.get("X")) for individual in island]
            unique_ratios.append(len(set(keys)) / max(1, len(keys)))
            x_distances.append(
                self._mean_pairwise_distance(self._decision_space_points(island))
            )
            f_distances.append(
                self._mean_pairwise_distance(normalization.apply(island.get("F")))
            )
        return {
            "founders_per_island": float(np.mean([len(island) for island in islands])),
            "unique_founder_ratio": float(np.mean(unique_ratios)),
            "mean_founder_distance_x": float(np.mean(x_distances)),
            "mean_founder_distance_f": float(np.mean(f_distances)),
        }

    def _produce_unique_offspring(
        self,
        parent_pool: Population,
        pairs: np.ndarray,
        budget: int,
        provenance: str,
    ) -> Population:
        offspring = Population.empty()
        attempts_without_progress = 0
        while len(offspring) < budget and self.remaining > 0:
            needed = min(budget - len(offspring), self.remaining)
            mating_count = math.ceil(needed / 2)
            repeated_pairs = np.resize(pairs, (mating_count, 2))
            self.rng.shuffle(repeated_pairs)
            children = self._mate(parent_pool, repeated_pairs, needed)
            children.set(
                "provenance",
                np.full(len(children), provenance, dtype=object),
            )
            evaluated = self._evaluate_unique(children)
            if len(evaluated) == 0:
                attempts_without_progress += 1
                if attempts_without_progress >= 50:
                    raise RuntimeError("连续生成重复决策向量，无法消耗剩余 FE")
                continue
            attempts_without_progress = 0
            offspring = _merge(offspring, evaluated)
        return offspring

    def _evolve_islands_with_budget(
        self,
        islands: list[Population],
        weights: list[np.ndarray],
        phase: str,
        budget: int,
        normalization: ObjectiveNormalization,
    ) -> tuple[list[Population], Population]:
        base, extra = divmod(budget, max(1, len(islands)))
        all_offspring = Population.empty()
        for island_id, island in enumerate(islands):
            quota = base + (1 if island_id < extra else 0)
            if quota == 0:
                continue
            order = self.rng.permutation(len(island))
            if len(order) % 2:
                order = np.append(order, order[0])
            pairs = order.reshape(-1, 2)
            offspring = self._produce_unique_offspring(
                island,
                pairs,
                quota,
                "local",
            )
            merged = _merge(island, offspring)
            if phase == "aggregation":
                scores = normalization.tchebycheff(
                    merged.get("F"),
                    weights[island_id],
                )
                islands[island_id] = merged[np.argsort(scores)[:len(island)]]
            else:
                islands[island_id] = self._front_truncate(
                    merged,
                    len(island),
                    weight=weights[island_id],
                    normalization=normalization,
                )
            all_offspring = _merge(all_offspring, offspring)
        return islands, all_offspring

    def _island_representatives(
        self,
        islands: list[Population],
        weights: list[np.ndarray],
        normalization: ObjectiveNormalization,
    ) -> Population:
        representatives = []
        for island, weight in zip(islands, weights):
            if not len(island):
                continue
            scores = normalization.tchebycheff(island.get("F"), weight)
            representatives.append(island[int(np.argmin(scores))].copy())
        return Population.create(*representatives) if representatives else Population.empty()

    def _representative_pairs(
        self,
        representatives: Population,
        weights: list[np.ndarray],
    ) -> np.ndarray:
        strategy = self.config.pairing_strategy
        if strategy == "none" or len(representatives) < 2:
            return np.empty((0, 2), dtype=int)
        unit_weights = np.asarray(weights, dtype=float)
        unit_weights /= np.maximum(
            np.linalg.norm(unit_weights, axis=1, keepdims=True),
            1e-12,
        )
        normalized_x = self._decision_space_points(representatives)
        pair_set: set[tuple[int, int]] = set()
        for i in range(len(representatives)):
            candidates = np.delete(np.arange(len(representatives)), i)
            count = min(self.config.partners_per_elite, len(candidates))
            if strategy == "random":
                ordered = self.rng.permutation(candidates)
            elif strategy == "farthest_decision":
                distances = np.linalg.norm(
                    normalized_x[candidates] - normalized_x[i],
                    axis=1,
                )
                ordered = candidates[np.argsort(-distances, kind="stable")]
            else:
                similarities = unit_weights[candidates] @ unit_weights[i]
                sign = -1.0 if strategy == "nearest_weight" else 1.0
                ordered = candidates[np.argsort(sign * similarities, kind="stable")]
            for j in ordered[:count]:
                pair_set.add(tuple(sorted((i, int(j)))))
        return np.asarray(sorted(pair_set), dtype=int)

    def _recombine_with_budget(
        self,
        representatives: Population,
        weights: list[np.ndarray],
        budget: int,
    ) -> Population:
        if budget == 0:
            return Population.empty()
        pairs = self._representative_pairs(representatives, weights)
        if len(pairs) == 0:
            raise RuntimeError("重组预算大于 0，但 pairing strategy 没有产生父代对")
        return self._produce_unique_offspring(
            representatives,
            pairs,
            budget,
            "recombination",
        )

    def _deduplicate_population(
        self,
        population: Population,
    ) -> tuple[Population, list[int], float]:
        unique_indices = []
        duplicate_indices = []
        seen: set[bytes] = set()
        for index, individual in enumerate(population):
            key = self._x_key(individual.get("X"))
            if key in seen:
                duplicate_indices.append(index)
            else:
                unique_indices.append(index)
                seen.add(key)
        unique = population[np.asarray(unique_indices, dtype=int)]
        ratio = len(unique) / max(1, len(population))
        return unique, duplicate_indices, ratio

    def _global_select_once(
        self,
        merged: Population,
    ) -> tuple[Population, float]:
        unique, duplicate_indices, unique_ratio = self._deduplicate_population(merged)
        selection_input = unique
        if len(selection_input) < self.pop_size and duplicate_indices:
            needed = min(self.pop_size - len(selection_input), len(duplicate_indices))
            fillers = merged[np.asarray(duplicate_indices[:needed], dtype=int)]
            selection_input = _merge(selection_input, fillers)
        selected = self._outer_select(selection_input, self.pop_size)
        self.global_selection_count += 1
        return selected, unique_ratio

    def _update_origin(
        self,
        population: Population,
        representatives: Population,
    ) -> Population:
        population_keys = {
            self._x_key(individual.get("X")): index
            for index, individual in enumerate(population)
        }
        preferred_indices = []
        seen: set[int] = set()
        for representative in representatives:
            index = population_keys.get(self._x_key(representative.get("X")))
            if index is not None and index not in seen:
                preferred_indices.append(index)
                seen.add(index)
            if len(preferred_indices) >= self.n_origin:
                break
        preferred = (
            population[np.asarray(preferred_indices, dtype=int)]
            if preferred_indices
            else Population.empty()
        )
        needed = self.n_origin - len(preferred)
        if needed <= 0:
            return preferred
        remaining_indices = [
            index for index in range(len(population)) if index not in seen
        ]
        remaining = population[np.asarray(remaining_indices, dtype=int)]
        fill = RankAndCrowding().do(
            self.problem,
            remaining,
            n_survive=min(needed, len(remaining)),
            random_state=self.rng,
        )
        return _merge(preferred, fill)

    def _survival_rate(
        self,
        offspring: Population,
        selected: Population,
    ) -> float:
        if not len(offspring):
            return 0.0
        selected_keys = {
            self._x_key(individual.get("X")) for individual in selected
        }
        survived = sum(
            self._x_key(individual.get("X")) in selected_keys
            for individual in offspring
        )
        return survived / len(offspring)

    def _unique_ratio(self, population: Population) -> float:
        if not len(population):
            return 0.0
        keys = {self._x_key(individual.get("X")) for individual in population}
        return len(keys) / len(population)

    def _direction_coverage(
        self,
        population: Population,
        normalization: ObjectiveNormalization,
    ) -> tuple[float, float]:
        values = np.maximum(normalization.apply(population.get("F")), 0.0)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        directions = self.ref_dirs / np.maximum(
            np.linalg.norm(self.ref_dirs, axis=1, keepdims=True),
            1e-12,
        )
        assigned = np.argmax(values @ directions.T, axis=1)
        occupancy = len(np.unique(assigned)) / len(directions)
        return occupancy, 1.0 - occupancy

    def _direction_improvement_rate(
        self,
        offspring: Population,
        representatives: Population,
        weights: list[np.ndarray],
        normalization: ObjectiveNormalization,
    ) -> float:
        if not len(offspring) or not len(representatives):
            return 0.0
        representative_scores = np.asarray([
            normalization.tchebycheff(
                representatives[i:i + 1].get("F"),
                weight,
            )[0]
            for i, weight in enumerate(weights[:len(representatives)])
        ])
        improved = 0
        for child in offspring:
            child_scores = np.asarray([
                normalization.tchebycheff(
                    np.asarray(child.get("F"), dtype=float)[None, :],
                    weight,
                )[0]
                for weight in weights[:len(representatives)]
            ])
            improved += int(np.any(child_scores < representative_scores - 1e-12))
        return improved / len(offspring)

    def _run_candidate(self) -> tuple[Population, int]:
        self._initialize_candidate()
        definitions = self._island_definitions()
        outer = 0
        while self.remaining > 0 and len(self.candidate_pool):
            fe_start = self.n_eval
            island_source_population_size = len(self.candidate_pool)
            normalization = ObjectiveNormalization.from_objectives(
                self.candidate_pool.get("F")
            )
            progress = self.n_eval / self.case.max_fes
            phase = "aggregation" if progress < self.config.switch_ratio else "pareto"
            islands, weights = self._create_islands(definitions, normalization)
            founder_statistics = self._founder_statistics(islands, normalization)

            outer_batch_fes = min(
                max(1, round(self.pop_size * self.config.outer_batch_ratio)),
                self.remaining,
            )
            local_budget = round(outer_batch_fes * self.config.local_fe_ratio)
            recombination_budget = outer_batch_fes - local_budget
            islands, local_offspring = self._evolve_islands_with_budget(
                islands,
                weights,
                phase,
                local_budget,
                normalization,
            )
            representatives = self._island_representatives(
                islands,
                weights,
                normalization,
            )
            recombined = self._recombine_with_budget(
                representatives,
                weights,
                recombination_budget,
            )
            island_pool = _merge(*islands)
            merged = _merge(
                self.candidate_pool,
                self.origin,
                island_pool,
                local_offspring,
                recombined,
            )
            self.candidate_pool, merged_unique_ratio = self._global_select_once(merged)
            self.origin = self._update_origin(self.candidate_pool, representatives)
            occupancy, empty_ratio = self._direction_coverage(
                self.candidate_pool,
                normalization,
            )
            outer += 1
            record = {
                "outer_iteration": outer,
                "phase": phase,
                "fe_start": fe_start,
                "fe_end": self.n_eval,
                "outer_batch_fes": self.n_eval - fe_start,
                "island_count": len(islands),
                **founder_statistics,
                "merged_unique_ratio": merged_unique_ratio,
                "local_offspring": len(local_offspring),
                "recombination_offspring": len(recombined),
                "recombination_unique_ratio": self._unique_ratio(recombined),
                "local_survival_rate": self._survival_rate(
                    local_offspring,
                    self.candidate_pool,
                ),
                "recombination_survival_rate": self._survival_rate(
                    recombined,
                    self.candidate_pool,
                ),
                "direction_improvement_rate": self._direction_improvement_rate(
                    recombined,
                    representatives,
                    weights,
                    normalization,
                ),
                "direction_occupancy": occupancy,
                "empty_direction_ratio": empty_ratio,
                "objective_extreme_count": sum(
                    objective is not None for _, objective in definitions
                ),
                "origin_population_size": len(self.origin),
                "candidate_population_size": len(self.candidate_pool),
                "island_initialization": self.config.island_initialization,
                "island_source_population_size": island_source_population_size,
                "expansion_fes": 0,
                "island_evolution_fes": len(local_offspring),
                "island_fes": len(local_offspring),
                "recombination_fes": len(recombined),
                "recombination_budget_ratio": self.config.recombination_fe_ratio,
                "island_state_reused": False,
            }
            if self.on_outer_selection is not None:
                event_metrics = self.on_outer_selection(self.n_eval, self.candidate_pool)
                if event_metrics:
                    record.update(event_metrics)
            self.outer_records.append(record)
        return self.candidate_pool, outer

    def _run_legacy(self) -> tuple[Population, int]:
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
            diagnostic_normalization = ObjectiveNormalization.from_objectives(
                self.candidate_pool.get("F")
            )
            if self.config.island_initialization == "multi_ancestor":
                founder_statistics = self._founder_statistics(
                    islands,
                    diagnostic_normalization,
                )
            else:
                founder_statistics = {
                    "founders_per_island": 1.0,
                    "unique_founder_ratio": 1.0,
                    "mean_founder_distance_x": 0.0,
                    "mean_founder_distance_f": 0.0,
                }
            fe_after_expansion = self.n_eval
            local_offspring = Population.empty()
            for _ in range(generations):
                for i in range(len(islands)):
                    islands[i], offspring = self._evolve_island(
                        islands[i],
                        weights[i],
                        phase,
                    )
                    local_offspring = _merge(local_offspring, offspring)
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
            _, _, merged_unique_ratio = self._deduplicate_population(merged_pool)
            # 这是无额外评价的环境选择，不是旧版的专属 PF 变异扩展。
            self.candidate_pool = self._outer_select(merged_pool, self.pop_size)
            self.global_selection_count += 1
            self.origin = self._outer_select(self.candidate_pool, self.n_origin)
            self.global_selection_count += 1
            selected_normalization = ObjectiveNormalization.from_objectives(
                self.candidate_pool.get("F")
            )
            occupancy, empty_ratio = self._direction_coverage(
                self.candidate_pool,
                selected_normalization,
            )
            outer += 1

            record = {
                "outer_iteration": outer,
                "phase": phase,
                "fe_start": fe_start,
                "fe_end": self.n_eval,
                "outer_batch_fes": self.n_eval - fe_start,
                "island_count": len(islands),
                **founder_statistics,
                "merged_unique_ratio": merged_unique_ratio,
                "expansion_fes": fe_after_expansion - fe_start,
                "island_evolution_fes": (
                    fe_after_island_evolution - fe_after_expansion
                ),
                "island_fes": fe_after_island_evolution - fe_start,
                "recombination_fes": (
                    fe_after_recombination - fe_after_island_evolution
                ),
                "local_offspring": len(local_offspring),
                "recombination_offspring": len(recombined),
                "recombination_unique_ratio": self._unique_ratio(recombined),
                "local_survival_rate": self._survival_rate(
                    local_offspring,
                    self.candidate_pool,
                ),
                "recombination_survival_rate": self._survival_rate(
                    recombined,
                    self.candidate_pool,
                ),
                "direction_occupancy": occupancy,
                "empty_direction_ratio": empty_ratio,
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
        self.global_selection_count += 1
        return final, outer

    def run(self) -> tuple[Population, int]:
        if self.config.variant == "candidate":
            return self._run_candidate()
        return self._run_legacy()
