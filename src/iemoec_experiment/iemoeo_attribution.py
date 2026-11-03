"""同一 S3 支撑层中的归因核心；旧 S3-Elite 和 founder 实现不变。"""
from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations

import numpy as np
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .config import IEMOECConfig
from .directional_memory import DirectionalMemory
from .iemoec import IEMOECRunner, _merge
from .iemoeo_founder import IEMOEORunner as FounderRunner, variable_blocks
from .normalization import ObjectiveNormalization
from .initialization import shared_initial_decisions


class AttributionRunner(IEMOECRunner):
    """共享初始化/算子/缓存/环境选择，只改变已登记因素。"""

    def __init__(self, problem, case, initial_X=None, on_checkpoint=None,
                 on_outer_selection=None, on_evaluation=None):
        case.validate()
        self.options = case.iemoeo
        support = IEMOECConfig.for_variant(
            "s3_elite", origin_ratio=1.,
            protect_direction_elites=self.options.direction_elite_protection,
        )
        super().__init__(problem, replace(case, iemoec=support), initial_X,
                         on_checkpoint, on_outer_selection, on_evaluation)
        self.case = case
        if case.max_fes < self.pop_size:
            raise ValueError("归因实验预算不能小于公共初始种群 N")
        if self.initial_X is None:
            self.initial_X = shared_initial_decisions(problem, self.pop_size, case.seed)
        if self.initial_X.shape != (self.pop_size, problem.n_var):
            raise ValueError("归因实验要求完全相同的 N 个初始决策")
        self.blocks = variable_blocks(problem, case.normalized_problem,
                                      self.options.block_mode, self.options.block_size, case.seed)
        self.weights = np.full(problem.n_obj, 1 / problem.n_obj)
        self.initial_evaluations = 0

    @property
    def combination_method(self):
        return self.options.combination_method

    def _mutate(self, decisions):
        population = Population.new("X", np.asarray(decisions).copy())
        return self.mutation.do(self.problem, population, inplace=False,
                                random_state=self.rng).get("X")

    def _local_search(self, pools, weights, normalization, quota):
        """同步更新：每个子步的共享池冻结，隔离时只在本方向池交配。"""
        offspring = Population.empty()
        foreign_count = 0
        for step in range(self.options.attribution_local_steps):
            remaining_quota = quota - len(offspring)
            step_quota = math.ceil(remaining_quota / (self.options.attribution_local_steps - step))
            base, extra = divmod(step_quota, len(pools))
            shared = _merge(*pools)
            updated = list(pools)
            offset = 0
            for direction, island in enumerate(pools):
                count = base + (direction < extra)
                if count:
                    parents = island if self.options.isolation else shared
                    pairs = self.rng.integers(len(parents), size=(max(1, math.ceil(count / 2)), 2))
                    if not self.options.isolation:
                        foreign_count += int(np.sum((pairs < offset) | (pairs >= offset + len(island))))
                    children = self._produce_unique_offspring(parents, pairs, count,
                                                              "isolated" if self.options.isolation else "shared")
                    candidates = _merge(island, children)
                    scores = normalization.tchebycheff(candidates.get("F"), weights[direction])
                    updated[direction] = candidates[np.argsort(scores, kind="stable")[:len(island)]]
                    offspring = _merge(offspring, children)
                offset += len(island)
            pools = updated
        if len(offspring) != quota:
            raise RuntimeError("局部阶段没有消耗登记的新增 FE 配额")
        return pools, offspring, foreign_count

    def _representatives(self, pools, weights, normalization, iteration):
        representatives = []
        for direction, pool in enumerate(pools):
            if self.options.representative_selection == "local_random":
                index = int(self.rng.integers(len(pool)))
            else:
                score = normalization.tchebycheff(pool.get("F"), weights[direction])
                index = int(np.argmin(score))
            member = pool[index].copy()
            member.set("source_island", direction)
            representatives.append(member)
        # 固定轮转槽位，不依据目标值挑选方向；控制亲本数量上限。
        count = min(len(representatives), self.options.parent_pool_limit)
        slots = (np.linspace(0, len(representatives) - 1, count, dtype=int) + iteration) % len(representatives)
        return Population.create(*(representatives[i] for i in slots))

    def _probe_representatives(self, representatives, weights, normalization, quota, iteration):
        """有限坐标邻域检查；缓存计 0 FE，未完整探测者不视为合格。"""
        accepted = np.zeros(len(representatives), dtype=bool)
        tested_count = 0
        probes = Population.empty()
        order = np.roll(np.arange(len(representatives)), iteration % len(representatives))
        radius = self.options.principle_probe_radius * (self.problem.xu - self.problem.xl)
        for index in order:
            representative = representatives[index]
            decisions = []
            for coordinate in range(self.problem.n_var):
                for sign in (-1, 1):
                    x = representative.X.copy()
                    x[coordinate] = np.clip(x[coordinate] + sign * radius[coordinate],
                                             self.problem.xl[coordinate], self.problem.xu[coordinate])
                    if not np.array_equal(x, representative.X):
                        decisions.append(x)
            keys = {self._x_key(x): x for x in decisions}
            available = quota - len(probes)
            fresh = [x for key, x in keys.items() if key not in self.evaluated_X][:available]
            if fresh:
                evaluated = self._evaluate_unique(Population.new("X", np.asarray(fresh)))
                probes = _merge(probes, evaluated)
            complete = bool(keys) and all(key in self.evaluated_X for key in keys)
            if complete:
                tested_count += 1
                weight = weights[int(representative.get("source_island"))]
                old = normalization.tchebycheff(representative.F[None, :], weight)[0]
                values = np.asarray([self.evaluated_X[key] for key in keys])
                accepted[index] = not np.any(normalization.tchebycheff(values, weight) < old - 1e-12)
        return accepted, probes, tested_count

    def _combine(self, parents, quota):
        """算子对照保持组合新增 FE 相同；亲本不足时明确转给局部阶段。"""
        method = self.options.combination_method
        if method == "none" or len(parents) < 2:
            return Population.empty()
        possible = [(a, b) for a, b in combinations(range(len(parents)), 2)
                    if not np.array_equal(parents[a].X, parents[b].X)]
        if not possible:
            return Population.empty()
        count = min(len(possible), self.options.combination_pairs_limit)
        chosen = self.rng.choice(len(possible), size=count, replace=False)
        pairs = np.asarray([possible[index] for index in chosen], dtype=int)
        if method == "sbx_only":
            return self._produce_unique_offspring(parents, pairs, quota, "combination")
        children = Population.empty()
        stalled = 0
        while len(children) < quota:
            needed = quota - len(children)
            selected = pairs[self.rng.integers(len(pairs), size=needed)]
            decisions = [FounderRunner._inherit(self, parents[a].X, parents[b].X, method == "structured")
                         for a, b in selected]
            evaluated = self._evaluate_unique(Population.new("X", self._mutate(decisions)))
            children = _merge(children, evaluated)
            stalled = stalled + 1 if not len(evaluated) else 0
            if stalled >= 50:
                raise RuntimeError("组合连续 50 次未产生新决策，不虚构 FE")
        return children

    def run(self):
        self._initialize_candidate()
        self.initial_evaluations = self.n_eval
        definitions = self._island_definitions()
        weights = [weight for weight, _ in definitions]
        capacity = max(2, math.ceil(self.pop_size / len(weights)))
        memory = None
        iteration = 0
        while self.remaining:
            start = self.n_eval
            normalization = self._normalization_for(self.candidate_pool)
            if memory is None or not self.options.persistent_direction_memory:
                memory = DirectionalMemory(self.problem, weights, capacity)
                memory.update(self.candidate_pool, normalization)
            pools = memory.parent_pools()
            initial_memory = memory.combined_population()
            batch = min(max(1, round(self.pop_size * self.options.attribution_batch_ratio)), self.remaining)
            combination_quota = batch // 4
            probe_quota = (int(batch * self.options.qualification_fe_ratio)
                           if self.options.qualification_mode != "none" else 0)
            local_quota = batch - combination_quota - probe_quota
            pools, local, foreign = self._local_search(pools, weights, normalization, local_quota)
            representatives = self._representatives(pools, weights, normalization, iteration)
            qualified = np.ones(len(representatives), dtype=bool)
            probes = Population.empty()
            tested = 0
            if self.options.qualification_mode != "none":
                qualified, probes, tested = self._probe_representatives(
                    representatives, weights, normalization, probe_quota, iteration)
            checked = self.options.qualification_mode == "probe_checked"
            eligible = representatives[qualified] if checked else representatives
            combined = self._combine(eligible, combination_quota)
            # 探测缓存、亲本不足、none 等情形的余额均按原 I 规则转给局部搜索。
            balance = batch - (self.n_eval - start)
            redirected = Population.empty()
            if balance:
                pools, redirected, additional_foreign = self._local_search(pools, weights, normalization, balance)
                foreign += additional_foreign
            merged = _merge(self.candidate_pool, initial_memory, *pools, local, probes, combined, redirected)
            unique, _, _ = self._deduplicate_population(merged)
            selection_normalization = self._normalization_for(unique)
            self.candidate_pool, _, protected = self._select_s3_population(merged, weights, selection_normalization)
            if self.options.persistent_direction_memory:
                memory.update(unique, selection_normalization)
            self.origin = self.candidate_pool.copy()
            iteration += 1
            if self.n_eval - start != batch or len(self.candidate_pool) != self.pop_size:
                raise RuntimeError("归因实验的 FE 或输出容量不守恒")
            child_keys = {self._x_key(ind.X) for ind in combined}
            before_keys = {self._x_key(ind.X) for ind in representatives}
            eligible_keys = {self._x_key(ind.X) for ind in eligible}
            front = NonDominatedSorting().do(self.candidate_pool.get("F"), only_non_dominated_front=True)
            record = {
                "outer_iteration": iteration, "phase": "s3_scaffold_attribution",
                "fe_start": start, "fe_end": self.n_eval, "outer_batch_fes": batch,
                "expansion_fes": 0, "island_evolution_fes": batch - len(combined),
                "island_fes": batch - len(combined), "recombination_offspring": len(combined),
                "recombination_candidates": len(combined), "founder_count": len(representatives),
                "island_count": len(pools),
                "representative_count": len(representatives), "qualified_parent_count": int(qualified.sum()) if checked else len(representatives),
                "parent_acceptance_rate": float(qualified.mean()) if checked else 1.,
                "eligible_parent_count": len(eligible), "parent_pool_size": len(eligible),
                "parent_fallback": False, "foreign_parent_count": foreign,
                "representative_selection": self.options.representative_selection,
                "representative_source_islands": len(set(representatives.get("source_island"))),
                "unique_representative_count": len({self._x_key(ind.X) for ind in representatives}),
                "parent_pool_changed_by_check": checked and before_keys != eligible_keys,
                "combination_budget_requested": combination_quota,
                "combination_reallocated_fes": combination_quota - len(combined),
                "extra_local_generation_calls": int(bool(balance)),
                "combination_unique_candidates": len(child_keys),
                "combination_children_selected": sum(self._x_key(ind.X) in child_keys for ind in self.candidate_pool),
                "combination_children_nondominated": sum(self._x_key(self.candidate_pool[i].X) in child_keys for i in front),
                "survival_method": "nsga3", "qualification_mode": self.options.qualification_mode,
                "probe_fes": len(probes), "probe_budget_requested": probe_quota,
                "probe_complete_representatives": tested, "probe_qualified_representatives": int(qualified.sum()) if self.options.qualification_mode != "none" else 0,
                "local_search_fes": len(local) + len(redirected),
                "persistent_direction_memory": self.options.persistent_direction_memory,
                "memory_unique_size": len(memory.combined_population()),
                "protected_elite_count": len(protected),
                "origin_population_size": self.pop_size, "front_size": len(front),
            }
            if self.on_outer_selection is not None:
                record.update(self.on_outer_selection(self.n_eval, self.candidate_pool) or {})
            self.outer_records.append(record)
        FounderRunner._select_base(self)
        return self.candidate_pool, iteration


def attribution_metrics(metrics: dict, options) -> dict:
    """补全新架构元数据；不修改公共入口的旧算法执行路径或科学摘要。"""
    certificates = {"none": "not_tested", "probe_checked": "finite_coordinate_neighborhood_only",
                    "probe_unchecked": "probed_but_not_required"}
    return {**metrics, "island_initialization": "direction_memory_or_current_population",
            "extremum_certificate": certificates[options.qualification_mode]}
