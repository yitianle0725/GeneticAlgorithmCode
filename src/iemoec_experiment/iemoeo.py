"""IE-MOEO：重建小群体、独立局部竞争、邻域亲本筛选和变量块组合。

邻域检查只比较已评价的小群体代表，不产生探测点，也不是驻点证明。
所有目标均按最小化处理；修复组件可由后续网络调度问题替换。
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .factory import make_operators, reference_directions
from .initialization import shared_initial_decisions
from .normalization import ObjectiveNormalization
from .problems import standard_problem_dimensions


def clip_to_bounds(problem, X: np.ndarray) -> np.ndarray:
    """连续箱约束的默认修复；可替换为具体问题的可行性修复。"""
    return np.clip(X, problem.xl, problem.xu)


def variable_blocks(problem, problem_name: str, mode: str, count: int) -> list[np.ndarray]:
    """返回互不重叠且覆盖所有坐标的变量块。count 表示块数。"""
    coordinates = np.arange(problem.n_var)
    if mode == "uniform":
        return list(np.array_split(coordinates, min(count, problem.n_var)))
    if mode != "known_separable":
        raise ValueError(f"未知变量分块方式: {mode}")
    name = problem_name.lower().replace("-", "")
    standard_n_var, k, _ = standard_problem_dimensions(name, problem.n_obj)
    if name.startswith("dtlz"):
        # 标准 DTLZ 的后 k 个坐标是距离变量；自定义 n_var 时位置块不变。
        boundary = standard_n_var - k
    elif name.startswith("wfg"):
        boundary = k
    else:
        raise ValueError("known_separable 仅支持 DTLZ/WFG")
    if not 0 < boundary < problem.n_var:
        raise ValueError("位置/距离分块要求两个块都非空")
    return [coordinates[:boundary], coordinates[boundary:]]


def merge_populations(populations) -> Population:
    merged = Population.empty()
    for population in populations:
        if len(population):
            merged = Population.merge(merged, population)
    return merged


class IEMOEORunner:
    def __init__(self, problem, case, initial_X=None,
                 on_checkpoint=None, on_outer_selection=None, on_evaluation=None):
        case.validate()
        if (problem.n_constr or problem.n_var < 2
                or not np.all(np.isfinite(problem.xl))
                or not np.all(np.isfinite(problem.xu))
                or np.any(np.asarray(problem.xu) <= np.asarray(problem.xl))):
            raise ValueError("IE-MOEO 仅支持至少两个变量、有限非零区间的无约束连续问题")
        self.problem = problem
        self.case = case
        self.config = case.iemoeo
        self.ref_dirs = reference_directions(case)
        self.pop_size = len(self.ref_dirs)
        self.n_origin = max(2, math.ceil(self.pop_size * self.config.origin_ratio))
        if case.max_fes < self.n_origin:
            raise ValueError("IE-MOEO max_fes 不得小于起源者数 P")
        if initial_X is None:
            initial_X = shared_initial_decisions(problem, self.n_origin, case.seed)
        self.initial_X = np.asarray(initial_X, dtype=float).copy()
        if self.initial_X.shape != (self.n_origin, problem.n_var):
            raise ValueError("IE-MOEO initial_X 必须恰好包含 P 个起源者")
        self.rng = np.random.default_rng(case.seed)
        self.evaluator = Evaluator()
        operators = make_operators()
        self.sbx = operators["crossover"]
        self.mutation = operators["mutation"]
        self.repair = clip_to_bounds
        self.weights = np.asarray(
            self.config.objective_weights
            if self.config.objective_weights is not None
            else np.full(problem.n_obj, 1 / problem.n_obj), dtype=float,
        )
        self.blocks = variable_blocks(
            problem, case.normalized_problem, self.config.block_mode, self.config.block_size,
        )
        self.on_checkpoint = on_checkpoint
        self.on_outer_selection = on_outer_selection
        self.on_evaluation = on_evaluation
        self.candidate_pool = Population.empty()
        self.founders = Population.empty()
        self.outer_records: list[dict] = []
        self.global_selection_count = 0
        self.initial_evaluations = 0
        self.evaluation_cache: dict[bytes, Individual] = {}
        self.normalization: ObjectiveNormalization | None = None
        self.x_base = None
        self.f_base = None
        self.base_asf = None

    @property
    def n_eval(self) -> int:
        return int(self.evaluator.n_eval)

    @property
    def remaining(self) -> int:
        return max(0, self.case.max_fes - self.n_eval)

    @staticmethod
    def _key(x) -> bytes:
        return np.asarray(x, dtype=np.float64).tobytes()

    def _unique(self, population) -> Population:
        seen = set()
        selected = []
        for i, individual in enumerate(population):
            key = self._key(individual.X)
            if key not in seen:
                selected.append(i)
                seen.add(key)
        return population[np.asarray(selected, dtype=int)]

    def _evaluate(self, X) -> Population:
        """按批次评价新决策；缓存命中保留候选但不重复扣 FE。"""
        values = np.asarray(X, dtype=float)
        if not values.size:
            return Population.empty()
        values = np.asarray(self.repair(self.problem, values), dtype=float)
        if (values.ndim != 2 or values.shape[1] != self.problem.n_var
                or not np.all(np.isfinite(values))
                or np.any(values < self.problem.xl) or np.any(values > self.problem.xu)):
            raise ValueError("修复后的决策必须是有限、维数正确且不越界的矩阵")
        keys = []
        pending = {}
        for x in values:
            key = self._key(x)
            if key not in self.evaluation_cache and key not in pending:
                if len(pending) >= self.remaining:
                    break
                pending[key] = x
            keys.append(key)
        if pending:
            new = Population.new("X", np.asarray(list(pending.values())))
            self.evaluator.eval(self.problem, new)
            if not np.all(np.isfinite(new.get("F"))):
                raise RuntimeError("IE-MOEO 评价产生 NaN/Inf")
            for key, individual in zip(pending, new):
                self.evaluation_cache[key] = individual.copy()
            if self.on_evaluation is not None:
                self.on_evaluation(self.n_eval, new)
            if self.on_checkpoint is not None:
                visible = self.candidate_pool if len(self.candidate_pool) else new
                self.on_checkpoint(self.n_eval, visible)
        return Population.create(*[self.evaluation_cache[key].copy() for key in keys])

    def _mutate(self, X) -> np.ndarray:
        population = Population.new("X", np.asarray(X, dtype=float).copy())
        return self.mutation.do(
            self.problem, population, inplace=False, random_state=self.rng,
        ).get("X")

    def _new_islands(self, founders) -> list[Population]:
        # 每轮只从本轮起源者重新变异 ng 次，不带入旧小群体或保留起源者。
        # 交错排列保证预算尾部依次覆盖起源者，而不偏向第一个小群体。
        X = np.tile(founders.get("X"), (self.config.island_population, 1))
        expanded = self._evaluate(self._mutate(X))
        return [expanded[i::len(founders)] for i in range(len(founders))]

    def _scores(self, population, normalization=None) -> np.ndarray:
        context = normalization if normalization is not None else self.normalization
        if context is None:
            raise RuntimeError("局部竞争前必须建立目标归一化上下文")
        return context.apply(population.get("F")) @ self.weights

    def _local_generation(self, islands) -> tuple[list[Population], int]:
        """同步产生一代后代；隔离关闭时，各小群体使用同一全局交配池。"""
        shared = merge_populations(islands)
        children_X = []
        sizes = []
        foreign_parent_count = 0
        offset = 0
        for island in islands:
            if not len(island):
                sizes.append(0)
                continue
            pool = island if self.config.isolation else shared
            context = self.normalization
            if self.config.normalization_mode == "legacy":
                context = ObjectiveNormalization.from_objectives(pool.get("F"))
            scores = self._scores(pool, context)
            size = self.config.island_population
            # 每对亲本各进行一次二元锦标赛，只按归一化加权和比较。
            contestants = self.rng.integers(len(pool), size=(math.ceil(size / 2), 2, 2))
            winners = np.argmin(scores[contestants], axis=2)
            pairs = np.take_along_axis(contestants, winners[:, :, None], axis=2)[:, :, 0]
            if not self.config.isolation:
                foreign_parent_count += int(np.sum((pairs < offset) | (pairs >= offset + len(island))))
            children = self.sbx.do(self.problem, pool, parents=pairs, random_state=self.rng)
            children_X.append(self._mutate(children.get("X")[:size]))
            sizes.append(size)
            offset += len(island)
        # 交错各岛的后代，预算不足时仍按同一顺序截断整个评价批次。
        active = [i for i, size in enumerate(sizes) if size]
        X = np.stack(children_X, axis=1).reshape(-1, self.problem.n_var)
        offspring = self._evaluate(X)
        updated = list(islands)
        for position, i in enumerate(active):
            merged = Population.merge(islands[i], offspring[position::len(active)])
            context = self.normalization
            if self.config.normalization_mode == "legacy":
                context = ObjectiveNormalization.from_objectives(merged.get("F"))
            order = np.argsort(self._scores(merged, context), kind="stable")
            updated[i] = merged[order[:self.config.island_population]]
        return updated, foreign_parent_count

    def _representatives(self, islands) -> Population:
        best = []
        for island in islands:
            if len(island):
                context = self.normalization
                if self.config.normalization_mode == "legacy":
                    context = ObjectiveNormalization.from_objectives(island.get("F"))
                best.append(island[int(np.argmin(self._scores(island, context)))].copy())
        return Population.create(*best)

    def _qualify(self, representatives) -> np.ndarray:
        radius = self.config.principle_probe_radius
        if not self.config.parent_check or radius == 0:
            return np.ones(len(representatives), dtype=bool)
        # WFG 等非单位区间先归一化。半径越大，可能发现的支配者越多，筛选越严。
        X = (representatives.get("X") - self.problem.xl) / (self.problem.xu - self.problem.xl)
        F = representatives.get("F")
        qualified = np.ones(len(representatives), dtype=bool)
        for i in range(len(representatives)):
            nearby = np.linalg.norm(X - X[i], axis=1) <= radius
            dominates = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            qualified[i] = not np.any(nearby & dominates)
        return qualified

    def _eligible_parents(self, representatives) -> tuple[Population, int, bool]:
        qualified = self._qualify(representatives)
        count = int(np.sum(qualified))
        fallback = count < 2
        if fallback:
            # 唯一按合成适应度补足资格池的分支，对应伪代码第 19 行。
            order = np.argsort(self._scores(representatives), kind="stable")
            return representatives[order[:2]], count, True
        return representatives[qualified], count, False

    def _parent_pool(self, eligible) -> Population:
        chosen = []
        scores = self._scores(eligible)
        for front in NonDominatedSorting().do(eligible.get("F")):
            order = front[np.argsort(scores[front], kind="stable")]
            chosen.extend(order.tolist())
            if len(chosen) >= self.config.parent_pool_limit:
                break
        return eligible[np.asarray(chosen[:self.config.parent_pool_limit], dtype=int)]

    def _inherit(self, a, b, structured: bool) -> np.ndarray:
        blocks = self.blocks if structured else [np.array([i]) for i in range(self.problem.n_var)]
        selected = self.rng.integers(0, 2, len(blocks)).astype(bool)
        # 排除空集和全集，确保确实从两个亲本继承元素。
        if np.all(selected) or not np.any(selected):
            selected[self.rng.integers(len(blocks))] = not selected[0]
        child = np.asarray(b).copy()
        for block, take_a in zip(blocks, selected):
            if take_a:
                child[block] = a[block]
        return child

    @property
    def combination_method(self) -> str:
        return self.config.combination_method if self.config.block_combination else "sbx_only"

    def _combine(self, parents) -> Population:
        if len(parents) < 2 or not self.remaining:
            return Population.empty()
        possible = list(combinations(range(len(parents)), 2))
        count = min(self.config.combination_pairs_limit, len(possible), self.remaining)
        indices = self.rng.choice(len(possible), size=count, replace=False)
        pairs = np.asarray([possible[i] for i in indices], dtype=int)
        if self.combination_method == "sbx_only":
            # SBX 返回两个对称子代，取每对的第一个；每对最多一个混血候选。
            X = self.sbx.do(self.problem, parents, parents=pairs, random_state=self.rng).get("X")[:count]
        else:
            X = np.asarray([
                self._inherit(parents[a].X, parents[b].X, self.combination_method == "structured")
                for a, b in pairs
            ])
        # 完全可组合/部分可组合元素统一经过变异和 repair；当前 repair 是越界裁剪。
        return self._evaluate(self._mutate(X))

    def _select_founders(self, candidates) -> Population:
        if len(candidates) < self.n_origin and self.remaining:
            # 规格要求下一代仍有 P 个起源者，但没有定义候选不足时的补足规则。
            # 先明确停止，不能擅自重复起源者、引入随机解或缩小群体规模。
            raise RuntimeError(
                f"IE-MOEO 去重候选只有 {len(candidates)} 个，少于 P={self.n_origin}；"
                f"FE={self.n_eval}/{self.case.max_fes}；候选不足时的起源者补足规则尚未定义"
            )
        size = min(self.n_origin, len(candidates))
        if self.config.diversity_maintenance and self.config.outer_survival == "nsga3":
            return ReferenceDirectionSurvival(self.ref_dirs).do(
                self.problem, candidates, n_survive=size, random_state=self.rng,
            )
        chosen = []
        for front in NonDominatedSorting().do(candidates.get("F")):
            remaining = size - len(chosen)
            if remaining <= 0:
                break
            take = front if len(front) <= remaining else self.rng.choice(front, remaining, replace=False)
            chosen.extend(take.tolist())
        return candidates[np.asarray(chosen, dtype=int)]

    def _set_front(self, candidates) -> None:
        unique = self._unique(candidates)
        front = NonDominatedSorting().do(unique.get("F"), only_non_dominated_front=True)
        self.candidate_pool = unique[front]

    def _select_base(self) -> None:
        normalized = ObjectiveNormalization.from_objectives(self.candidate_pool.get("F"))
        asf = np.max(normalized.apply(self.candidate_pool.get("F")) / np.maximum(self.weights, 1e-6), axis=1)
        index = int(np.argmin(asf))
        self.x_base = self.candidate_pool[index].X.copy()
        self.f_base = self.candidate_pool[index].F.copy()
        self.base_asf = float(asf[index])

    def run(self) -> tuple[Population, int]:
        self.founders = self._evaluate(self.initial_X)
        self.initial_evaluations = self.n_eval
        self._set_front(self.founders)
        stalled_rounds = 0
        while self.remaining:
            start = self.n_eval
            islands = self._new_islands(self.founders)
            expansion_end = self.n_eval
            local_population = merge_populations(islands)
            self.normalization = ObjectiveNormalization.from_objectives(local_population.get("F"))
            generations = (
                self.config.inner_generations_early
                if start / self.case.max_fes < self.config.switch_ratio
                else self.config.inner_generations_late
            )
            foreign_parents = 0
            for _ in range(generations):
                if not self.remaining:
                    break
                islands, count = self._local_generation(islands)
                foreign_parents += count
            local_end = self.n_eval
            representatives = self._representatives(islands)
            eligible, qualified_count, fallback = self._eligible_parents(representatives)
            parents = self._parent_pool(eligible)
            offspring = self._combine(parents)
            candidates = merge_populations(
                [*islands, eligible, offspring] if self.config.elitist_pool else [eligible, offspring]
            )
            candidates = self._unique(candidates)
            self.founders = self._select_founders(candidates)
            self._set_front(candidates)
            self.global_selection_count += 1
            record = {
                "outer_iteration": self.global_selection_count,
                "fe_start": start, "fe_end": self.n_eval,
                "outer_batch_fes": self.n_eval - start,
                "expansion_fes": expansion_end - start,
                "island_evolution_fes": local_end - expansion_end,
                "island_fes": local_end - start,
                "recombination_offspring": self.n_eval - local_end,
                "recombination_candidates": len(offspring),
                "merge_fes": 0,
                "founder_count": len(islands),
                "representative_count": len(representatives),
                "qualified_parent_count": qualified_count,
                "parent_acceptance_rate": qualified_count / len(islands),
                "eligible_parent_count": len(eligible),
                "parent_pool_size": len(parents),
                "parent_fallback": fallback,
                "foreign_parent_count": foreign_parents,
                "merged_population_size": len(candidates),
                "origin_population_size": len(self.founders),
                "front_size": len(self.candidate_pool),
                "budget_exhausted": self.remaining == 0,
            }
            if self.on_outer_selection is not None:
                values = self.on_outer_selection(self.n_eval, self.candidate_pool)
                if values:
                    record.update(values)
            self.outer_records.append(record)
            # 固定点等退化情形不得无限循环，也不能虚构 FE 或偷偷随机重启。
            stalled_rounds = stalled_rounds + 1 if self.n_eval == start else 0
            if stalled_rounds >= 20:
                raise RuntimeError("IE-MOEO 连续 20 轮未产生新决策，未耗尽预算；请检查修复/变异组件")
        accounted = self.initial_evaluations + sum(
            row["island_fes"] + row["recombination_offspring"] for row in self.outer_records
        )
        if accounted != self.n_eval or self.n_eval != self.case.max_fes:
            raise RuntimeError("IE-MOEO FE 分账与总预算不一致")
        self._select_base()
        return self.candidate_pool, self.global_selection_count
