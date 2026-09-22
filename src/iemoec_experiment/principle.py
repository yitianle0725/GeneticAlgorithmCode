"""Single-founder, closed-lineage principle verification (continuous boxes only).

Local optimality is a finite numerical test, not a stationarity theorem.
Coordinate recombination tests an element-combination hypothesis; it does not
assert that arbitrary objective functions are separable.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class Lineage:
    identity: int
    population: Population
    weight: np.ndarray
    generations: int = 0
    stagnant: int = 0
    qualified: bool = False
    qualification_attempts: int = 0


class PrincipleRunner:
    def __init__(self, problem, case, initial_X, on_checkpoint=None,
                 on_outer_selection=None, on_evaluation=None):
        case.iemoec.validate()
        if problem.n_constr or not np.all(np.isfinite(problem.xl)) or not np.all(np.isfinite(problem.xu)):
            raise ValueError("principle currently supports unconstrained finite continuous boxes only")
        if problem.n_var < 2:
            raise ValueError("element recombination requires at least two decision variables")
        self.problem = problem
        self.case = case
        self.config = case.iemoec
        # N is only the baseline budget scale, NOT the initial population size.
        from .factory import reference_directions
        self.pop_size = len(reference_directions(case))
        self.n_origin = max(2, math.ceil(self.pop_size * self.config.origin_ratio))
        if len(initial_X) != self.n_origin or case.max_fes < self.n_origin:
            raise ValueError("principle requires exactly P initial ancestors and FE >= P")
        self.initial_X = np.asarray(initial_X).copy()
        self.rng = np.random.default_rng(case.seed)
        self.evaluator = Evaluator()
        self.sbx = SBX(prob=1.0, eta=30)
        self.mutation = PM(prob=1.0, prob_var=1 / problem.n_var, eta=20)
        self.on_checkpoint = on_checkpoint
        self.on_outer_selection = on_outer_selection
        self.on_evaluation = on_evaluation
        self.candidate_pool = Population.empty()
        self.outer_records = []
        self.lineage_records = []
        self.global_selection_count = 0
        self.initial_evaluations = 0
        self.termination_status = "not_started"
        self._next_identity = 0

    @property
    def n_eval(self):
        return int(self.evaluator.n_eval)

    @property
    def remaining(self):
        return self.case.max_fes - self.n_eval

    def _evaluate(self, X, lineage_id=None):
        X = np.asarray(X, dtype=float)[:self.remaining]
        pop = Population.new("X", X.copy())
        if lineage_id is not None:
            pop.set("lineage_id", np.full(len(pop), lineage_id, dtype=int))
        if len(pop):
            self.evaluator.eval(self.problem, pop)
            if self.on_evaluation:
                self.on_evaluation(self.n_eval, pop)
            if self.on_checkpoint:
                visible = self.candidate_pool if len(self.candidate_pool) else pop
                self.on_checkpoint(self.n_eval, visible)
        return pop

    def _mutate(self, X):
        pop = Population.new("X", np.asarray(X).copy())
        return self.mutation.do(self.problem, pop, inplace=False,
                                random_state=self.rng).get("X")

    @staticmethod
    def _unique(pop):
        seen = set()
        indices = []
        for i, x in enumerate(pop.get("X")):
            key = np.asarray(x, dtype=np.float64).tobytes()
            if key not in seen:
                indices.append(i)
                seen.add(key)
        return pop[np.asarray(indices, dtype=int)]

    def _scores(self, pop, weight):
        # Smooth fixed weighted sum: the local environment is frozen per round.
        return ((pop.get("F") - self.ideal) / self.scale) @ weight

    def _best(self, lineage):
        return lineage.population[int(np.argmin(self._scores(lineage.population, lineage.weight)))]

    def _keep_local(self, lineage, children):
        merged = Population.merge(lineage.population, children)
        order = np.argsort(self._scores(merged, lineage.weight), kind="stable")
        lineage.population = merged[order[:self.config.island_population]]

    def _new_lineages(self, founders, round_id):
        self.ideal = np.min(founders.get("F"), axis=0)
        self.scale = np.ptp(founders.get("F"), axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        lineages = []
        for founder in founders:
            identity = self._next_identity
            self._next_identity += 1
            # A single ancestor, no neighbor founders or imported candidates.
            lineage = Lineage(identity, Population.create(founder.copy()),
                              self.rng.dirichlet(np.ones(self.problem.n_obj)))
            lineage.population.set("lineage_id", np.array([identity]))
            lineages.append(lineage)
        # Round-robin expansion keeps tail budgets from favoring one founder.
        for _ in range(self.config.island_population - 1):
            for lineage in lineages:
                if not self.remaining:
                    return lineages
                child = self._evaluate(self._mutate(lineage.population[:1].get("X")), lineage.identity)
                lineage.population = Population.merge(lineage.population, child)
                self.lineage_records.append(dict(round=round_id, lineage=lineage.identity,
                                                 event="founder_mutation", evaluations=len(child),
                                                 foreign_parent_count=0, fe=self.n_eval))
        return lineages

    def _local_generation(self, lineage, round_id, reserve=0):
        if not np.all(lineage.population.get("lineage_id") == lineage.identity):
            raise RuntimeError("foreign individual entered a closed lineage")
        old = float(np.min(self._scores(lineage.population, lineage.weight)))
        size = min(self.config.island_population, max(0, self.remaining - reserve))
        if size == 0:
            return
        parents = []
        scores = self._scores(lineage.population, lineage.weight)
        for _ in range(2 * math.ceil(size / 2)):
            contestants = self.rng.integers(len(scores), size=2)
            parents.append(int(contestants[np.argmin(scores[contestants])]))
        children = self.sbx.do(self.problem, lineage.population,
                               parents=np.asarray(parents).reshape(-1, 2),
                               random_state=self.rng)
        children = self._evaluate(self._mutate(children.get("X")[:size]), lineage.identity)
        self._keep_local(lineage, children)
        new = float(np.min(self._scores(lineage.population, lineage.weight)))
        lineage.generations += 1
        lineage.stagnant = lineage.stagnant + 1 if old - new <= self.config.principle_tolerance else 0
        self.lineage_records.append(dict(round=round_id, lineage=lineage.identity,
                                         event="local_generation", evaluations=len(children),
                                         foreign_parent_count=0, fe=self.n_eval,
                                         best_score=new, generations=lineage.generations,
                                         stagnant=lineage.stagnant))

    def _qualify(self, lineage, round_id, reserve=0):
        best = self._best(lineage)
        x = best.get("X")
        radius_ratio = self.config.principle_probe_radius
        if self.config.variant == "s4":
            radius_ratio *= max(0.125, 0.5 ** lineage.qualification_attempts)
        radius = radius_ratio * (self.problem.xu - self.problem.xl)
        probes = []
        for j in range(self.problem.n_var):
            for sign in (-1, 1):
                trial = x.copy()
                trial[j] = np.clip(x[j] + sign * radius[j], self.problem.xl[j], self.problem.xu[j])
                if not np.array_equal(trial, x):
                    probes.append(trial)
        # Include coupled perturbations, since coordinate checks alone miss saddles.
        for _ in range(2 * self.problem.n_var):
            trial = np.clip(x + self.rng.uniform(-1, 1, self.problem.n_var) * radius,
                            self.problem.xl, self.problem.xu)
            if not np.array_equal(trial, x):
                probes.append(trial)
        old = float(self._scores(Population.create(best), lineage.weight)[0])
        available = max(0, self.remaining - reserve)
        tested = self._evaluate(probes[:available], lineage.identity)
        improved = bool(len(tested) and np.min(self._scores(tested, lineage.weight)) < old - self.config.principle_tolerance)
        complete = len(tested) == len(probes) and bool(probes)
        self._keep_local(lineage, tested)
        lineage.qualified = complete and not improved
        lineage.qualification_attempts += 1
        if improved:
            lineage.stagnant = 0
        self.lineage_records.append(dict(round=round_id, lineage=lineage.identity,
                                         event="qualification", evaluations=len(tested),
                                         foreign_parent_count=0, fe=self.n_eval,
                                         probe_complete=complete, probe_improved=improved,
                                         qualified=lineage.qualified,
                                         probe_radius_ratio=radius_ratio,
                                         qualification_attempt=lineage.qualification_attempts))

    def _combine(self, representatives, lineage_ids, round_id=0):
        X = representatives.get("X")
        children = []
        records = []
        count = min(self.pop_size, self.remaining)
        for child_id in range(count):
            a, b = self.rng.choice(len(X), size=2, replace=False)
            # Explicit coordinate-element decomposition, NOT SBX interpolation.
            mask = self.rng.integers(0, 2, self.problem.n_var).astype(bool)
            if self.problem.n_var > 1:
                chosen = self.rng.choice(self.problem.n_var, size=2, replace=False)
                mask[chosen[0]], mask[chosen[1]] = True, False
            children.append(np.where(mask, X[a], X[b]))
            records.append(dict(event="extreme_combination", round=round_id,
                                             lineage_a=lineage_ids[a], lineage_b=lineage_ids[b],
                                             parent_a_qualified=True, parent_b_qualified=True,
                                             child=child_id, elements_from_a=int(mask.sum()),
                                             parent_a_x=json.dumps(X[a].tolist(), ensure_ascii=False),
                                             parent_b_x=json.dumps(X[b].tolist(), ensure_ascii=False),
                                             inherited_x=json.dumps(children[-1].tolist(), ensure_ascii=False)))
        evaluated = self._evaluate(self._mutate(children))
        for record, x in zip(records, evaluated.get("X")):
            record["mutated_x"] = json.dumps(x.tolist(), ensure_ascii=False)
            record["fe"] = self.n_eval
            self.lineage_records.append(record)
        return evaluated

    def _select_founders(self, mixed):
        mixed = self._unique(mixed)
        chosen = []
        # Only mixed children are candidates. No old parent/elite/archive survives.
        for front in NonDominatedSorting().do(mixed.get("F")):
            left = self.n_origin - len(chosen)
            if left <= 0:
                break
            take = front if len(front) <= left else self.rng.choice(front, left, replace=False)
            chosen.extend(take.tolist())
        return mixed[np.asarray(chosen, dtype=int)]

    def run(self):
        founders = self._evaluate(self.initial_X)
        self.initial_evaluations = len(founders)
        self.candidate_pool = founders
        self.termination_status = "budget_exhausted_before_local_search"
        round_id = 0
        while self.remaining:
            round_id += 1
            start = self.n_eval
            lineages = self._new_lineages(founders, round_id)
            expansion_end = self.n_eval
            if self.config.variant == "principle":
                self._run_legacy_qualification(lineages, round_id)
            else:
                self._run_s4_qualification(lineages, round_id)
            local_end = self.n_eval
            qualified = [l for l in lineages if l.qualified]
            distinct = []
            seen = set()
            for lineage in qualified:
                key = np.asarray(self._best(lineage).get("X"), dtype=np.float64).tobytes()
                if key not in seen:
                    seen.add(key)
                    distinct.append(lineage)
            mixed = Population.empty()
            if self.remaining >= self.pop_size and len(distinct) >= 2:
                representatives = Population.create(*[self._best(l).copy() for l in distinct])
                mixed = self._combine(representatives, [l.identity for l in distinct], round_id)
            if len(mixed):
                founders = self._select_founders(mixed)
                self.candidate_pool = founders
                self.global_selection_count += 1
                self.termination_status = "completed_global_round"
                if self.on_outer_selection:
                    self.on_outer_selection(self.n_eval, founders)
            else:
                # No manufactured 'extrema': continue the same closed lineages
                # to exhaust FE when distinct qualified parents cannot be formed.
                while self.remaining:
                    for lineage in lineages:
                        if not self.remaining:
                            break
                        lineage.qualified = False
                        self._local_generation(lineage, round_id)
                local_end = self.n_eval
                self.termination_status = (
                    "completed_then_budget_exhausted_local"
                    if self.global_selection_count
                    else "budget_exhausted_without_global_combination"
                )
            self.outer_records.append(dict(outer_iteration=round_id, fe_start=start,
                                           fe_end=self.n_eval, founder_count=len(lineages),
                                           qualified_parent_count=len(qualified),
                                           distinct_qualified_parent_count=len(distinct),
                                           expansion_fes=expansion_end-start,
                                           island_evolution_fes=local_end-expansion_end,
                                           island_fes=local_end-start,
                                           recombination_offspring=len(mixed),
                                           isolation_violations=0,
                                           status=self.termination_status))
        return self.candidate_pool, round_id

    def _run_legacy_qualification(self, lineages, round_id):
        """Preserve schema-9 behavior so completed results stay reproducible."""
        while self.remaining and not all(lineage.qualified for lineage in lineages):
            for lineage in lineages:
                if not self.remaining:
                    break
                if lineage.qualified:
                    continue
                self._local_generation(lineage, round_id)
                if (self.remaining
                        and lineage.generations >= self.config.principle_min_generations
                        and lineage.stagnant >= self.config.principle_stagnation_generations):
                    self._qualify(lineage, round_id)

    def _run_s4_qualification(self, lineages, round_id):
        """Stop when two distinct qualified extrema can form mixed offspring."""
        reserve = self.pop_size
        while self.remaining > reserve:
            for lineage in lineages:
                if lineage.qualified:
                    continue
                for _ in range(self.config.principle_min_generations):
                    if self.remaining <= reserve:
                        return
                    self._local_generation(lineage, round_id, reserve)
                if self.remaining > reserve:
                    self._qualify(lineage, round_id, reserve)
            qualified = [lineage for lineage in lineages if lineage.qualified]
            distinct = {
                np.asarray(self._best(lineage).get("X"), dtype=np.float64).tobytes()
                for lineage in qualified
            }
            if len(distinct) >= 2:
                return
