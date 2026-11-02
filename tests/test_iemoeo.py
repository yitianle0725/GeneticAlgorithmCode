from __future__ import annotations

import csv
import json
import io
import sys
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iemoec_experiment.config import ExperimentCase, IEMOECConfig, IEMOEOConfig
from iemoec_experiment.factory import reference_directions
from iemoec_experiment.iemoeo import IEMOEORunner, variable_blocks
from iemoec_experiment.iemoeo_diagnostics import combination_audit, selection_pair_audit
from iemoec_experiment.initialization import initialization_hash, shared_initial_decisions
from iemoec_experiment.normalization import ObjectiveNormalization
from iemoec_experiment.problems import make_problem
from iemoec_experiment.runner import run_case
from run import build_parser, resolve_cases
from summarize import paired_tests, win_tie_loss
from create_manifest import main as create_manifest_main


class BoxProblem(Problem):
    def __init__(self, scale=1.0):
        super().__init__(n_var=4, n_obj=3, xl=np.zeros(4), xu=scale * np.arange(1, 5))

    def _evaluate(self, X, out, *args, **kwargs):
        normalized = X / self.xu
        out["F"] = np.column_stack([
            np.sum(normalized ** 2, axis=1),
            np.sum((normalized - 0.5) ** 2, axis=1),
            np.sum((normalized - 1) ** 2, axis=1),
        ])


class IEMOEOTests(unittest.TestCase):
    def make_runner(self, budget=160, problem=None, **changes):
        config = replace(IEMOEOConfig(origin_ratio=0.02, island_population=3), **changes)
        case = ExperimentCase("IEMOEO", "dtlz2", 3, 7, budget, iemoeo=config)
        return IEMOEORunner(problem or BoxProblem(), case)

    def test_m4_configuration_and_legacy_serialization(self):
        case = ExperimentCase("iemoeo", "dtlz2", 4, 1, 840)
        case.validate()
        self.assertEqual(len(reference_directions(case)), 84)
        self.assertEqual(case.algorithm_label, "IE-MOEO")
        self.assertEqual(case.algorithm_schema_version, 3)
        self.assertNotIn("iemoec", case.to_dict())
        old = ExperimentCase("IEMOEC", "dtlz2", 3, 1, 182, iemoec=IEMOECConfig.for_variant("s3_elite"))
        self.assertNotIn("iemoeo", old.to_dict())
        self.assertEqual(old.algorithm_schema_version, 7)

    def test_invalid_configuration_is_rejected(self):
        invalid = [
            {"origin_ratio": 0}, {"island_population": 1},
            {"inner_generations_early": 0}, {"parent_pool_limit": 1},
            {"combination_pairs_limit": -1}, {"block_size": 1},
            {"block_mode": "unknown"}, {"combination_method": "unknown"},
            {"principle_probe_radius": -0.1}, {"principle_probe_radius": float("nan")},
            {"outer_survival": "unknown"}, {"use_crowding": True},
            {"representative_selection": "unknown"},
            {"parent_check_mode": "unknown"},
            {"block_mode": "shuffled_known", "block_size": 4},
            {"combination_method": "none", "block_combination": False},
            {"objective_weights": (0.5, 0.5)}, {"objective_weights": (1.0, 1.0, 1.0)},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(IEMOEOConfig(), **changes).validate(3)
        with self.assertRaises(ValueError):
            ExperimentCase("IEMOEO", "c2dtlz2", 3, 1, 182).validate()
        with self.assertRaises(ValueError):
            self.make_runner(budget=1)

    def test_rebuilding_historical_manifest_preserves_schema1(self):
        from summarize import load_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "historical"
            case = ExperimentCase("IEMOEO", "dtlz2", 3, 7, 182, str(root),
                                  reference_points=30, high_dim_hv_samples=1000)
            run_case(case)
            for name in ("config.json", "metrics.json"):
                path = case.output_dir / name
                data = json.loads(path.read_text(encoding="utf-8"))
                data["algorithm_schema_version"] = 1
                if name == "config.json":
                    data["iemoeo"].pop("representative_selection")
                    data["iemoeo"].pop("selection_audit")
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            arguments = ["create_manifest.py", str(root), "--problems", "dtlz2", "--objectives", "3",
                         "--seeds", "7", "--algorithms", "IEMOEO", "--metric-schema-version", "5"]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                self.assertEqual(create_manifest_main(), 0)
            rows, validation = load_rows(root)
            self.assertEqual(rows[0]["algorithm_schema_version"], 1)
            self.assertEqual(validation["completed"], 1)

    def test_representative_controls_preserve_slots_and_expose_sources(self):
        islands = [
            Population.new("X", np.array([[.1] * 4, [.2] * 4]), "F", np.array([[0.] * 3, [1.] * 3])),
            Population.new("X", np.array([[.3] * 4, [.4] * 4]), "F", np.array([[5.] * 3, [6.] * 3])),
        ]
        for mode, expected in (("local_best", [0, 5]), ("global_best", [0, 1])):
            runner = self.make_runner(representative_selection=mode)
            runner.normalization = ObjectiveNormalization.from_objectives(np.array([[0.] * 3, [6.] * 3]))
            representatives = runner._representatives(islands)
            self.assertEqual(len(representatives), 2)
            np.testing.assert_array_equal(representatives.get("F")[:, 0], expected)
            self.assertEqual(runner.n_eval, 0)
        self.assertEqual(len(set(representatives.get("source_island"))), 1)
        runner = self.make_runner(representative_selection="local_random")
        runner.normalization = ObjectiveNormalization.from_objectives(np.array([[0.] * 3, [6.] * 3]))
        runner.rng = np.random.default_rng(123)
        representatives = runner._representatives(islands)
        np.testing.assert_array_equal(representatives.get("F")[:, 0], [0, 6])
        self.assertEqual(len(set(representatives.get("source_island"))), 2)

    def test_crowding_control_retains_tradeoff_extremes(self):
        runner = self.make_runner(outer_survival="rank_crowding")
        candidates = Population.new(
            "X", np.asarray([[value] * 4 for value in (.1, .2, .3, .4)]),
            "F", np.asarray([[0, 1, .5], [.3, .7, .5], [.7, .3, .5], [1, 0, .5]]),
        )
        selected = runner._select_founders(candidates)
        self.assertEqual(set(selected.get("F")[:, 0]), {0., 1.})
        self.assertTrue(np.all(np.isinf(selected.get("crowding"))))
        self.assertEqual(runner.n_eval, 0)

    def test_global_best_does_not_fill_representative_slots_with_clones(self):
        runner = self.make_runner(representative_selection="global_best")
        runner.normalization = ObjectiveNormalization.from_objectives(np.array([[0.] * 3, [2.] * 3]))
        islands = [
            Population.new("X", np.array([[.1] * 4, [.2] * 4]), "F", np.array([[0.] * 3, [1.] * 3])),
            Population.new("X", np.array([[.1] * 4, [.3] * 4]), "F", np.array([[0.] * 3, [2.] * 3])),
        ]
        selected = runner._representatives(islands)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({runner._key(ind.X) for ind in selected}), 2)
        np.testing.assert_array_equal(selected.get("F")[:, 0], [0, 1])

    def test_rejecting_parent_need_not_change_rank_limited_parent_pool(self):
        runner = self.make_runner(principle_probe_radius=10, parent_pool_limit=2)
        representatives = Population.new("X", np.array([[.1] * 4, [.2] * 4, [.3] * 4]),
                                         "F", np.array([[0, 3, 3], [3, 0, 3], [4, 4, 4]]))
        runner.normalization = ObjectiveNormalization.from_objectives(representatives.get("F"))
        eligible, qualified, _ = runner._eligible_parents(representatives)
        self.assertEqual(qualified, 2)
        selected = runner._parent_pool(eligible)
        audit = runner._parent_pool_audit(representatives, selected)
        self.assertFalse(audit["parent_pool_changed_by_check"])
        self.assertEqual(audit["parent_pool_unique_count"], 2)

    def test_no_combination_redirects_budget_and_obeys_tail_budgets(self):
        for budget in (2, 3, 7, 59, 160, 420):
            with self.subTest(budget=budget):
                runner = self.make_runner(budget=budget, combination_method="none")
                with patch.object(runner, "_combine", side_effect=AssertionError("执行了全局组合")):
                    runner.run()
                self.assertEqual(runner.n_eval, budget)
                self.assertTrue(all(row["recombination_offspring"] == 0 for row in runner.outer_records))
                self.assertTrue(all(row["combination_reallocated_fes"] == row["combination_budget_requested"]
                                    for row in runner.outer_records))
                self.assertEqual(runner.initial_evaluations + sum(row["island_fes"] for row in runner.outer_records), budget)
                if budget >= 160:
                    self.assertGreater(sum(row["combination_reallocated_fes"] for row in runner.outer_records), 0)

    def test_local_scalar_check_can_change_nondominated_parent_pool(self):
        runner = self.make_runner(parent_check_mode="local_scalar", principle_probe_radius=.2)
        representatives = Population.new(
            "X", np.array([[.1] * 4, [.2] * 4, [.9] * 4]) * runner.problem.xu,
            "F", np.array([[0., 2., 2.], [2., 0., 0.], [0., 1., 3.]]),
        )
        runner.normalization = ObjectiveNormalization.from_objectives(representatives.get("F"))
        # 前两个互不支配；第二个的统一标量值更小，且在 RMS 半径内。
        qualified = runner._qualify(representatives, representatives)
        np.testing.assert_array_equal(qualified, [False, True, True])
        eligible, count, fallback = runner._eligible_parents(representatives, representatives)
        self.assertEqual(count, 2)
        self.assertFalse(fallback)
        audit = runner._parent_pool_audit(representatives, runner._parent_pool(eligible))
        self.assertTrue(audit["parent_pool_changed_by_check"])
        self.assertEqual(runner.n_eval, 0)

    def test_local_scalar_uses_nonrepresentative_witness_and_normalized_rms(self):
        results = []
        for scale in (1, 100):
            runner = self.make_runner(problem=BoxProblem(scale), parent_check_mode="local_scalar",
                                      principle_probe_radius=.11)
            pool = Population.new("X", np.array([[.1] * 4, [.2] * 4, [.8] * 4]) * runner.problem.xu,
                                  "F", np.array([[3.] * 3, [1.] * 3, [0.] * 3]))
            runner.normalization = ObjectiveNormalization.from_objectives(pool.get("F"))
            results.append(runner._qualify(pool[[0, 2]], pool))
            np.testing.assert_array_equal(results[-1], [False, True])
            runner.config = replace(runner.config, parent_check=False)
            self.assertTrue(np.all(runner._qualify(pool[[0, 2]])))
            self.assertEqual(runner.n_eval, 0)
        np.testing.assert_array_equal(*results)

    def test_local_scalar_modes_keep_tail_budget_and_audit_rng(self):
        for budget in (2, 3, 7, 59, 160):
            for method in ("structured", "none"):
                options = dict(budget=budget, parent_check_mode="local_scalar",
                               principle_probe_radius=.2, combination_method=method)
                plain, audited = self.make_runner(**options), self.make_runner(**options, selection_audit=True)
                first, _ = plain.run()
                second, _ = audited.run()
                np.testing.assert_array_equal(first.get("X"), second.get("X"))
                self.assertEqual(audited.n_eval, budget)
                self.assertEqual(audited.initial_evaluations + sum(
                    r["island_fes"] + r["recombination_offspring"] for r in audited.outer_records), budget)

    def test_shuffled_known_preserves_sizes_and_has_independent_seed(self):
        problem = make_problem("dtlz2", 4)
        known = variable_blocks(problem, "dtlz2", "known_separable", 2)
        shuffled = variable_blocks(problem, "dtlz2", "shuffled_known", 2, seed=7)
        self.assertEqual(list(map(len, known)), list(map(len, shuffled)))
        np.testing.assert_array_equal(np.sort(np.concatenate(shuffled)), np.arange(problem.n_var))
        self.assertFalse(np.array_equal(known[0], shuffled[0]))
        for first, second in zip(shuffled, variable_blocks(problem, "dtlz2", "shuffled_known", 2, seed=7)):
            np.testing.assert_array_equal(first, second)

    def test_combination_audit_detects_mutation_damage_without_evaluations(self):
        parents = Population.new("X", np.array([[0.] * 4, [1.] * 4]))
        before = np.array([[0., 0., 1., 1.]])
        after = np.array([[.2, 0., 1., 1.]])
        audit = combination_audit(parents, np.array([[0, 1]]), before, after,
                                  [np.array([0, 1]), np.array([2, 3])])
        self.assertEqual(audit["combination_block_observations"], 2)
        self.assertEqual(audit["combination_intact_blocks_before_mutation"], 2)
        self.assertEqual(audit["combination_intact_blocks_after_mutation"], 1)
        self.assertEqual(audit["combination_parent_copies_before_mutation"], 0)

    def test_reallocated_generation_cannot_overspend_or_fake_evaluations(self):
        runner = self.make_runner()
        evaluated = runner._evaluate([[.1] * 4, [.2] * 4, [.3] * 4], evaluation_limit=2)
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(runner.n_eval, 2)
        with patch.object(runner, "_local_generation", return_value=([evaluated], 0)):
            with self.assertRaisesRegex(RuntimeError, "20 次"):
                runner._redirect_combination_budget([evaluated], 1)
        self.assertEqual(runner.n_eval, 2)

    def test_selection_audit_preserves_search_random_stream(self):
        plain, audited = self.make_runner(), self.make_runner(selection_audit=True)
        first, _ = plain.run()
        second, _ = audited.run()
        np.testing.assert_array_equal(first.get("X"), second.get("X"))
        np.testing.assert_array_equal(first.get("F"), second.get("F"))
        self.assertEqual(plain.n_eval, audited.n_eval)
        self.assertTrue(all("audited_pair_count" in row for row in audited.outer_records))

    def test_selection_audit_counts_near_objectives_far_decisions(self):
        runner = self.make_runner()
        candidates = Population.new("X", np.array([runner.problem.xl, runner.problem.xu]),
                                    "F", np.zeros((2, 3)))
        result = selection_pair_audit(runner.problem, candidates, candidates[:1], 7)
        self.assertEqual(result["near_f_far_x_pair_count"], 1)
        self.assertEqual(result["near_f_far_x_both_retained"], 0)
        self.assertEqual(result["near_f_far_x_endpoints_retained"], 1)
        self.assertEqual(runner.n_eval, 0)

    def test_tail_budgets_accounting_callbacks_and_repeatability(self):
        for budget in (2, 3, 7, 12, 17, 59, 160):
            with self.subTest(budget=budget):
                a, b = self.make_runner(budget), self.make_runner(budget)
                batches, checkpoints, selections = [], [], []
                a.on_evaluation = lambda fe, pop: batches.append((fe, len(pop)))
                a.on_checkpoint = lambda fe, pop: checkpoints.append(fe)
                a.on_outer_selection = lambda fe, pop: selections.append(fe)
                pa, rounds = a.run()
                pb, _ = b.run()
                self.assertEqual(a.n_eval, budget)
                self.assertEqual(sum(size for _, size in batches), budget)
                self.assertEqual(checkpoints, [fe for fe, _ in batches])
                self.assertEqual(len(selections), rounds)
                self.assertEqual(a.initial_evaluations + sum(r["island_fes"] + r["recombination_offspring"] for r in a.outer_records), budget)
                np.testing.assert_array_equal(pa.get("X"), pb.get("X"))
                self.assertEqual(a.outer_records, b.outer_records)
                self.assertTrue(np.all(np.isfinite(pa.get("F"))))
                self.assertEqual(len(NonDominatedSorting().do(pa.get("F"), only_non_dominated_front=True)), len(pa))
                self.assertTrue(any(np.array_equal(a.x_base, ind.X) for ind in pa))

    def test_evaluation_cache_and_replaceable_repair(self):
        runner = self.make_runner()
        calls = []
        def repair(problem, X):
            calls.append(X.copy())
            return np.clip(X, problem.xl, problem.xu)
        runner.repair = repair
        first = runner._evaluate([[-1, 5, 9, -3], [-1, 5, 9, -3]])
        self.assertEqual(runner.n_eval, 1)
        np.testing.assert_array_equal(first.get("X"), [[0, 2, 3, 0], [0, 2, 3, 0]])
        second = runner._evaluate(first.get("X"))
        self.assertEqual(runner.n_eval, 1)
        np.testing.assert_array_equal(first.get("F"), second.get("F"))
        first[0].F[:] = -100
        self.assertTrue(np.all(second[0].F >= 0))
        self.assertEqual(len(calls), 2)

    def test_islands_rebuilt_from_current_founders(self):
        runner = self.make_runner()
        runner._mutate = lambda X: X + 0.001
        first = runner._evaluate(runner.initial_X)
        islands = runner._new_islands(first)
        for i, island in enumerate(islands):
            np.testing.assert_allclose(island.get("X"), np.tile(first[i].X + 0.001, (3, 1)))
        next_founders = runner._evaluate([[0.1, 0.2, 0.3, 0.4], [0.4, 0.6, 0.9, 1.2]])
        rebuilt = runner._new_islands(next_founders)
        for i, island in enumerate(rebuilt):
            np.testing.assert_allclose(island.get("X"), np.tile(next_founders[i].X + 0.001, (3, 1)))
        self.assertEqual(runner.n_eval, 8)  # 重复候选不重复计费。

    def test_radius_zero_and_disabled_check_admit_all(self):
        reps = Population.new("X", np.zeros((3, 4)), "F", np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        for changes in ({"principle_probe_radius": 0}, {"parent_check": False}):
            runner = self.make_runner(**changes)
            self.assertTrue(np.all(runner._qualify(reps)))
            self.assertEqual(runner.n_eval, 0)

    def test_neighborhood_dominance_normalized_distance_and_radius_direction(self):
        fractions = np.array([[0.1] * 4, [0.2] * 4, [0.9] * 4])
        F = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        results = []
        for scale in (1, 100):
            runner = self.make_runner(problem=BoxProblem(scale), principle_probe_radius=0.21)
            reps = Population.new("X", fractions * runner.problem.xu, "F", F)
            qualified = runner._qualify(reps)
            np.testing.assert_array_equal(qualified, [True, False, True])
            runner.config = replace(runner.config, principle_probe_radius=2.0)
            np.testing.assert_array_equal(runner._qualify(reps), [True, False, False])
            results.append(qualified)
            self.assertEqual(runner.n_eval, 0)
        np.testing.assert_array_equal(*results)

    def test_parent_fallback_and_rank_then_scalar_limit(self):
        runner = self.make_runner(principle_probe_radius=10, parent_pool_limit=2)
        reps = Population.new("X", np.arange(12).reshape(3, 4) / 20, "F", np.array([[2, 2, 2], [1, 1, 1], [3, 3, 3]]))
        runner.normalization = ObjectiveNormalization.from_objectives(reps.get("F"))
        eligible, qualified, fallback = runner._eligible_parents(reps)
        self.assertEqual(qualified, 1)
        self.assertTrue(fallback)
        np.testing.assert_array_equal(eligible.get("F"), [[1, 1, 1], [2, 2, 2]])
        F = np.array([[0, 4, 4], [2, 2, 2], [3, 3, 3], [4, 0, 0]])
        pool = Population.new("X", np.zeros((4, 4)), "F", F)
        runner.normalization = ObjectiveNormalization.from_objectives(F)
        selected = runner._parent_pool(pool)
        np.testing.assert_array_equal(selected.get("F"), [[4, 0, 0], [2, 2, 2]])

    def test_blocks_cover_coordinates_and_known_boundaries(self):
        for name in ("dtlz1", "dtlz2", "dtlz7", "wfg1", "wfg9"):
            for m in (3, 4):
                problem = make_problem(name, m)
                blocks = variable_blocks(problem, name, "known_separable", 4)
                np.testing.assert_array_equal(np.concatenate(blocks), np.arange(problem.n_var))
                self.assertEqual(len(blocks[0]), m - 1 if name.startswith("dtlz") else 2 * (m - 1))
        runner = self.make_runner(block_size=8)
        self.assertEqual(len(runner.blocks), 4)
        np.testing.assert_array_equal(np.concatenate(runner.blocks), np.arange(4))

    def test_structured_inheritance_preserves_whole_blocks(self):
        runner = self.make_runner(block_size=2)
        a, b = np.zeros(4), np.ones(4)
        for _ in range(30):
            x = runner._inherit(a, b, structured=True)
            self.assertEqual(x.shape, (4,))
            self.assertTrue(all(np.all(x[block] == x[block[0]]) for block in runner.blocks))
            self.assertTrue(np.any(x == 0) and np.any(x == 1))
        broken = [runner._inherit(a, b, structured=False) for _ in range(30)]
        self.assertTrue(any(x[0] != x[1] or x[2] != x[3] for x in broken))

    def test_combination_methods_and_pair_limits(self):
        for method in ("structured", "random_coord", "sbx_only"):
            runner = self.make_runner(combination_method=method, combination_pairs_limit=2)
            parents = runner._evaluate([[0.1] * 4, [0.5] * 4, [0.9] * 4])
            children = runner._combine(parents)
            self.assertEqual(len(children), 2)
            self.assertTrue(np.all(children.get("X") >= runner.problem.xl))
            self.assertTrue(np.all(children.get("X") <= runner.problem.xu))
        runner = self.make_runner(block_combination=False)
        parents = runner._evaluate(runner.initial_X)
        with patch.object(runner, "_inherit", side_effect=AssertionError("unexpected inheritance")):
            self.assertEqual(len(runner._combine(parents)), 1)
        self.assertEqual(runner.combination_method, "sbx_only")

    def test_isolation_and_shared_breeding_are_different(self):
        isolated = self.make_runner(budget=300)
        shared = self.make_runner(budget=300, isolation=False)
        isolated.run()
        shared.run()
        self.assertEqual(sum(r["foreign_parent_count"] for r in isolated.outer_records), 0)
        self.assertGreater(sum(r["foreign_parent_count"] for r in shared.outer_records), 0)

    def test_rank_survival_ignores_reference_directions_and_uses_fronts(self):
        runner = self.make_runner(diversity_maintenance=False)
        candidates = Population.new("X", np.zeros((4, 4)), "F", np.array([[3, 3, 3], [1, 1, 1], [4, 4, 4], [2, 2, 2]]))
        with patch("iemoec_experiment.iemoeo.ReferenceDirectionSurvival", side_effect=AssertionError("unexpected NSGA3")):
            selected = runner._select_founders(candidates)
        np.testing.assert_array_equal(selected.get("F"), [[1, 1, 1], [2, 2, 2]])

    def test_elitist_pool_off_excludes_nonrepresentatives(self):
        runner = self.make_runner(elitist_pool=False)
        runner.run()
        for row in runner.outer_records:
            self.assertLessEqual(row["merged_population_size"], row["eligible_parent_count"] + row["recombination_candidates"])

    def test_normalized_scalar_and_asf_with_zero_weight(self):
        runner = self.make_runner(objective_weights=(0.0, 0.5, 0.5))
        population = Population.new("X", np.array([[0.2] * 4, [0.5] * 4, [0.8] * 4]), "F", np.array([[1, 4, 1000], [2, 2, 500], [4, 1, 250]]))
        runner.normalization = ObjectiveNormalization.from_objectives(population.get("F"))
        scores = runner._scores(population)
        scaled = population.copy()
        scaled.set("F", population.get("F") * [100, 1, 0.001])
        context = ObjectiveNormalization.from_objectives(scaled.get("F"))
        np.testing.assert_allclose(scores, runner._scores(scaled, context))
        runner.candidate_pool = population
        runner._select_base()
        np.testing.assert_array_equal(runner.x_base, population[0].X)
        self.assertTrue(np.isfinite(runner.base_asf))

    def test_insufficient_candidates_fail_without_fake_evaluations(self):
        runner = self.make_runner()
        runner.repair = lambda problem, X: np.zeros_like(X)
        with self.assertRaisesRegex(RuntimeError, "P=2"):
            runner.run()
        self.assertEqual(runner.n_eval, 1)

    def test_integration_metrics_history_resume_and_all_ablations(self):
        with tempfile.TemporaryDirectory() as temp:
            for i, changes in enumerate(({}, {"isolation": False}, {"parent_check": False}, {"block_combination": False}, {"diversity_maintenance": False}, {"elitist_pool": False}, {"normalization_mode": "legacy"})):
                case = ExperimentCase("IEMOEO", "dtlz2", 4, 1, 420, str(Path(temp) / str(i)), reference_points=30, high_dim_hv_samples=1000, iemoeo=replace(IEMOEOConfig(island_population=3), **changes))
                result = run_case(case)
                self.assertEqual(result["n_eval"], 420)
                self.assertEqual(sum(result["evaluation_breakdown"].values()), 420)
                self.assertEqual(result["origin_population_size"], 17)
                self.assertTrue(np.isfinite(result["igd_plus"]))
                self.assertEqual(run_case(case)["status"], "skipped")
                with (case.output_dir / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
                    history = list(csv.DictReader(handle))
                self.assertEqual(float(history[-1]["igd_plus"]), result["igd_plus"])
                self.assertTrue((case.output_dir / "iemoeo_diagnostics.csv").exists())
                base = json.loads((case.output_dir / "baseline_solution.json").read_text(encoding="utf-8"))
                self.assertEqual(len(base["x_base"]), 13)
                replay = run_case(replace(case, output_root=str(Path(temp) / f"repeat_{i}")))
                self.assertEqual(result["igd_plus"], replay["igd_plus"])
                self.assertEqual(result["hv"], replay["hv"])
                with self.assertRaises(RuntimeError):
                    run_case(replace(case, iemoeo=replace(case.iemoeo, block_size=2)), force=True)

    def test_wfg_bounds_and_shared_initialization(self):
        runner = self.make_runner(problem=make_problem("wfg1", 3), block_mode="uniform")
        expected = shared_initial_decisions(runner.problem, runner.n_origin, runner.case.seed)
        np.testing.assert_array_equal(runner.initial_X, expected)
        pop, _ = runner.run()
        self.assertTrue(np.all(pop.get("X") >= runner.problem.xl))
        self.assertTrue(np.all(pop.get("X") <= runner.problem.xu))

    def test_cli_mixed_algorithms_and_weights_round_trip(self):
        args = build_parser().parse_args([
            "--algorithms", "iemoeo", "iemoec", "nsga3", "--objectives", "4",
            "--problems", "dtlz2", "--seeds", "1", "--evals-per-pop", "10",
            "--objective-weights", "0.1", "0.2", "0.3", "0.4",
            "--no-isolation", "--no-parent-check", "--no-block-combination",
            "--no-diversity-maintenance", "--no-elitist-pool", "--principle-probe-radius", "0",
        ])
        cases = resolve_cases(args)
        self.assertEqual([c.normalized_algorithm for c in cases], ["IEMOEO", "IEMOEC", "NSGA3"])
        self.assertTrue(all(c.max_fes == 840 for c in cases))
        config = cases[0].iemoeo
        self.assertFalse(any([config.isolation, config.parent_check, config.block_combination, config.diversity_maintenance, config.elitist_pool]))
        self.assertEqual(json.loads(json.dumps(cases[0].to_dict(), ensure_ascii=False)), cases[0].to_dict())
        with tempfile.TemporaryDirectory() as temp:
            case = replace(cases[0], output_root=temp, result_store=str(Path(temp) / "algorithms"),
                           reference_points=30, high_dim_hv_samples=1000)
            result = run_case(case)
            expected = shared_initial_decisions(make_problem("dtlz2", 4), 17, 1)
            self.assertEqual(result["initialization_hash"], initialization_hash(expected))
            self.assertEqual(run_case(case)["status"], "skipped")

    def test_rank_sum_statistics_and_win_tie_loss(self):
        from scipy.stats import ranksums
        rows = []
        for algorithm, offset in (("IEMOEO", 0), ("NSGA3", 10)):
            for seed in range(10):
                rows.append(dict(problem="dtlz2", n_obj=4, algorithm=algorithm,
                                 seed=seed, igd_plus=seed + offset, hv=30 - seed - offset))
        tests = paired_tests(rows, "IEMOEO", 0.05, "rank_sum")
        self.assertEqual(len(tests), 2)
        self.assertAlmostEqual(tests[0]["p_value"], ranksums(np.arange(10), np.arange(10) + 10).pvalue)
        self.assertTrue(all(test["target_result"] == "+" for test in tests))
        self.assertTrue(all(row["wins"] == 1 and row["losses"] == 0 for row in win_tie_loss(tests)))

    def test_known_structure_wfg_and_dtlz_m4_integration(self):
        for name in ("dtlz1", "dtlz7", "wfg1", "wfg9"):
            with self.subTest(problem=name), tempfile.TemporaryDirectory() as temp:
                case = ExperimentCase("IEMOEO", name, 4, 1, 840, temp,
                                      reference_points=30, high_dim_hv_samples=1000,
                                      iemoeo=IEMOEOConfig(block_mode="known_separable"))
                result = run_case(case)
                self.assertEqual(result["n_eval"], 840)
                self.assertTrue(np.isfinite(result["igd_plus"]))
                self.assertTrue(np.isfinite(result["hv"]))


if __name__ == "__main__":
    unittest.main()
