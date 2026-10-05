from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
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
from iemoec_experiment.initialization import initialization_hash, shared_initial_decisions
from iemoec_experiment.normalization import ObjectiveNormalization
from iemoec_experiment.problems import make_problem
from iemoec_experiment.runner import run_case
from run import build_parser, resolve_cases
from summarize import paired_tests, win_tie_loss


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
        self.assertEqual(case.algorithm_schema_version, 1)
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
            {"outer_survival": "rank_crowding"}, {"use_crowding": True},
            {"objective_weights": (0.5, 0.5)}, {"objective_weights": (1.0, 1.0, 1.0)},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(IEMOEOConfig(), **changes).validate(3)
        with self.assertRaises(ValueError):
            ExperimentCase("IEMOEO", "c2dtlz2", 3, 1, 182).validate()
        with self.assertRaises(ValueError):
            self.make_runner(budget=1)

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
            case = replace(cases[0], output_root=temp, reference_points=30, high_dim_hv_samples=1000)
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
