from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.core.population import Population

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iemoec_experiment.config import ExperimentCase, IEMOECConfig
from iemoec_experiment.directions import direction_objective, reference_direction_subset
from iemoec_experiment.factory import make_baseline, reference_directions
from iemoec_experiment.iemoec import IEMOECRunner
from iemoec_experiment.initialization import shared_initial_decisions
from iemoec_experiment.metrics import MetricSuite
from iemoec_experiment.normalization import ObjectiveNormalization
from iemoec_experiment.problems import make_problem, standard_problem_dimensions
from iemoec_experiment.runner import run_case
from summarize import vargha_delaney_a12


class ProblemTests(unittest.TestCase):
    def test_dtlz2_true_front_equation(self):
        problem = make_problem("dtlz2", 3)
        X = np.full((8, problem.n_var), 0.5)
        X[:, :2] = np.linspace(0.05, 0.95, 16).reshape(8, 2)
        F = problem.evaluate(X)
        np.testing.assert_allclose(np.sum(F ** 2, axis=1), 1.0, atol=1e-10)

    def test_wfg_comes_from_pymoo_and_has_expected_shape(self):
        problem = make_problem("wfg1", 5)
        X = np.tile((problem.xl + problem.xu) / 2, (3, 1))
        self.assertEqual(problem.evaluate(X).shape, (3, 5))

    def test_dtlz_uses_standard_problem_specific_k(self):
        self.assertEqual(make_problem("dtlz1", 5).n_var, 9)
        self.assertEqual(make_problem("dtlz2", 5).n_var, 14)
        self.assertEqual(make_problem("dtlz7", 5).n_var, 24)

    def test_all_dtlz_and_wfg_use_standard_dimensions(self):
        for n_obj in (3, 5, 10):
            for name in [f"dtlz{i}" for i in range(1, 8)]:
                n_var, k, _ = standard_problem_dimensions(name, n_obj)
                problem = make_problem(name, n_obj)
                self.assertEqual(problem.n_var, n_var)
                self.assertEqual(problem.n_var - n_obj + 1, k)
            for name in [f"wfg{i}" for i in range(1, 10)]:
                n_var, k, l = standard_problem_dimensions(name, n_obj)
                problem = make_problem(name, n_obj)
                self.assertEqual(problem.n_var, n_var)
                self.assertEqual(problem.k, k)
                self.assertEqual(problem.l, l)

    def test_legacy_convex_dtlz2_wrapper_is_finite(self):
        problem = make_problem("c-dtlz2", 3)
        F = problem.evaluate(np.full((4, problem.n_var), 0.5))
        self.assertEqual(F.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(F)))

    def test_reference_population_sizes(self):
        expected = {3: 91, 5: 210, 8: 120, 10: 220, 15: 120}
        for n_obj, size in expected.items():
            case = ExperimentCase("NSGA2", "dtlz2", n_obj, 1, size)
            self.assertEqual(len(reference_directions(case)), size)


class MetricTests(unittest.TestCase):
    def test_metrics_use_pymoo_gd_plus_and_spacing(self):
        problem = make_problem("dtlz2", 3)
        suite = MetricSuite(problem, n_reference_points=30, hv_samples=1000)
        F = suite.ref_pf[:10] * 1.05

        result = suite.calculate(F)
        normalized = (F - suite.ideal) / np.maximum(suite.nadir - suite.ideal, 1e-12)

        self.assertNotIn("gd", result)
        self.assertAlmostEqual(result["gd_plus"], float(GDPlus(suite.ref_pf)(F)))
        self.assertAlmostEqual(result["spacing"], float(SpacingIndicator()(normalized)))

    def test_high_dimensional_reference_cache_across_problem_instances(self):
        for problem_name in ("dtlz1", "dtlz2", "dtlz3", "dtlz4"):
            for n_obj in (5, 10):
                problem_a = make_problem(problem_name, n_obj)
                suite_a = MetricSuite(
                    problem_a,
                    n_reference_points=100,
                    hv_samples=1000,
                )

                # 新 Problem 实例命中相同的进程内缓存键。
                problem_b = make_problem(problem_name, n_obj)
                suite_b = MetricSuite(
                    problem_b,
                    n_reference_points=100,
                    hv_samples=1000,
                )

                np.testing.assert_allclose(suite_a.ref_pf, suite_b.ref_pf)
                np.testing.assert_allclose(suite_a.ideal, suite_b.ideal)
                np.testing.assert_allclose(suite_a.nadir, suite_b.nadir)

    def test_high_dimensional_hv_is_deterministic_monte_carlo(self):
        problem = make_problem("dtlz2", 8)
        suite_a = MetricSuite(problem, n_reference_points=30, hv_samples=1000)
        suite_b = MetricSuite(problem, n_reference_points=30, hv_samples=1000)
        F = suite_a.ref_pf[:10]
        self.assertEqual(suite_a.hv_method, "monte_carlo")
        self.assertEqual(suite_a.calculate(F)["hv"], suite_b.calculate(F)["hv"])

    def test_dtlz7_and_wfg_reference_fronts_are_bounded_and_reproducible(self):
        for problem_name in ("dtlz7", "wfg1", "wfg2", "wfg4", "wfg9"):
            for n_obj in (3, 5, 10):
                problem_a = make_problem(problem_name, n_obj)
                problem_b = make_problem(problem_name, n_obj)
                suite_a = MetricSuite(problem_a, n_reference_points=100, hv_samples=1000)
                suite_b = MetricSuite(problem_b, n_reference_points=100, hv_samples=1000)

                self.assertLessEqual(len(suite_a.ref_pf), 100)
                np.testing.assert_allclose(suite_a.ref_pf, suite_b.ref_pf)


class StructureHelperTests(unittest.TestCase):
    def test_objective_normalization_is_scale_invariant(self):
        F = np.asarray([
            [1.0, 8.0, 3.0],
            [2.0, 5.0, 7.0],
            [4.0, 2.0, 9.0],
        ])
        weight = np.asarray([0.2, 0.3, 0.5])
        scaled = F * np.asarray([100.0, 0.01, 7.0]) + np.asarray([9.0, -3.0, 20.0])
        original_context = ObjectiveNormalization.from_objectives(F)
        scaled_context = ObjectiveNormalization.from_objectives(scaled)

        np.testing.assert_allclose(
            original_context.tchebycheff(F, weight),
            scaled_context.tchebycheff(scaled, weight),
        )

    def test_reference_subset_is_deterministic_and_contains_axes(self):
        case = ExperimentCase("IEMOEC", "dtlz2", 5, 3, 420)
        ref_dirs = reference_directions(case)
        first = reference_direction_subset(ref_dirs, 5, 10)
        second = reference_direction_subset(ref_dirs, 5, 10)

        np.testing.assert_allclose(first, second)
        axes = [direction_objective(weight) for weight in first]
        self.assertEqual({axis for axis in axes if axis is not None}, set(range(5)))

        four_m = reference_direction_subset(ref_dirs, 5, 20)
        self.assertEqual(len(four_m), 20)
        four_m_axes = [direction_objective(weight) for weight in four_m]
        self.assertEqual(
            {axis for axis in four_m_axes if axis is not None},
            set(range(5)),
        )

    def test_shared_initialization_is_used_by_baseline_factory(self):
        case = ExperimentCase("NSGA2", "dtlz2", 3, 5, 182)
        problem = make_problem("dtlz2", 3)
        pop_size = len(reference_directions(case))
        X = shared_initial_decisions(problem, pop_size, case.seed)
        algorithm, _, _ = make_baseline(case, initial_X=X)

        np.testing.assert_allclose(algorithm.initialization.sampling, X)

    def test_vargha_delaney_reports_target_superiority(self):
        target = np.asarray([1.0, 2.0, 3.0])
        competitor = np.asarray([4.0, 5.0, 6.0])
        self.assertEqual(vargha_delaney_a12(target, competitor, True), 1.0)
        self.assertEqual(vargha_delaney_a12(target, competitor, False), 0.0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.output = self.temp.name
        self.small_iemoec = IEMOECConfig(
            island_population=4,
            inner_generations_early=1,
            inner_generations_late=1,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_iemoec_rejects_zero_inner_generations(self):
        for field_name in ("inner_generations_early", "inner_generations_late"):
            values = {field_name: 0}
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "岛内演化代数"):
                    IEMOECConfig(**values).validate()

    def test_iemoec_rejects_invalid_origin_ratio(self):
        for origin_ratio in (0.0, 1.01):
            with self.subTest(origin_ratio=origin_ratio):
                with self.assertRaisesRegex(ValueError, "origin_ratio"):
                    IEMOECConfig(origin_ratio=origin_ratio).validate()

    def test_iemoec_rejects_invalid_island_initialization(self):
        with self.assertRaisesRegex(ValueError, "island_initialization"):
            IEMOECConfig(island_initialization="unknown").validate()

        for field_name in ("direction_neighbor_ancestors", "diverse_ancestors"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "多祖先数量"):
                    IEMOECConfig(**{field_name: -1}).validate()

    def test_iemoec_rejects_invalid_recombination_ratio(self):
        invalid_configs = (
            {"recombination_budget_ratio": -0.01},
            {"recombination_budget_ratio": 1.01},
            {"late_recombination_budget_ratio": -0.01},
            {"late_recombination_budget_ratio": 1.01},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "recombination_budget_ratio"):
                    IEMOECConfig(**values).validate()

    def test_iemoec_defaults_to_one_inner_generation(self):
        config = IEMOECConfig()
        self.assertEqual(config.inner_generations_early, 1)
        self.assertEqual(config.inner_generations_late, 1)
        self.assertEqual(config.origin_ratio, 0.2)
        self.assertEqual(config.recombination_budget_ratio, 1.0)
        self.assertEqual(config.late_recombination_budget_ratio, 0.25)
        self.assertEqual(config.island_initialization, "multi_ancestor")
        self.assertEqual(config.direction_neighbor_ancestors, 4)
        self.assertEqual(config.diverse_ancestors, 2)
        self.assertFalse(config.retain_island_state)

    def test_iemoec_variant_profiles_and_labels(self):
        v0 = IEMOECConfig.for_variant("v0")
        s1 = IEMOECConfig.for_variant("s1")
        candidate = IEMOECConfig.for_variant("candidate")
        self.assertEqual(v0.island_initialization, "single_ancestor")
        self.assertEqual(s1.island_initialization, "multi_ancestor")
        self.assertEqual(candidate.initialization_mode, "shared_population")
        self.assertEqual(candidate.normalization_mode, "global")
        self.assertEqual(candidate.fe_scheduler, "fixed_batch")
        self.assertEqual(candidate.algorithm_schema_version, 2)

        expected = {
            "rank": "IEMOEC-Rank",
            "rank_crowding": "IEMOEC-CD",
            "nsga3": "IEMOEC-RD",
        }
        for survival, label in expected.items():
            config = IEMOECConfig.for_variant(
                "candidate",
                outer_survival=survival,
            )
            case = ExperimentCase("IEMOEC", "dtlz2", 3, 1, 182, iemoec=config)
            self.assertEqual(case.algorithm_label, label)

    def case(self, algorithm: str, seed: int = 7):
        return ExperimentCase(
            algorithm, "dtlz2", 3, seed, 182,
            output_root=self.output,
            history_points=3,
            reference_points=30,
            iemoec=self.small_iemoec,
        )

    def test_every_algorithm_obeys_identical_fe_budget(self):
        for algorithm in ("NSGA2", "NSGA3", "MOEAD", "IEMOEC"):
            case = self.case(algorithm)
            result = run_case(case, force=True)
            self.assertEqual(result["n_eval"], 182)
            self.assertEqual(result["reference_population_size"], 91)
            self.assertEqual(result["population_size"], 91)
            self.assertTrue(np.isfinite(result["igd_plus"]))
            self.assertEqual(result["metric_schema_version"], 4)
            self.assertIn("gd_plus", result)
            self.assertNotIn("gd", result)
            with (case.output_dir / "history.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                history = list(csv.DictReader(handle))
            final_history = history[-1]
            self.assertEqual(int(final_history["fe"]), result["n_eval"])
            self.assertAlmostEqual(float(final_history["igd_plus"]), result["igd_plus"])
            self.assertAlmostEqual(float(final_history["hv"]), result["hv"])
            checkpoints = [row for row in history if row["event"] == "checkpoint"]
            expected_fes = np.linspace(182 / 3, 182, 3, dtype=int)[:-1]
            self.assertEqual(
                [int(row["fe"]) for row in checkpoints],
                expected_fes.tolist(),
            )
            self.assertTrue(
                all(int(row["observed_fe"]) >= int(row["fe"]) for row in checkpoints)
            )

            if algorithm == "IEMOEC":
                self.assertEqual(result["origin_population_size"], 20)
                self.assertEqual(result["island_initialization"], "multi_ancestor")
                self.assertEqual(result["island_expansion_fes_total"], 0)
                later_checkpoints = [
                    row for row in checkpoints if int(row["fe"]) > 91
                ]
                self.assertTrue(later_checkpoints)
                self.assertTrue(
                    all(int(row["population_size"]) > 4 for row in later_checkpoints)
                )

    def test_seed_is_reproducible_and_completed_case_is_skipped(self):
        for algorithm in ("NSGA2", "IEMOEC"):
            case = self.case(algorithm)
            run_case(case, force=True)
            with (case.output_dir / "final_population.csv").open("rb") as handle:
                population_bytes = handle.read()
            second = run_case(case)
            self.assertEqual(second["status"], "skipped")
            run_case(case, force=True)
            with (case.output_dir / "final_population.csv").open("rb") as handle:
                self.assertEqual(population_bytes, handle.read())

    def test_changed_configuration_requires_new_run_name_even_with_force(self):
        original = self.case("NSGA2")
        run_case(original, force=True)
        changed = ExperimentCase(
            "NSGA2", "dtlz2", 3, 7, 273,
            output_root=self.output,
            history_points=3,
            reference_points=30,
            iemoec=self.small_iemoec,
        )
        with self.assertRaisesRegex(RuntimeError, "不同配置"):
            run_case(changed)
        with self.assertRaisesRegex(RuntimeError, "不同配置"):
            run_case(changed, force=True)

    def test_different_algorithm_schema_cannot_overwrite_results(self):
        v0 = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 7, 182,
            output_root=self.output,
            history_points=3,
            reference_points=30,
            iemoec=IEMOECConfig.for_variant("v0", island_population=4),
        )
        candidate = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 7, 182,
            output_root=self.output,
            history_points=3,
            reference_points=30,
            iemoec=IEMOECConfig.for_variant("candidate", island_population=4),
        )
        run_case(v0, force=True)
        with self.assertRaisesRegex(RuntimeError, "algorithm schema"):
            run_case(candidate, force=True)

    def test_different_metric_schema_requires_new_run_name(self):
        case = self.case("NSGA2")
        run_case(case, force=True)
        metrics_path = case.output_dir / "metrics.json"
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["metric_schema_version"] = 3
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

        with self.assertRaisesRegex(RuntimeError, "旧 metric schema"):
            run_case(case, force=True)

    def test_common_initialization_matches_all_four_algorithms(self):
        hashes = []
        for algorithm in ("NSGA2", "NSGA3", "MOEAD", "IEMOEC"):
            config = (
                IEMOECConfig.for_variant("candidate", island_population=4)
                if algorithm == "IEMOEC"
                else self.small_iemoec
            )
            case = ExperimentCase(
                algorithm, "dtlz2", 3, 29, 182,
                output_root=str(Path(self.output) / algorithm),
                history_points=3,
                reference_points=30,
                iemoec=config,
            )
            hashes.append(run_case(case, force=True)["initialization_hash"])
        self.assertEqual(len(set(hashes)), 1)

    def test_iemoec_ablation_without_recombination_obeys_budget(self):
        config = IEMOECConfig(
            island_population=4,
            inner_generations_early=1,
            inner_generations_late=1,
            outer_survival="rank",
            use_crowding=True,
            enable_recombination=False,
        )
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 11, 182,
            output_root=self.output, history_points=3, reference_points=30,
            iemoec=config,
        )
        result = run_case(case, force=True)
        self.assertEqual(result["n_eval"], 182)
        diagnostics_path = case.output_dir / "iemoec_diagnostics.csv"
        with diagnostics_path.open(encoding="utf-8-sig", newline="") as handle:
            diagnostics = list(csv.DictReader(handle))
        self.assertEqual(len(diagnostics), result["outer_iterations"])
        self.assertEqual(
            sum(int(row["island_fes"]) for row in diagnostics),
            result["island_fes_total"],
        )
        self.assertTrue(
            all(row["island_state_reused"] == "False" for row in diagnostics)
        )

    def test_iemoec_can_retain_island_state_between_outer_iterations(self):
        config = IEMOECConfig(
            variant="v0",
            island_population=4,
            island_initialization="single_ancestor",
            retain_island_state=True,
        )
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 13, 273,
            output_root=self.output, history_points=3, reference_points=30,
            iemoec=config,
        )
        result = run_case(case, force=True)
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            diagnostics = list(csv.DictReader(handle))

        self.assertGreater(len(diagnostics), 1)
        self.assertGreater(int(diagnostics[0]["expansion_fes"]), 0)
        self.assertTrue(
            all(row["island_state_reused"] == "True" for row in diagnostics[1:])
        )
        self.assertTrue(all(int(row["expansion_fes"]) == 0 for row in diagnostics[1:]))
        self.assertEqual(
            result["island_expansion_fes_total"],
            int(diagnostics[0]["expansion_fes"]),
        )

    def test_multi_ancestor_islands_rebuild_from_evaluated_global_pool(self):
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 19, 364,
            output_root=self.output,
            history_points=3,
            reference_points=30,
            iemoec=self.small_iemoec,
        )
        result = run_case(case, force=True)
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            diagnostics = list(csv.DictReader(handle))

        self.assertGreater(len(diagnostics), 1)
        self.assertEqual(result["island_expansion_fes_total"], 0)
        self.assertTrue(
            all(row["island_initialization"] == "multi_ancestor" for row in diagnostics)
        )
        source_sizes = [
            int(row["island_source_population_size"]) for row in diagnostics
        ]
        self.assertEqual(source_sizes[0], 20)
        self.assertEqual(source_sizes, sorted(source_sizes))
        self.assertEqual(source_sizes[-1], 91)
        required = {
            "island_count", "founders_per_island", "unique_founder_ratio",
            "mean_founder_distance_x", "mean_founder_distance_f",
            "merged_unique_ratio", "local_offspring",
            "recombination_offspring", "recombination_unique_ratio",
            "local_survival_rate", "recombination_survival_rate",
            "direction_occupancy", "empty_direction_ratio", "outer_batch_fes",
        }
        self.assertTrue(required.issubset(diagnostics[0]))

    def test_candidate_diagnostics_and_fixed_fe_allocation(self):
        config = IEMOECConfig.for_variant("candidate", island_population=4)
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 31, 250,
            output_root=self.output,
            history_points=4,
            reference_points=30,
            iemoec=config,
        )
        result = run_case(case, force=True)
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            diagnostics = list(csv.DictReader(handle))

        self.assertEqual(result["n_eval"], 250)
        self.assertEqual(result["global_selection_count"], len(diagnostics))
        self.assertTrue(all(float(row["unique_founder_ratio"]) == 1.0 for row in diagnostics))
        self.assertTrue(all(int(row["expansion_fes"]) == 0 for row in diagnostics))
        self.assertTrue(all(int(row["island_count"]) == 6 for row in diagnostics))
        self.assertTrue(all(
            int(row["local_offspring"]) + int(row["recombination_offspring"])
            == int(row["outer_batch_fes"])
            for row in diagnostics
        ))
        self.assertEqual(sum(int(row["outer_batch_fes"]) for row in diagnostics), 159)
        required = {
            "founders_per_island", "mean_founder_distance_x",
            "mean_founder_distance_f", "merged_unique_ratio",
            "recombination_unique_ratio",
            "local_survival_rate", "recombination_survival_rate",
            "direction_occupancy", "empty_direction_ratio",
        }
        self.assertTrue(required.issubset(diagnostics[0]))

    def test_candidate_founders_include_origin_anchor_without_evaluation(self):
        config = IEMOECConfig.for_variant("candidate", island_population=6)
        case = ExperimentCase("IEMOEC", "dtlz2", 3, 47, 182, iemoec=config)
        problem = make_problem("dtlz2", 3)
        X = shared_initial_decisions(problem, 91, case.seed)
        runner = IEMOECRunner(problem, case, initial_X=X)
        runner._initialize_candidate()
        before = runner.n_eval
        normalization = ObjectiveNormalization.from_objectives(
            runner.candidate_pool.get("F")
        )
        definitions = runner._island_definitions()
        islands, _ = runner._create_islands(definitions, normalization)
        origin_keys = {runner._x_key(item.get("X")) for item in runner.origin}

        self.assertEqual(runner.n_eval, before)
        for island in islands:
            keys = [runner._x_key(item.get("X")) for item in island]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(any(key in origin_keys for key in keys))

    def test_candidate_never_reevaluates_known_decision(self):
        config = IEMOECConfig.for_variant("candidate", island_population=4)
        case = ExperimentCase("IEMOEC", "dtlz2", 3, 53, 182, iemoec=config)
        problem = make_problem("dtlz2", 3)
        X = shared_initial_decisions(problem, 91, case.seed)
        runner = IEMOECRunner(problem, case, initial_X=X)
        runner._initialize_candidate()
        before = runner.n_eval
        duplicates = Population.new("X", X[:3].copy())

        evaluated = runner._evaluate_unique(duplicates)

        self.assertEqual(len(evaluated), 0)
        self.assertEqual(runner.n_eval, before)

    def test_candidate_merge_deduplicates_decisions(self):
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 37, 182,
            iemoec=IEMOECConfig.for_variant("candidate", island_population=4),
        )
        problem = make_problem("dtlz2", 3)
        X = shared_initial_decisions(problem, 91, case.seed)
        runner = IEMOECRunner(problem, case, initial_X=X)
        population = Population.new(
            "X", np.vstack([X[:3], X[1:3]]),
            "F", problem.evaluate(np.vstack([X[:3], X[1:3]])),
        )
        unique, duplicates, ratio = runner._deduplicate_population(population)

        self.assertEqual(len(unique), 3)
        self.assertEqual(len(duplicates), 2)
        self.assertAlmostEqual(ratio, 3 / 5)

    def test_pairing_strategies_are_reproducible(self):
        representatives = Population.new(
            "X", np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "F", np.asarray([[0.1, 1.0], [0.4, 0.6], [0.7, 0.3], [1.0, 0.1]]),
        )
        weights = [
            np.asarray([1.0, 0.0]), np.asarray([0.7, 0.3]),
            np.asarray([0.3, 0.7]), np.asarray([0.0, 1.0]),
        ]
        for strategy in (
            "farthest_weight", "nearest_weight", "random", "farthest_decision"
        ):
            config = IEMOECConfig.for_variant(
                "candidate",
                pairing_strategy=strategy,
                partners_per_elite=1,
            )
            case = ExperimentCase("IEMOEC", "zdt1", 2, 41, 200, iemoec=config)
            problem = make_problem("zdt1", 2, n_var=2)
            first = IEMOECRunner(problem, case)
            second = IEMOECRunner(problem, case)
            np.testing.assert_array_equal(
                first._representative_pairs(representatives, weights),
                second._representative_pairs(representatives, weights),
            )

    def test_candidate_dtlz2_and_wfg1_integration(self):
        for problem_name in ("dtlz2", "wfg1"):
            for n_obj in (3, 5, 10):
                with self.subTest(problem=problem_name, n_obj=n_obj):
                    probe = ExperimentCase("IEMOEC", problem_name, n_obj, 43, 1)
                    pop_size = len(reference_directions(probe))
                    case = ExperimentCase(
                        "IEMOEC", problem_name, n_obj, 43, pop_size + 12,
                        output_root=self.output,
                        history_points=2,
                        reference_points=30,
                        high_dim_hv_samples=1000,
                        iemoec=IEMOECConfig.for_variant(
                            "candidate",
                            island_population=4,
                        ),
                    )
                    result = run_case(case, force=True)
                    self.assertEqual(result["n_eval"], pop_size + 12)
                    with (case.output_dir / "history.csv").open(
                        encoding="utf-8-sig", newline=""
                    ) as handle:
                        final = list(csv.DictReader(handle))[-1]
                    self.assertAlmostEqual(float(final["igd_plus"]), result["igd_plus"])
                    self.assertAlmostEqual(float(final["hv"]), result["hv"])

    def test_iemoec_recombination_obeys_ratio_budget(self):
        config = IEMOECConfig(
            island_population=4,
            recombination_budget_ratio=0.1,
            late_recombination_budget_ratio=0.1,
        )
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 17, 273,
            output_root=self.output, history_points=3, reference_points=30,
            iemoec=config,
        )
        run_case(case, force=True)
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            diagnostics = list(csv.DictReader(handle))

        expected_budget = math.ceil(len(reference_directions(case)) * 0.1)
        offspring_counts = [int(row["recombination_offspring"]) for row in diagnostics]
        self.assertTrue(any(count > 0 for count in offspring_counts))
        self.assertTrue(all(count <= expected_budget for count in offspring_counts))


if __name__ == "__main__":
    unittest.main()
