from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymoo.decomposition.pbi import PBI
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.core.population import Population

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iemoec_experiment.config import ExperimentCase, IEMOECConfig
from iemoec_experiment.constraints import feasibility_first_order
from iemoec_experiment.directional_memory import DirectionalMemory
from iemoec_experiment.directions import direction_objective, reference_direction_subset
from iemoec_experiment.factory import make_baseline, reference_directions
from iemoec_experiment.iemoec import IEMOECRunner
from iemoec_experiment.initialization import shared_initial_decisions
from iemoec_experiment.manifest import build_manifest, validate_result_rows
from iemoec_experiment.metrics import MetricSuite
from iemoec_experiment.normalization import ObjectiveNormalization
from iemoec_experiment.problems import make_problem, standard_problem_dimensions
from iemoec_experiment.runner import HistoryRecorder, run_case
from iemoec_experiment.source_budget import SourceBudgetController
from run import PRESETS, build_parser, resolve_cases
from summarize import (
    audit_constraint_rows,
    constraint_summary,
    feasibility_comparison,
    vargha_delaney_a12,
)


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
        problem = make_problem("convex-dtlz2", 3)
        F = problem.evaluate(np.full((4, problem.n_var), 0.5))
        self.assertEqual(F.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(F)))

    def test_ambiguous_cdtlz2_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "有歧义"):
            make_problem("c-dtlz2", 3)

    def test_constrained_problem_matrix_and_dascmop_difficulty(self):
        for name in ("c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz4"):
            for n_obj in (3, 5, 8, 10, 15):
                with self.subTest(problem=name, n_obj=n_obj):
                    problem = make_problem(name, n_obj)
                    self.assertTrue(problem.has_constraints())
                    self.assertEqual(problem.n_obj, n_obj)

        expected = {
            4: (0.25, 0.25, 0.25),
            8: (0.50, 0.50, 0.50),
            12: (0.75, 0.75, 0.75),
            16: (0.50, 1.00, 0.50),
        }
        for difficulty, factors in expected.items():
            problem = make_problem(f"dascmop7_d{difficulty}", 3)
            self.assertEqual(
                (problem.eta, problem.zeta, problem.gamma),
                factors,
            )

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
        self.assertEqual(result["hv_reference_point"], 1.1)
        self.assertIn("hv_eligible_solution_count", result)

    def test_hv_eligible_count_distinguishes_zero_volume_points(self):
        problem = make_problem("dtlz2", 3)
        suite = MetricSuite(
            problem,
            n_reference_points=30,
            hv_samples=1000,
            hv_reference_point=1.1,
        )
        F = np.vstack([suite.ideal, suite.nadir + 10.0])

        result = suite.calculate(F)

        self.assertEqual(result["hv_eligible_solution_count"], 1)
        self.assertGreater(result["hv"], 0.0)

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
        for problem_name in (
            "dtlz5", "dtlz6", "dtlz7",
            "wfg1", "wfg2", "wfg3", "wfg4", "wfg5",
            "wfg6", "wfg7", "wfg8", "wfg9",
        ):
            for n_obj in (3, 5, 10):
                problem_a = make_problem(problem_name, n_obj)
                problem_b = make_problem(problem_name, n_obj)
                suite_a = MetricSuite(problem_a, n_reference_points=100, hv_samples=1000)
                suite_b = MetricSuite(problem_b, n_reference_points=100, hv_samples=1000)

                self.assertLessEqual(len(suite_a.ref_pf), 100)
                np.testing.assert_allclose(suite_a.ref_pf, suite_b.ref_pf)

    def test_constrained_metrics_ignore_infeasible_objectives(self):
        problem = make_problem("c2dtlz2", 3)
        directions = np.eye(3)
        suite = MetricSuite(
            problem,
            n_reference_points=30,
            hv_samples=1000,
            direction_directions=directions,
        )
        feasible_F = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mixed_F = np.vstack([feasible_F, np.zeros(3)])

        feasible_only = suite.calculate(feasible_F, CV=np.zeros((2, 1)))
        mixed = suite.calculate(mixed_F, CV=np.asarray([[0.0], [0.0], [9.0]]))

        self.assertEqual(mixed["feasible_count"], 2)
        self.assertAlmostEqual(mixed["feasible_ratio"], 2 / 3)
        self.assertEqual(mixed["igd_plus"], feasible_only["igd_plus"])
        self.assertEqual(mixed["hv"], feasible_only["hv"])
        self.assertEqual(
            mixed["feasible_direction_coverage"],
            mixed["direction_occupancy"],
        )

    def test_constrained_metrics_have_explicit_no_feasible_state(self):
        problem = make_problem("c1dtlz1", 3)
        suite = MetricSuite(
            problem,
            n_reference_points=30,
            hv_samples=1000,
            direction_directions=np.eye(3),
        )
        result = suite.calculate(
            np.asarray([[0.1, 0.1, 0.1]]),
            CV=np.asarray([[1.0]]),
        )

        self.assertFalse(result["has_feasible"])
        self.assertIsNone(result["igd_plus"])
        self.assertEqual(result["hv"], 0.0)
        self.assertEqual(result["feasible_direction_coverage"], 0.0)

    def test_c2_reference_front_keeps_high_dimensional_coverage(self):
        problem = make_problem("c2dtlz2", 15)
        suite = MetricSuite(problem, n_reference_points=100, hv_samples=1000)

        self.assertEqual(len(suite.ref_pf), 100)
        np.testing.assert_allclose(suite.ideal, np.zeros(15), atol=1e-12)
        np.testing.assert_allclose(suite.nadir, np.ones(15), atol=1e-12)

    def test_all_frozen_dascmop_reference_fronts_are_bundled(self):
        for problem_number in (7, 8, 9):
            for difficulty in (4, 8, 12, 16):
                with self.subTest(
                    problem=problem_number,
                    difficulty=difficulty,
                ):
                    suite = MetricSuite(
                        make_problem(
                            f"dascmop{problem_number}_d{difficulty}",
                            3,
                        ),
                        n_reference_points=10,
                        hv_samples=1000,
                    )
                    self.assertEqual(len(suite.ref_pf), 10)
                    self.assertEqual(
                        suite.reference_front_method,
                        "bundled_pymoo_data_pf",
                    )


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

    def test_moead_pbi_uses_pbi_decomposition(self):
        case = ExperimentCase("MOEADPBI", "dtlz2", 3, 5, 182)
        problem = make_problem("dtlz2", 3)
        pop_size = len(reference_directions(case))
        X = shared_initial_decisions(problem, pop_size, case.seed)

        algorithm, _, _ = make_baseline(case, initial_X=X)

        self.assertIsInstance(algorithm.decomposition, PBI)
        self.assertEqual(algorithm.decomposition.theta, 5.0)
        np.testing.assert_allclose(algorithm.initialization.sampling, X)

    def test_stable_age_matches_original_geometry_and_guards_zero_norm(self):
        original, _, _ = make_baseline(
            ExperimentCase("AGEMOEA2", "dtlz2", 3, 5, 182)
        )
        stable_case = ExperimentCase("AGEMOEA2STABLE", "dtlz2", 3, 5, 182)
        stable, _, _ = make_baseline(stable_case)
        front = np.asarray([
            [0.1, 0.7, 0.9],
            [0.5, 0.4, 0.8],
            [0.8, 0.6, 0.2],
        ])

        np.testing.assert_allclose(
            stable.survival.pairwise_distances(front, 2.0),
            original.survival.pairwise_distances(front, 2.0),
        )
        guarded = stable.survival.pairwise_distances(
            np.vstack([np.zeros(3), -np.ones(3), front]),
            1.5,
        )
        self.assertTrue(np.all(np.isfinite(guarded)))
        self.assertEqual(stable_case.algorithm_schema_version, 3)
        self.assertEqual(stable_case.algorithm_variant, "stable_zero_norm_guard")

    def test_vargha_delaney_reports_target_superiority(self):
        target = np.asarray([1.0, 2.0, 3.0])
        competitor = np.asarray([4.0, 5.0, 6.0])
        self.assertEqual(vargha_delaney_a12(target, competitor, True), 1.0)
        self.assertEqual(vargha_delaney_a12(target, competitor, False), 0.0)

    def test_constraint_summary_keeps_failed_runs_visible(self):
        def row(algorithm, seed, feasible, min_cv, first_feasible_fe):
            return {
                "problem": "c1dtlz1",
                "n_obj": 3,
                "algorithm": algorithm,
                "algorithm_label": algorithm,
                "seed": seed,
                "population_size": 10,
                "feasible_count": 2 if feasible else 0,
                "feasible_ratio": 0.2 if feasible else 0.0,
                "feasible_direction_coverage": 0.1 if feasible else 0.0,
                "has_feasible": feasible,
                "min_cv": min_cv,
                "mean_cv": min_cv + 0.5,
                "first_feasible_fe": first_feasible_fe,
                "max_fes": 100,
                "igd_plus": 0.5 if feasible else None,
                "gd_plus": 0.4 if feasible else None,
                "hv": 0.2 if feasible else 0.0,
            }

        rows = [
            row("IEMOEC", 1, True, 0.0, 40),
            row("IEMOEC", 2, False, 1.0, None),
            row("NSGA2", 1, False, 3.0, None),
            row("NSGA2", 2, False, 2.0, None),
        ]

        audit = audit_constraint_rows(rows)
        summaries = constraint_summary(rows)
        comparison = feasibility_comparison(rows, "IEMOEC")[0]

        self.assertEqual(audit["no_feasible_rows"], 3)
        iemoec = next(row for row in summaries if row["algorithm"] == "IEMOEC")
        self.assertEqual(iemoec["feasible_runs"], 1)
        self.assertEqual(iemoec["quality_valid_runs"], 1)
        self.assertEqual(iemoec["feasibility_success_rate"], 0.5)
        self.assertEqual(comparison["target_feasibility_wins"], 1)
        self.assertEqual(comparison["both_infeasible"], 1)
        self.assertEqual(
            comparison["target_lower_cv_when_both_infeasible"],
            1,
        )

        rows[1]["igd_plus"] = 99.0
        with self.assertRaisesRegex(RuntimeError, "质量指标定义错误"):
            audit_constraint_rows(rows)

        self.assertEqual(
            feasibility_comparison([
                {
                    "problem": "dtlz2",
                    "n_obj": 3,
                    "algorithm": "IEMOEC",
                    "seed": 1,
                }
            ], "IEMOEC"),
            [],
        )


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.output = self.temp.name
        self.small_iemoec = IEMOECConfig(
            island_population=4,
            inner_generations_early=1,
            inner_generations_late=1,
        )

    def test_formal_preset_has_expected_16800_tasks(self):
        preset = PRESETS["formal"]
        task_count = (
            len(preset["problems"])
            * len(preset["objectives"])
            * len(preset["seeds"])
            * len(preset["algorithms"])
        )
        self.assertEqual(task_count, 16_800)

    def test_s3_development_preset_has_expected_100_tasks(self):
        preset = PRESETS["s3_development"]
        task_count = (
            len(preset["problems"])
            * len(preset["objectives"])
            * len(preset["seeds"])
            * len(preset["algorithms"])
        )
        self.assertEqual(task_count, 100)

        args = build_parser().parse_args([
            "--preset", "s3_development",
            "--iemoec-variant", "s3_memory",
            "--run-name", "s3_memory_m8_m15",
            "--dry-run",
        ])
        cases = resolve_cases(args)
        self.assertEqual(len(cases), 100)
        self.assertTrue(all(case.algorithm_schema_version == 5 for case in cases))

        no_recombination_args = build_parser().parse_args([
            "--preset", "smoke",
            "--algorithms", "IEMOEC",
            "--iemoec-variant", "s3_elite",
            "--no-recombination",
        ])
        no_recombination_case = resolve_cases(no_recombination_args)[0]
        no_recombination_case.validate()
        self.assertEqual(
            no_recombination_case.iemoec.recombination_fe_ratio,
            0.0,
        )

    def test_constrained_presets_match_frozen_experiment_sizes(self):
        expected = {
            "constrained_smoke": (48, 4, range(1, 4), 50),
            "constrained_pilot": (960, 32, range(1, 6), 200),
            "constrained_formal": (5760, 32, range(31, 61), 400),
        }
        for preset, (task_count, scenario_count, seeds, budget_scale) in expected.items():
            with self.subTest(preset=preset):
                args = build_parser().parse_args([
                    "--preset", preset,
                    "--dry-run",
                ])
                cases = resolve_cases(args)
                self.assertEqual(len(cases), task_count)
                self.assertEqual(
                    len({(case.problem, case.n_obj) for case in cases}),
                    scenario_count,
                )
                self.assertEqual({case.seed for case in cases}, set(seeds))
                self.assertTrue(all(
                    case.max_fes
                    == len(reference_directions(case)) * budget_scale
                    for case in cases
                ))
                self.assertTrue(all(
                    case.iemoec.variant == "s3_elite_constrained"
                    for case in cases
                ))
                if preset != "constrained_smoke":
                    self.assertEqual(
                        {case.algorithm for case in cases},
                        {
                            "NSGA2", "NSGA3", "RVEA", "AGEMOEA2STABLE",
                            "CTAEA", "IEMOEC",
                        },
                    )

    def test_constrained_schema_label_and_incompatible_algorithms(self):
        config = IEMOECConfig.for_variant("s3_elite_constrained")
        case = ExperimentCase(
            "IEMOEC", "c2dtlz2", 3, 1, 182, iemoec=config
        )
        case.validate()
        self.assertEqual(case.algorithm_schema_version, 11)
        self.assertEqual(case.algorithm_label, "IEMOEC-C")

        with self.assertRaisesRegex(ValueError, "MOEA/D 不支持约束"):
            ExperimentCase("MOEAD", "c2dtlz2", 3, 1, 182).validate()
        with self.assertRaisesRegex(ValueError, "s3_elite_constrained"):
            ExperimentCase(
                "IEMOEC",
                "c2dtlz2",
                3,
                1,
                182,
                iemoec=IEMOECConfig.for_variant("s4"),
            ).validate()

    def test_first_feasible_fe_uses_position_inside_evaluation_batch(self):
        suite = MetricSuite(
            make_problem("c1dtlz1", 3),
            n_reference_points=30,
            hv_samples=1000,
        )
        recorder = HistoryRecorder(suite, 100, 2, False)
        population = Population.new(
            "X", np.zeros((3, 7)),
            "F", np.zeros((3, 3)),
            "CV", np.asarray([[2.0], [0.0], [0.0]]),
        )

        recorder.observe_evaluated(10, population)

        self.assertEqual(recorder.first_feasible_fe, 9)

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
        s2 = IEMOECConfig.for_variant("s2")
        self.assertEqual(v0.island_initialization, "single_ancestor")
        self.assertEqual(s1.island_initialization, "multi_ancestor")
        self.assertEqual(candidate.initialization_mode, "shared_population")
        self.assertEqual(candidate.normalization_mode, "global")
        self.assertEqual(candidate.fe_scheduler, "fixed_batch")
        self.assertEqual(candidate.algorithm_schema_version, 2)
        self.assertEqual(s2.algorithm_schema_version, 3)
        self.assertTrue(s2.uses_candidate_architecture)
        self.assertEqual(s2.island_direction_mode, "axis_random")
        self.assertEqual(s2.island_count_multiplier, 2)
        self.assertEqual(s2.outer_survival, "nsga3")
        self.assertEqual(s2.pairing_strategy, "farthest_weight")
        self.assertEqual(s2.local_fe_ratio, 0.75)
        self.assertEqual(s2.recombination_fe_ratio, 0.25)
        self.assertEqual(s2.outer_batch_ratio, 1.0)

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

    def test_s3_variant_profiles_have_isolated_schemas(self):
        expected = {
            "s3_memory": (5, 0.75, 0.0, False, False, "IEMOEC-RD-S3-Memory"),
            "s3_hybrid": (6, 0.50, 0.25, False, False, "IEMOEC-RD-S3-Hybrid"),
            "s3_elite": (7, 0.50, 0.25, True, False, "IEMOEC-RD-S3-Elite"),
            "s3": (8, 0.50, 0.25, True, True, "IEMOEC-RD-S3-Full"),
        }
        for variant, values in expected.items():
            with self.subTest(variant=variant):
                config = IEMOECConfig.for_variant(variant)
                config.validate()
                case = ExperimentCase(
                    "IEMOEC", "dtlz2", 3, 1, 182, iemoec=config
                )
                self.assertEqual(config.algorithm_schema_version, values[0])
                self.assertEqual(config.isolated_fe_ratio, values[1])
                self.assertEqual(config.shared_fe_ratio, values[2])
                self.assertEqual(config.protect_direction_elites, values[3])
                self.assertEqual(config.adaptive_source_budget, values[4])
                self.assertEqual(case.algorithm_label, values[5])

        s2_case = ExperimentCase(
            "IEMOEC",
            "dtlz2",
            3,
            1,
            182,
            iemoec=IEMOECConfig.for_variant("s2"),
        )
        self.assertNotIn("direction_memory", s2_case.to_dict()["iemoec"])

    def test_s3_rejects_invalid_source_budget(self):
        with self.assertRaisesRegex(ValueError, "三类来源 FE 比例"):
            IEMOECConfig.for_variant(
                "s3_hybrid",
                isolated_fe_ratio=0.60,
            ).validate()

    def test_source_budget_is_exact_and_adapts_to_contribution(self):
        controller = SourceBudgetController(
            {"isolated": 0.50, "shared": 0.25, "recombination": 0.25},
            adaptive=True,
        )
        initial = controller.allocate(91)
        for _ in range(6):
            controller.update(
                {"isolated": 0.1, "shared": 2.0, "recombination": 0.1}
            )
        adapted = controller.allocate(91)

        self.assertEqual(sum(initial.values()), 91)
        self.assertEqual(initial, {
            "isolated": 45,
            "shared": 23,
            "recombination": 23,
        })
        self.assertEqual(sum(adapted.values()), 91)
        self.assertGreater(adapted["shared"], initial["shared"])
        self.assertGreaterEqual(adapted["isolated"], math.floor(91 * 0.30))
        self.assertLessEqual(adapted["shared"], math.ceil(91 * 0.40))
        self.assertLessEqual(adapted["recombination"], math.ceil(91 * 0.30))

        without_recombination = SourceBudgetController(
            {"isolated": 2 / 3, "shared": 1 / 3, "recombination": 0.0},
            adaptive=True,
        ).allocate(91)
        self.assertEqual(sum(without_recombination.values()), 91)
        self.assertEqual(without_recombination["recombination"], 0)

    def test_direction_memory_persists_and_returns_independent_copies(self):
        problem = make_problem("zdt1", 2, n_var=2)
        X = np.asarray([
            [0.05, 0.10], [0.10, 0.30], [0.20, 0.50],
            [0.40, 0.70], [0.70, 0.20], [0.90, 0.90],
        ])
        population = Population.new("X", X, "F", problem.evaluate(X))
        normalization = ObjectiveNormalization.from_objectives(population.get("F"))
        memory = DirectionalMemory(
            problem,
            [np.asarray([1.0, 1e-3]), np.asarray([1e-3, 1.0])],
            capacity=3,
        )
        memory.update(population, normalization)
        before = {
            IEMOECRunner._x_key(individual.get("X"))
            for individual in memory.combined_population()
        }

        memory.update(population[-2:], normalization)
        after = {
            IEMOECRunner._x_key(individual.get("X"))
            for individual in memory.combined_population()
        }
        pools = memory.parent_pools()
        identities = [id(individual) for pool in pools for individual in pool]
        pools[0][0].get("X")[0] = -1.0
        after_external_mutation = {
            IEMOECRunner._x_key(individual.get("X"))
            for individual in memory.combined_population()
        }

        self.assertTrue(before & after)
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(after_external_mutation, after)

    def test_feasibility_first_order_and_memory_never_promote_infeasible(self):
        problem = make_problem("c2dtlz2", 3)
        decisions = np.zeros((3, problem.n_var))
        decisions[:, 0] = np.asarray([0.1, 0.2, 0.3])
        population = Population.new(
            "X", decisions,
            "F", np.asarray([
                [0.0, 0.0, 0.0],
                [0.8, 0.2, 0.2],
                [0.9, 0.1, 0.2],
            ]),
            "CV", np.asarray([[0.01], [0.0], [0.0]]),
        )
        scores = np.asarray([0.0, 0.8, 0.9])

        order = feasibility_first_order(population, scores)
        self.assertEqual(order.tolist(), [1, 2, 0])

        memory = DirectionalMemory(
            problem,
            [np.asarray([1.0, 1e-3, 1e-3])],
            capacity=2,
            constraint_aware=True,
        )
        normalization = ObjectiveNormalization.from_objectives(
            population.get("F")
        )
        memory.update(population, normalization)
        selected_cv = memory.parent_pools()[0].get("CV").reshape(-1)
        np.testing.assert_allclose(selected_cv, 0.0)

    def test_constrained_elite_protection_excludes_infeasible_solutions(self):
        config = IEMOECConfig.for_variant("s3_elite_constrained")
        case = ExperimentCase(
            "IEMOEC", "c2dtlz2", 3, 1, 182, iemoec=config
        )
        problem = make_problem("c2dtlz2", 3)
        runner = IEMOECRunner(problem, case)
        population = Population.new(
            "X", np.zeros((3, problem.n_var)),
            "F", np.asarray([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            "CV", np.asarray([[1.0], [0.0], [0.0]]),
        )
        normalization = ObjectiveNormalization.from_objectives(
            population[1:].get("F")
        )
        protected = runner._eligible_direction_elites(
            population,
            [np.asarray([1.0, 1e-3, 1e-3])],
            normalization,
        )

        self.assertGreater(len(protected), 0)
        np.testing.assert_allclose(protected.get("CV"), 0.0)

    def test_constrained_baselines_and_iemoec_complete_small_run(self):
        initialization_hashes = []
        for algorithm in (
            "NSGA2", "NSGA3", "RVEA", "AGEMOEA2STABLE", "CTAEA", "IEMOEC",
        ):
            with self.subTest(algorithm=algorithm):
                case = ExperimentCase(
                    algorithm,
                    "c2dtlz2",
                    3,
                    5,
                    182,
                    output_root=str(Path(self.output) / "constrained"),
                    history_points=2,
                    reference_points=30,
                    iemoec=IEMOECConfig.for_variant(
                        "s3_elite_constrained"
                    ),
                )
                result = run_case(case, force=True)

                self.assertEqual(result["n_eval"], 182)
                self.assertEqual(result["metric_schema_version"], 6)
                self.assertIn("feasible_ratio", result)
                self.assertIn("min_cv", result)
                self.assertIn("mean_cv", result)
                self.assertIn("feasible_direction_coverage", result)
                self.assertIn("first_feasible_fe", result)
                initialization_hashes.append(result["initialization_hash"])
        self.assertEqual(len(set(initialization_hashes)), 1)

    def test_s3_variants_obey_budget_and_write_diagnostics(self):
        for variant in ("s3_memory", "s3_hybrid", "s3_elite", "s3"):
            with self.subTest(variant=variant):
                config = IEMOECConfig.for_variant(variant)
                case = ExperimentCase(
                    "IEMOEC",
                    "dtlz2",
                    3,
                    61,
                    273,
                    output_root=str(Path(self.output) / variant),
                    history_points=3,
                    reference_points=30,
                    iemoec=config,
                )
                result = run_case(case, force=True)
                with (case.output_dir / "iemoec_diagnostics.csv").open(
                    encoding="utf-8-sig",
                    newline="",
                ) as handle:
                    diagnostics = list(csv.DictReader(handle))
                with (case.output_dir / "final_population.csv").open(
                    encoding="utf-8-sig",
                    newline="",
                ) as handle:
                    final_rows = list(csv.DictReader(handle))

                self.assertEqual(result["n_eval"], 273)
                self.assertEqual(len(diagnostics), 2)
                self.assertEqual(result["global_selection_count"], 2)
                self.assertEqual(
                    result["isolated_offspring_total"]
                    + result["shared_offspring_total"]
                    + result["recombination_offspring_total"],
                    case.max_fes - result["reference_population_size"],
                )
                self.assertTrue(all(
                    int(row["isolated_offspring"])
                    + int(row["shared_offspring"])
                    + int(row["recombination_offspring"])
                    == int(row["outer_batch_fes"])
                    for row in diagnostics
                ))
                self.assertEqual(diagnostics[0]["island_state_reused"], "False")
                self.assertEqual(diagnostics[1]["island_state_reused"], "True")
                self.assertEqual(int(diagnostics[0]["direction_memory_capacity"]), 16)
                decision_columns = [
                    key for key in final_rows[0] if key.startswith("x")
                ]
                decisions = [
                    tuple(row[column] for column in decision_columns)
                    for row in final_rows
                ]
                self.assertEqual(len(decisions), len(set(decisions)))
                if variant == "s3_memory":
                    self.assertTrue(all(
                        int(row["shared_offspring"]) == 0 for row in diagnostics
                    ))
                if variant in ("s3_elite", "s3"):
                    self.assertTrue(all(
                        int(row["protected_elite_count"]) > 0
                        for row in diagnostics
                    ))

    def test_s3_obeys_partial_final_batch(self):
        config = IEMOECConfig.for_variant("s3_hybrid")
        case = ExperimentCase(
            "IEMOEC",
            "dtlz2",
            3,
            67,
            98,
            output_root=str(Path(self.output) / "partial_s3"),
            history_points=2,
            reference_points=30,
            iemoec=config,
        )
        result = run_case(case, force=True)
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            diagnostic = list(csv.DictReader(handle))[0]

        self.assertEqual(result["n_eval"], 98)
        self.assertEqual(int(diagnostic["outer_batch_fes"]), 7)
        self.assertEqual(
            int(diagnostic["isolated_offspring"])
            + int(diagnostic["shared_offspring"])
            + int(diagnostic["recombination_offspring"]),
            7,
        )

    def test_s3_is_reproducible_with_the_same_seed(self):
        config = IEMOECConfig.for_variant("s3")
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 71, 273, iemoec=config
        )
        problem = make_problem("dtlz2", 3)
        initial_X = shared_initial_decisions(problem, 91, case.seed)
        first = IEMOECRunner(problem, case, initial_X=initial_X)
        second = IEMOECRunner(problem, case, initial_X=initial_X)

        first_population, _ = first.run()
        second_population, _ = second.run()

        np.testing.assert_allclose(
            first_population.get("F"),
            second_population.get("F"),
        )
        first_budgets = [
            (
                row["source_budget_isolated"],
                row["source_budget_shared"],
                row["source_budget_recombination"],
            )
            for row in first.outer_records
        ]
        second_budgets = [
            (
                row["source_budget_isolated"],
                row["source_budget_shared"],
                row["source_budget_recombination"],
            )
            for row in second.outer_records
        ]
        self.assertEqual(first_budgets, second_budgets)

    def test_s3_m15_smoke_uses_two_m_directions(self):
        config = IEMOECConfig.for_variant("s3_memory")
        case = ExperimentCase(
            "IEMOEC",
            "wfg6",
            15,
            73,
            240,
            output_root=str(Path(self.output) / "s3_m15"),
            history_points=2,
            reference_points=30,
            high_dim_hv_samples=1000,
            iemoec=config,
        )
        result = run_case(case, force=True)
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            diagnostic = list(csv.DictReader(handle))[0]

        self.assertEqual(result["n_eval"], 240)
        self.assertEqual(int(diagnostic["island_count"]), 30)
        self.assertEqual(int(diagnostic["direction_memory_capacity"]), 4)

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
            self.assertEqual(result["metric_schema_version"], 5)
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

    def test_rvea_and_moead_pbi_obey_identical_fe_budget(self):
        initialization_hashes = []
        for algorithm in ("RVEA", "MOEADPBI"):
            case = ExperimentCase(
                algorithm,
                "dtlz2",
                3,
                41,
                182,
                output_root=str(Path(self.output) / algorithm),
                history_points=3,
                reference_points=30,
                iemoec=self.small_iemoec,
            )

            result = run_case(case, force=True)

            self.assertEqual(result["n_eval"], 182)
            self.assertEqual(result["reference_population_size"], 91)
            self.assertGreater(result["population_size"], 0)
            initialization_hashes.append(result["initialization_hash"])
        self.assertEqual(len(set(initialization_hashes)), 1)

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

    def test_timing_only_separates_algorithm_metrics_and_io(self):
        case = ExperimentCase(
            "NSGA2", "dtlz2", 3, 71, 182,
            output_root=str(Path(self.output) / "timing"),
            history_points=3,
            reference_points=30,
            timing_only=True,
            iemoec=self.small_iemoec,
        )

        result = run_case(case, force=True)

        self.assertGreater(result["algorithm_runtime_seconds"], 0.0)
        self.assertGreater(result["metric_runtime_seconds"], 0.0)
        self.assertGreaterEqual(result["io_runtime_seconds"], 0.0)
        self.assertGreater(result["total_runtime_seconds"], 0.0)
        with (case.output_dir / "history.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            history = list(csv.DictReader(handle))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["event"], "final")

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

    def test_s2_uses_candidate_execution_path(self):
        config = IEMOECConfig.for_variant("s2", island_population=4)
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 37, 182,
            output_root=str(Path(self.output) / "s2"),
            history_points=3,
            reference_points=30,
            iemoec=config,
        )

        result = run_case(case, force=True)

        self.assertEqual(result["algorithm_variant"], "s2")
        self.assertEqual(result["algorithm_schema_version"], 3)
        self.assertEqual(result["n_eval"], 182)
        self.assertEqual(result["global_selection_count"], 1)

    def test_s2_no_isolation_uses_shared_pool_and_obeys_budget(self):
        config = IEMOECConfig.for_variant(
            "s2_no_isolation",
            island_population=4,
        )
        case = ExperimentCase(
            "IEMOEC", "dtlz2", 3, 39, 250,
            output_root=str(Path(self.output) / "no_isolation"),
            history_points=3,
            reference_points=30,
            iemoec=config,
        )

        result = run_case(case, force=True)

        self.assertEqual(result["n_eval"], 250)
        self.assertEqual(result["algorithm_variant"], "s2_no_isolation")
        self.assertEqual(result["algorithm_label"], "IEMOEC-RD-NoIsolation")
        with (case.output_dir / "iemoec_diagnostics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            diagnostics = list(csv.DictReader(handle))
        self.assertTrue(
            all(row["local_evolution_mode"] == "shared" for row in diagnostics)
        )
        self.assertEqual(
            sum(int(row["outer_batch_fes"]) for row in diagnostics),
            case.max_fes - result["reference_population_size"],
        )

    def test_manifest_validation_rejects_missing_and_unexpected_tasks(self):
        cases = [self.case("NSGA2", seed=1), self.case("NSGA3", seed=1)]
        manifest = build_manifest(cases, metric_schema_version=5)
        first = cases[0].to_dict()
        first["metric_schema_version"] = 5

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_result_rows([first], manifest)
        validation = validate_result_rows(
            [first], manifest, allow_incomplete=True
        )
        self.assertEqual(validation["expected"], 2)
        self.assertEqual(validation["completed"], 1)

        unexpected = dict(first, seed=99)
        with self.assertRaisesRegex(RuntimeError, "outside the manifest"):
            validate_result_rows([unexpected], manifest, allow_incomplete=True)

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
