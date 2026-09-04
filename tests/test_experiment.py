from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iemoec_experiment.config import ExperimentCase, IEMOECConfig
from iemoec_experiment.factory import reference_directions
from iemoec_experiment.metrics import MetricSuite
from iemoec_experiment.problems import make_problem
from iemoec_experiment.runner import run_case


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

    def test_iemoec_defaults_to_one_inner_generation(self):
        config = IEMOECConfig()
        self.assertEqual(config.inner_generations_early, 1)
        self.assertEqual(config.inner_generations_late, 1)
        self.assertFalse(config.retain_island_state)

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

    def test_changed_configuration_requires_force(self):
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
            island_population=4,
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


if __name__ == "__main__":
    unittest.main()
