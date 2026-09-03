from __future__ import annotations

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
            result = run_case(self.case(algorithm), force=True)
            self.assertEqual(result["n_eval"], 182)
            self.assertEqual(result["reference_population_size"], 91)
            self.assertEqual(result["population_size"], 91)
            self.assertTrue(np.isfinite(result["igd_plus"]))

    def test_seed_is_reproducible_and_completed_case_is_skipped(self):
        case = self.case("NSGA2")
        first = run_case(case, force=True)
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


if __name__ == "__main__":
    unittest.main()
