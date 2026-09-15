from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iemoec_experiment.config import ExperimentCase, IEMOECConfig
from iemoec_experiment.principle import Lineage, PrincipleRunner
from iemoec_experiment.runner import run_case
from run import build_parser, resolve_cases


class FlatProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=3, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.zeros((len(X), 3))


class BowlProblem(FlatProblem):
    def _evaluate(self, X, out, *args, **kwargs):
        value = np.sum((X - 0.5) ** 2, axis=1)
        out["F"] = np.column_stack([value] * 3)


class PrincipleTests(unittest.TestCase):
    def make_runner(self, problem=None, budget=200, **changes):
        config = IEMOECConfig.for_variant(
            "principle", origin_ratio=0.02, island_population=3,
            principle_min_generations=1, principle_stagnation_generations=1,
            **changes,
        )
        case = ExperimentCase("IEMOEC", "dtlz2", 3, 1, budget, "unused", iemoec=config)
        return PrincipleRunner(problem or FlatProblem(), case,
                               np.array([[0.2, 0.3], [0.8, 0.7]]))

    def test_closed_lineages_and_qualified_combination(self):
        runner = self.make_runner()
        pop, rounds = runner.run()
        self.assertEqual(runner.n_eval, 200)
        self.assertEqual(runner.initial_evaluations, 2)
        self.assertGreater(rounds, 0)
        self.assertGreater(len(pop), 0)
        combos = [r for r in runner.lineage_records if r["event"] == "extreme_combination"]
        self.assertTrue(combos)
        for r in combos:
            self.assertNotEqual(r["lineage_a"], r["lineage_b"])
            self.assertTrue(r["parent_a_qualified"] and r["parent_b_qualified"])
        for r in runner.lineage_records:
            if "foreign_parent_count" in r:
                self.assertEqual(r["foreign_parent_count"], 0)
        self.assertEqual(sum(r["island_fes"] + r["recombination_offspring"] for r in runner.outer_records) + 2, 200)

    def test_one_founder_expansion_and_isolation_guard(self):
        runner = self.make_runner()
        founders = runner._evaluate(runner.initial_X)
        # Disable mutation: each pool must contain ONLY its own ancestor.
        runner._mutate = lambda X: X.copy()
        lineages = runner._new_lineages(founders, 1)
        for i, lineage in enumerate(lineages):
            np.testing.assert_allclose(lineage.population.get("X"), np.tile(runner.initial_X[i], (3, 1)))
        lineages[0].population[0].set("lineage_id", lineages[1].identity)
        with self.assertRaises(RuntimeError):
            runner._local_generation(lineages[0], 1)

    def test_probe_rejects_non_extremum(self):
        runner = self.make_runner(BowlProblem())
        runner.ideal = np.zeros(3)
        runner.scale = np.ones(3)
        pop = runner._evaluate([[0.2, 0.2]], 0)
        lineage = Lineage(0, pop, np.ones(3) / 3)
        runner._qualify(lineage, 1)
        self.assertFalse(lineage.qualified)
        self.assertTrue(runner.lineage_records[-1]["probe_improved"])

    def test_probe_accepts_bowl_minimum(self):
        runner = self.make_runner(BowlProblem())
        runner.ideal = np.zeros(3)
        runner.scale = np.ones(3)
        lineage = Lineage(0, runner._evaluate([[0.5, 0.5]], 0), np.ones(3) / 3)
        runner._qualify(lineage, 1)
        self.assertTrue(lineage.qualified)

    def test_incomplete_probe_never_qualifies(self):
        runner = self.make_runner(budget=3)
        runner.ideal = np.zeros(3)
        runner.scale = np.ones(3)
        lineage = Lineage(0, runner._evaluate([[0.5, 0.5]], 0), np.ones(3) / 3)
        runner._qualify(lineage, 1)
        self.assertFalse(lineage.qualified)
        self.assertEqual(runner.n_eval, 3)

    def test_s4_uses_auditable_multiscale_probes(self):
        runner = self.make_runner(BowlProblem())
        runner.config = IEMOECConfig.for_variant(
            "s4", origin_ratio=0.02, island_population=3,
            principle_min_generations=1,
        )
        runner.ideal = np.zeros(3)
        runner.scale = np.ones(3)
        lineage = Lineage(0, runner._evaluate([[0.2, 0.2]], 0), np.ones(3) / 3)
        runner._qualify(lineage, 1)
        runner._qualify(lineage, 1)
        checks = [row for row in runner.lineage_records if row["event"] == "qualification"]
        self.assertEqual(checks[0]["probe_radius_ratio"], 0.01)
        self.assertEqual(checks[1]["probe_radius_ratio"], 0.005)

    def test_coordinate_inheritance_before_mutation(self):
        runner = self.make_runner()
        runner._mutate = lambda X: np.asarray(X).copy()
        reps = Population.new("X", np.array([[0.2, 0.3], [0.8, 0.7]]))
        children = runner._combine(reps, [1, 2])
        for x in children.get("X"):
            self.assertIn(tuple(x), [(0.2, 0.7), (0.8, 0.3)])

    def test_rank_only_children_select_founders(self):
        runner = self.make_runner()
        mixed = Population.new("X", np.array([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]),
                               "F", np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        chosen = runner._select_founders(mixed)
        np.testing.assert_array_equal(chosen.get("F"), [[1, 1, 1], [2, 2, 2]])

    def test_tail_budgets_and_reproducibility(self):
        for budget in [2, 3, 7, 31, 120, 200]:
            a = self.make_runner(budget=budget)
            b = self.make_runner(budget=budget)
            pa, _ = a.run()
            pb, _ = b.run()
            self.assertEqual(a.n_eval, budget)
            np.testing.assert_array_equal(pa.get("X"), pb.get("X"))

    def test_legacy_snapshots_unchanged_and_cli(self):
        old = ExperimentCase("IEMOEC", "dtlz2", 3, 1, 200, "unused",
                             iemoec=IEMOECConfig.for_variant("s2")).to_dict()
        self.assertFalse(any(k.startswith("principle_") for k in old["iemoec"]))
        args = build_parser().parse_args(["--preset", "smoke", "--algorithms", "IEMOEC",
                                         "--iemoec-variant", "principle"])
        case = resolve_cases(args)[0]
        case.validate()
        self.assertEqual(case.iemoec.outer_survival, "rank")
        self.assertEqual(case.algorithm_schema_version, 9)
        args.no_recombination = True
        with self.assertRaises(ValueError):
            resolve_cases(args)
        args.no_recombination = False
        args.iemoec_survival = "nsga3"
        with self.assertRaises(ValueError):
            resolve_cases(args)

    def test_run_case_output_and_actual_initial_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            case = ExperimentCase("IEMOEC", "dtlz2", 3, 1, 120, temp,
                                  timing_only=True,
                                  iemoec=IEMOECConfig.for_variant("principle", island_population=3))
            result = run_case(case)
            self.assertEqual(result["n_eval"], 120)
            self.assertEqual(result["initial_evaluations"], 19)
            self.assertTrue((case.output_dir / "lineage_audit.csv").exists())
            config = json.loads((case.output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertNotIn("outer_survival", config["iemoec"])

    def test_s4_combines_as_soon_as_two_extrema_qualify(self):
        runner = self.make_runner()
        runner.config = IEMOECConfig.for_variant(
            "s4", origin_ratio=0.02, island_population=3,
            principle_min_generations=1,
        )
        population, _ = runner.run()
        self.assertEqual(runner.n_eval, 200)
        self.assertGreater(runner.global_selection_count, 0)
        self.assertTrue(any(
            record["event"] == "extreme_combination"
            for record in runner.lineage_records
        ))

    def test_s4_cli_uses_new_schema(self):
        args = build_parser().parse_args([
            "--preset", "smoke", "--algorithms", "IEMOEC",
            "--iemoec-variant", "s4",
        ])
        case = resolve_cases(args)[0]
        case.validate()
        self.assertEqual(case.algorithm_schema_version, 10)
        self.assertEqual(case.algorithm_label, "IEMOEC-S4")

    def test_s4_global_batches_are_complete(self):
        runner = self.make_runner(budget=200)
        runner.config = IEMOECConfig.for_variant(
            "s4", origin_ratio=0.02, island_population=3,
            principle_min_generations=1,
        )
        runner.run()
        combined = [
            row for row in runner.outer_records
            if row["recombination_offspring"]
        ]
        self.assertTrue(combined)
        self.assertTrue(all(
            row["recombination_offspring"] == runner.pop_size
            for row in combined
        ))


if __name__ == "__main__":
    unittest.main()
