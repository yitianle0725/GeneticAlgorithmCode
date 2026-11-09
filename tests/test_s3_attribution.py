from __future__ import annotations

import itertools
import json
import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from iemoec_experiment.config import ExperimentCase, IEMOEOConfig
from iemoec_experiment.iemoeo import IEMOEORunner
from iemoec_experiment.normalization import ObjectiveNormalization
from iemoec_experiment.runner import run_case
from run_iemoeo_mechanisms import build_commands
from run import build_parser, resolve_cases, load_frozen_iemoeo
from summarize_iemoeo_mechanisms import factorial_effect_rows


class QuadraticProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=3, xl=np.zeros(4), xu=np.ones(4))

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack([np.sum((X - target) ** 2, axis=1) for target in (0., .5, 1.)])


class AttributionCoreTests(unittest.TestCase):
    def make_runner(self, budget=182, **changes):
        options = IEMOEOConfig(architecture="s3_attribution", origin_ratio=1., parent_check=False)
        case = ExperimentCase("IEMOEO", "dtlz2", 3, 7, budget, iemoeo=replace(options, **changes))
        return IEMOEORunner(QuadraticProblem(), case)

    def test_all_factor_cells_keep_budget_population_and_locality(self):
        for isolated, best, blocks, memory, elites in itertools.product([False, True], repeat=5):
            with self.subTest(I=isolated, P=best, C=blocks, M=memory, E=elites):
                runner = self.make_runner(isolation=isolated,
                    representative_selection="local_best" if best else "local_random",
                    combination_method="structured" if blocks else "sbx_only",
                    persistent_direction_memory=memory, direction_elite_protection=elites)
                population, _ = runner.run()
                self.assertEqual(runner.n_eval, 182)
                self.assertEqual(len(population), runner.pop_size)
                record = runner.outer_records[0]
                self.assertEqual(record["local_search_fes"] + record["probe_fes"] + record["recombination_offspring"], record["outer_batch_fes"])
                self.assertEqual(record["foreign_parent_count"] == 0, isolated)
                self.assertEqual(record["protected_elite_count"] > 0, elites)

    def test_memory_cannot_change_the_first_batch(self):
        on, off = self.make_runner(), self.make_runner(persistent_direction_memory=False)
        first, _ = on.run()
        second, _ = off.run()
        np.testing.assert_array_equal(first.get("X"), second.get("X"))

    def test_qualification_is_finite_complete_and_spends_actual_fes(self):
        runner = self.make_runner(qualification_mode="probe_checked")
        # 在零点对三个相同目标的坐标邻域检查可通过；中点存在下降方向。
        runner.problem._evaluate = lambda X, out, *a, **k: out.update(F=np.repeat(np.sum(X ** 2, axis=1)[:, None], 3, axis=1))
        representatives = runner._evaluate_unique(Population.new("X", np.array([[0.] * 4, [.5] * 4])))
        representatives.set("source_island", [0, 1])
        normalization = ObjectiveNormalization.from_objectives(np.array([[0.] * 3, [1.] * 3]))
        before = runner.n_eval
        qualified, probes, tested = runner._probe_representatives(representatives, [np.ones(3)] * 2,
                                                                 normalization, 12, 0)
        np.testing.assert_array_equal(qualified, [True, False])
        self.assertEqual(tested, 2)
        self.assertEqual(runner.n_eval - before, len(probes))
        self.assertEqual(len(probes), 12)
        qualified, probes, tested = runner._probe_representatives(representatives, [np.ones(3)] * 2,
                                                                 normalization, 0, 0)
        self.assertEqual(len(probes), 0)  # 完整缓存可用，但不重复计 FE。
        self.assertEqual(tested, 2)

    def test_no_unqualified_fallback_and_no_combination_budget_loss(self):
        runner = self.make_runner(budget=500, qualification_mode="probe_checked")
        def reject(representatives, *args):
            return np.zeros(len(representatives), dtype=bool), Population.empty(), 0
        with patch.object(runner, "_probe_representatives", side_effect=reject):
            runner.run()
        self.assertTrue(all(r["recombination_offspring"] == 0 for r in runner.outer_records))
        self.assertTrue(all(not r["parent_fallback"] for r in runner.outer_records))
        self.assertEqual(runner.n_eval, 500)

    def test_tail_budgets_and_probe_controls_are_repeatable(self):
        for budget in (91, 92, 95, 97, 365, 500):
            for mode in ("none", "probe_checked", "probe_unchecked"):
                runner = self.make_runner(budget, qualification_mode=mode)
                population, _ = runner.run()
                self.assertEqual(runner.n_eval, budget)
                self.assertEqual(len(population), runner.pop_size)
                self.assertEqual(runner.initial_evaluations + sum(
                    r["local_search_fes"] + r["probe_fes"] + r["recombination_offspring"] for r in runner.outer_records), budget)
                if mode == "probe_checked":
                    self.assertTrue(all(r["qualified_parent_count"] <= r["probe_complete_representatives"] for r in runner.outer_records))
        a, b = self.make_runner(500), self.make_runner(500)
        np.testing.assert_array_equal(a.run()[0].get("X"), b.run()[0].get("X"))

    def test_common_result_core_schema_metadata_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = self.make_runner().case.iemoeo
            case = ExperimentCase("IEMOEO", "dtlz2", 3, 7, 182, temporary,
                result_store=str(Path(temporary) / "store"), reference_points=30, iemoeo=options)
            result = run_case(case)
            self.assertEqual(result["algorithm_schema_version"], 4)
            self.assertEqual(result["island_initialization"], "direction_memory_or_current_population")
            self.assertEqual(result["extremum_certificate"], "not_tested")
            self.assertEqual(sum(result["evaluation_breakdown"].values()), 182)
            with patch("iemoec_experiment.iemoeo_attribution.AttributionRunner.run", side_effect=AssertionError("recomputed")):
                self.assertEqual(run_case(case)["status"], "skipped")


class AttributionPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "experiments/iemoeo_s3_attribution_v1.json").read_text(encoding="utf-8"))

    def test_factorial_has_all_32_unique_scientific_cells(self):
        commands = build_commands(self.plan, "smoke", "factorial", Path("study"), Path("store"), 4)
        lookup = {c["variant"]: c for c in commands}
        family = self.plan["effect_families"][1]
        self.assertEqual(len(family["cells"]), 32)
        self.assertEqual(set(family["cells"]), {"".join(bits) for bits in itertools.product("01", repeat=5)})
        configs = {name: lookup[name]["iemoeo_config"] for name in family["cells"].values()}
        self.assertEqual(len({json.dumps(c, ensure_ascii=False, sort_keys=True) for c in configs.values()}), 32)
        for factor, config_field in zip("IPCME", ["isolation", "representative_selection", "combination_method", "persistent_direction_memory", "direction_elite_protection"]):
            for bits, name in family["cells"].items():
                position = "IPCME".index(factor)
                flipped = list(bits); flipped[position] = "1" if bits[position] == "0" else "0"
                other = configs[family["cells"]["".join(flipped)]]
                changed = [key for key in configs[name] if configs[name][key] != other[key]]
                self.assertEqual(changed, [config_field])
        self.assertEqual(lookup["s3_elite_reference"]["case_template"]["algorithm_schema_version"], 7)
        elite = lookup["s3_elite_reference"]["iemoec_config"]
        hybrid = lookup["s3_hybrid_reference"]["iemoec_config"]
        self.assertEqual({k for k in elite if elite[k] != hybrid[k]}, {"variant", "protect_direction_elites"})

    def test_package_corners_and_independent_confirmation(self):
        names = set(self.plan["suites"]["screen"])
        for interaction in self.plan["interactions"]:
            self.assertTrue(set(interaction["cells"]) <= names)
        self.assertTrue(set(self.plan["effect_families"][0]["cells"].values()) <= names)
        self.assertEqual(self.plan["stages"]["confirm"]["seeds"], "61-90")
        self.assertEqual(self.plan["stages"]["pilot"]["objectives"], "3,5,8")

    def test_marginal_effect_uses_seed_as_unit_and_separates_factors(self):
        family = self.plan["effect_families"][0]
        arms = {}
        for bits, name in family["cells"].items():
            arms[name] = [{"problem": "dtlz2", "n_obj": 3, "seed": seed,
                           "igd_plus": seed * (.5 if bits[1] == "1" else 1.)} for seed in range(1, 6)]
        rows = factorial_effect_rows(arms, [family])
        lookup = {r["factor"]: r for r in rows}
        self.assertAlmostEqual(lookup["M"]["median_ratio"], .5)
        self.assertEqual(lookup["M"]["n_pairs"], 5)
        self.assertEqual(lookup["M"]["background_pairs"], 4)
        self.assertEqual(lookup["K"]["median_log_effect"], 0.)
        self.assertEqual(lookup["E"]["p_value"], 1.)

    def test_confirmation_candidate_and_matched_controls_are_frozen(self):
        plan = json.loads((ROOT / "experiments/iemoeo_s3_confirmation_v1.json").read_text(encoding="utf-8"))
        commands = build_commands(plan, "confirm", "confirm_core", Path("study"), Path("store"), 4)
        self.assertEqual(sum(c["tasks"] for c in commands), 4860)
        self.assertEqual(sum(c["total_fes"] for c in commands), 136_404_000)
        configs = {c["variant"]: c["iemoeo_config"] for c in commands if "iemoeo_config" in c}
        self.assertEqual(configs["f11011"]["combination_method"], "sbx_only")
        for contrast in plan["contrasts"]:
            if contrast["id"] == "candidate_vs_s3":
                continue
            a, b = configs[contrast["target"]], configs[contrast["control"]]
            self.assertEqual(sum(a[k] != b[k] for k in a), 1, contrast["id"])
        for command in commands:
            args = command["arguments"]
            self.assertEqual(args[args.index("--seeds") + 1], "61-90")
            self.assertEqual(args[args.index("--evals-per-pop") + 1], "200")
            self.assertEqual(args[args.index("--objectives") + 1], "3,5,8")

    def test_frozen_profile_reproduces_candidate_without_changing_s3(self):
        profile = ROOT / "experiments/iemoeo_s3_frozen_v1.json"
        args = build_parser().parse_args(["--preset", "custom", "--algorithms", "IEMOEC", "IEMOEO",
            "--iemoec-variant", "s3_elite", "--iemoeo-frozen", str(profile), "--problems", "dtlz2",
            "--objectives", "3", "--seeds", "61", "--evals-per-pop", "200"])
        s3, candidate = resolve_cases(args)
        self.assertEqual(s3.iemoec.origin_ratio, .2)
        self.assertEqual(candidate.iemoeo.origin_ratio, 1.)
        self.assertEqual(candidate.iemoeo.combination_method, "sbx_only")
        self.assertTrue(candidate.iemoeo.persistent_direction_memory)
        self.assertFalse(candidate.iemoeo.parent_check)
        self.assertEqual(candidate.iemoeo.qualification_mode, "none")
        plan = json.loads((ROOT / "experiments/iemoeo_s3_confirmation_v1.json").read_text(encoding="utf-8"))
        commands = build_commands(plan, "confirm", "confirm_core", Path("study"), Path("store"), 4)
        original = resolve_cases(build_parser().parse_args(commands[0]["arguments"]))[0]
        self.assertEqual(candidate.iemoeo, original.iemoeo)

    def test_frozen_profile_refuses_changed_parameters_and_implementation(self):
        original = json.loads((ROOT / "experiments/iemoeo_s3_frozen_v1.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            changed = copy.deepcopy(original)
            changed["iemoeo"]["origin_ratio"] = .3
            path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "冻结参数摘要不匹配"):
                load_frozen_iemoeo(path)
            changed = copy.deepcopy(original)
            changed["implementation"]["sources"]["iemoeo_attribution"] = "changed"
            path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "当前实现或依赖与冻结版本不符"):
                load_frozen_iemoeo(path)

    def test_frozen_profile_cannot_be_mixed_with_mechanism_overrides(self):
        args = build_parser().parse_args(["--preset", "smoke", "--algorithms", "IEMOEO",
            "--iemoeo-frozen", str(ROOT / "experiments/iemoeo_s3_frozen_v1.json"), "--no-isolation"])
        with self.assertRaisesRegex(ValueError, "冻结配置不能与"):
            resolve_cases(args)


if __name__ == "__main__":
    unittest.main()
