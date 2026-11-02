from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from analyze_iemoeo_pilot import descriptive_comparisons
from run_iemoeo_mechanisms import DEFAULT_PLAN, build_commands, write_schedule
from summarize_iemoeo_mechanisms import (
    apply_global_holm, contrast_overview, evidence_scope, interaction_overview,
    interaction_rows, paired_log_ratio,
)


class MechanismPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "experiments" / "iemoeo_mechanisms_v2.json").read_text(encoding="utf-8"))

    def test_pilot_is_720_ie_only_tasks_and_single_factor_contrasts(self):
        commands = build_commands(self.plan, "pilot", "core", Path("study"), Path("store"), 4)
        self.assertEqual(sum(command["tasks"] for command in commands), 720)
        self.assertEqual(sum(command["total_fes"] for command in commands), 18_960_000)
        configurations = {command["variant"]: command["iemoeo_config"] for command in commands}
        for command in commands:
            arguments = command["arguments"]
            self.assertEqual(arguments[arguments.index("--algorithms") + 1], "IEMOEO")
            self.assertNotIn("--force", arguments)
        for contrast in self.plan["contrasts"]:
            if contrast["target"] not in configurations or contrast["control"] not in configurations:
                continue
            target, control = configurations[contrast["target"]], configurations[contrast["control"]]
            changed = [field for field in target if target[field] != control[field]]
            self.assertEqual(len(changed), 1, (contrast["id"], changed))

    def test_confirmation_uses_new_seeds_and_400n(self):
        commands = build_commands(self.plan, "confirm", "core", Path("study"), Path("store"), 4)
        self.assertEqual(sum(command["tasks"] for command in commands), 4320)
        self.assertEqual(sum(command["total_fes"] for command in commands), 227_520_000)
        for command in commands:
            args = command["arguments"]
            self.assertEqual(args[args.index("--seeds") + 1], "31-60")
            self.assertEqual(args[args.index("--evals-per-pop") + 1], "400")

    def test_known_blocks_control_preserves_block_count(self):
        commands = build_commands(self.plan, "smoke", "sensitivity", Path("study"), Path("store"), 4)
        configs = {command["variant"]: command["iemoeo_config"] for command in commands}
        uniform, known = configs["uniform_2"], configs["known_2"]
        self.assertEqual(uniform["block_size"], known["block_size"])
        self.assertEqual([key for key in uniform if uniform[key] != known[key]], ["block_mode"])

    def test_timing_cannot_use_multiple_workers(self):
        with self.assertRaisesRegex(ValueError, "workers 1"):
            build_commands(self.plan, "timing", "core", Path("study"), Path("store"), 4)
        commands = build_commands(self.plan, "timing", "core", Path("study"), Path("store"), 1)
        self.assertTrue(all(not command["iemoeo_config"]["selection_audit"] for command in commands))

    def test_registration_rejects_plan_changes_but_allows_worker_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schedule.json"
            commands = build_commands(self.plan, "smoke", "core", Path("study"), Path("store"), 1)
            write_schedule(path, self.plan, "smoke", "core", commands)
            commands = build_commands(self.plan, "smoke", "core", Path("study"), Path("store"), 2)
            write_schedule(path, self.plan, "smoke", "core", commands)
            changed = copy.deepcopy(self.plan)
            changed["stages"]["smoke"]["seeds"] = "2"
            with self.assertRaisesRegex(ValueError, "冻结记录"):
                write_schedule(path, changed, "smoke", "core", commands)
            changed_commands = copy.deepcopy(commands)
            changed_commands[0]["case_template"]["reference_points"] = 2000
            with self.assertRaisesRegex(ValueError, "冻结记录"):
                write_schedule(path, self.plan, "smoke", "core", changed_commands)

    def test_unimplemented_controls_are_explicitly_pending(self):
        old_plan = json.loads((ROOT / "experiments" / "iemoeo_mechanisms_v1.json").read_text(encoding="utf-8"))
        pending = {item["id"] for item in old_plan["pending_controls"]}
        self.assertIn("true_crowding_control", pending)
        self.assertIn("local_representative_selection", pending)
        self.assertIn("no_combination_equal_budget", pending)
        self.assertTrue(pending.isdisjoint(old_plan["variants"]))

    def test_v2_attribution_has_all_missing_controls(self):
        commands = build_commands(self.plan, "pilot", "attribution", Path("study"), Path("store"), 4)
        self.assertEqual(sum(command["tasks"] for command in commands), 1350)
        configs = {command["variant"]: command["iemoeo_config"] for command in commands}
        self.assertEqual(configs["local_random"]["representative_selection"], "local_random")
        self.assertEqual(configs["global_best"]["representative_selection"], "global_best")
        self.assertEqual(configs["crowding"]["outer_survival"], "rank_crowding")
        self.assertEqual(configs["no_combination"]["combination_method"], "none")
        self.assertTrue(all(config["selection_audit"] for config in configs.values()))
        for item in self.plan["interactions"]:
            self.assertTrue(all(name in configs for name in item["cells"]))

    def test_combination_suite_includes_complete_factorial_cells(self):
        names = set(self.plan["suites"]["combination"])
        for item in self.plan["interactions"]:
            if item["id"] in ("isolation_x_blocks", "isolation_x_combination_presence"):
                self.assertTrue(set(item["cells"]) <= names)


class MechanismV3PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))

    def test_pilot_counts_and_every_contrast_changes_one_factor(self):
        self.assertEqual(self.plan["name"], "iemoeo_mechanisms_v3")
        commands = build_commands(self.plan, "pilot", "attribution", Path("study"), Path("store"), 4)
        self.assertEqual(sum(c["tasks"] for c in commands), 3060)
        self.assertEqual(sum(c["total_fes"] for c in commands), 80_580_000)
        configs = {c["variant"]: c["iemoeo_config"] for c in commands}
        for contrast in self.plan["contrasts"]:
            first, second = configs[contrast["target"]], configs[contrast["control"]]
            changed = [key for key in first if first[key] != second[key]]
            self.assertEqual(len(changed), 1, (contrast["id"], changed))
        self.assertTrue(all(c["case_template"]["algorithm_schema_version"] == 3 for c in commands))

    def test_core_confirmation_is_complete_and_independent_of_development(self):
        commands = build_commands(self.plan, "confirm", "core", Path("study"), Path("store"), 4)
        self.assertEqual(sum(c["tasks"] for c in commands), 4320)
        self.assertEqual(sum(c["total_fes"] for c in commands), 227_520_000)
        self.assertEqual(self.plan["stages"]["pilot"]["seeds"], "6-15")
        self.assertEqual(self.plan["stages"]["confirm"]["seeds"], "31-60")
        self.assertTrue(set(self.plan["interactions"][0]["cells"]) <= set(self.plan["suites"]["core"]))

    def test_check_controls_and_block_sizes_are_registered(self):
        commands = build_commands(self.plan, "smoke", "attribution", Path("study"), Path("store"), 4)
        configs = {c["variant"]: c["iemoeo_config"] for c in commands}
        self.assertEqual(sum(c["tasks"] for c in commands), 34)
        for suffix, radius in (("02", .2), ("04", .4)):
            self.assertEqual(configs[f"checked_{suffix}"]["parent_check_mode"], "local_scalar")
            self.assertEqual(configs[f"checked_{suffix}"]["principle_probe_radius"], radius)
            self.assertFalse(configs[f"unchecked_{suffix}"]["parent_check"])
        for name in ("known_2", "shuffled_known_2", "known_sbx", "known_coord"):
            self.assertEqual(configs[name]["block_size"], 2)


class MechanismStatisticsTests(unittest.TestCase):
    def test_smoke_and_timing_cannot_be_labelled_confirmation(self):
        self.assertEqual(evidence_scope("smoke"), "implementation_only")
        self.assertEqual(evidence_scope("pilot"), "development")
        self.assertEqual(evidence_scope("timing"), "timing_diagnostic")
        self.assertEqual(evidence_scope("confirm"), "confirmation")

    def make_arms(self, factors):
        return {name: [{"problem": "dtlz2", "n_obj": 3, "seed": seed,
                        "igd_plus": factor * seed} for seed in range(1, 6)]
                for name, factor in factors.items()}

    def test_interaction_is_zero_for_equal_isolation_effects(self):
        arms = self.make_arms({"full": 0.5, "shared": 1.0, "rank": 0.5, "shared_rank": 1.0})
        reports = interaction_rows(arms, [{"id": "test", "cells": list(arms)}])
        self.assertAlmostEqual(reports[0]["median_log_interaction"], 0.0)
        self.assertEqual(reports[0]["p_value"], 1.0)

    def test_interaction_direction_and_missing_seed_guard(self):
        arms = self.make_arms({"full": 0.25, "shared": 1.0, "rank": 0.5, "shared_rank": 1.0})
        interactions = [{"id": "test", "cells": list(arms)}]
        reports = interaction_rows(arms, interactions)
        self.assertAlmostEqual(reports[0]["median_log_interaction"], np.log(0.5))
        arms["rank"].pop()
        with self.assertRaisesRegex(ValueError, "不完整配对"):
            interaction_rows(arms, interactions)

    def test_log_ratio_is_finite_for_zero_igd(self):
        self.assertTrue(np.all(np.isfinite(paired_log_ratio(np.array([0., 1.]), np.array([1., 0.])))))

    def test_holm_is_applied_across_all_registered_tests(self):
        rows = [{"p_value": 0.01}, {"p_value": 0.02}, {"p_value": 0.03}]
        apply_global_holm(rows)
        np.testing.assert_allclose([row["p_holm"] for row in rows], [0.03, 0.04, 0.04])

    def test_median_direction_is_not_labelled_as_significance(self):
        groups = {("dtlz2", 3): {"IEMOEO": [{"igd_plus": 1, "hv": 2}],
                                  "NSGA3": [{"igd_plus": 2, "hv": 1}]}}
        rows = descriptive_comparisons(groups, "IEMOEO")
        self.assertTrue(all(row["better"] == 1 for row in rows))
        self.assertTrue(all("p_value" not in row for row in rows))

    def test_overview_separates_median_direction_from_significance(self):
        rows = [dict(contrast="test", metric="igd_plus", claim="test", target="full",
                     competitor="control", target_median=a, competitor_median=b,
                     target_result="=", p_holm=1., evidence_scope="development")
                for a, b in [(1., 2.), (2., 1.), (1., 1.)]]
        report = contrast_overview(rows)[0]
        self.assertEqual([report[key] for key in ("median_better", "median_equal", "median_worse")], [1, 1, 1])
        self.assertEqual(report["significant_better"], 0)
        self.assertEqual(report["significant_worse"], 0)
        self.assertEqual(report["median_relative_improvement_percent"], 0.)
        for row in rows:
            row["metric"] = "hv"
            row["competitor_median"] = 0.
        report = contrast_overview(rows)[0]
        self.assertEqual(report["median_better"], 3)
        self.assertIsNone(report["median_relative_improvement_percent"])
        self.assertEqual(report["relative_change_scenes"], 0)

    def test_interaction_overview_does_not_label_sign_as_significance(self):
        rows = [dict(interaction="test", median_log_interaction=-.2, p_holm=1.)]
        report = interaction_overview(rows)[0]
        self.assertEqual(report["negative"], 1)
        self.assertEqual(report["significant_negative"], 0)


if __name__ == "__main__":
    unittest.main()
