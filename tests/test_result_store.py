from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from iemoec_experiment.config import ExperimentCase, IEMOECConfig, IEMOEOConfig
from iemoec_experiment.manifest import build_manifest, result_files, task_key, write_manifest
from iemoec_experiment.result_store import (
    complete_result, implementation_signature, legacy_result_candidates,
    stored_result_complete,
)
from iemoec_experiment.runner import run_case
from plot_results import validate_dataset
from summarize import load_rows


class ResultStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = ExperimentCase(
            "NSGA3", "dtlz2", 3, 7, 182, str(self.root / "first"),
            history_points=3, reference_points=30, high_dim_hv_samples=1000,
            result_store=str(self.root / "algorithms"),
        )

    def write_manifest(self, cases):
        manifest = build_manifest(cases, 5)
        write_manifest(Path(cases[0].output_root) / "experiment_manifest.json", manifest)
        return manifest

    def test_cross_batch_reuse_never_calls_optimizer(self):
        run_case(self.case)
        later = replace(self.case, output_root=str(self.root / "later"))
        self.assertEqual(self.case.output_dir, later.output_dir)
        self.assertEqual(self.case.output_dir.relative_to(self.root / "algorithms").parts[:4],
                         ("NSGA3", "DTLZ2", "M3", "seed_007"))
        with patch("iemoec_experiment.runner.minimize", side_effect=AssertionError("重跑了基线")):
            self.assertEqual(run_case(later)["status"], "skipped")

    def test_baselines_ignore_custom_algorithm_parameters(self):
        run_case(self.case)
        later = replace(self.case, iemoec=IEMOECConfig.for_variant("s3_elite"))
        self.assertEqual(self.case.output_dir, later.output_dir)
        self.assertEqual(run_case(later)["status"], "skipped")

    def test_different_budgets_metrics_and_seeds_have_separate_results(self):
        changes = [
            {"max_fes": 273}, {"seed": 8}, {"reference_points": 40},
            {"timing_only": True}, {"history_points": 4}, {"n_obj": 4},
        ]
        for values in changes:
            with self.subTest(values=values):
                self.assertNotEqual(self.case.output_dir, replace(self.case, **values).output_dir)
        run_case(self.case)
        changed = replace(self.case, max_fes=273)
        self.assertEqual(run_case(changed)["n_eval"], 273)
        self.assertTrue(stored_result_complete(self.case))

    def test_iemoeo_changes_do_not_invalidate_baselines(self):
        custom = replace(self.case, algorithm="IEMOEO")
        baseline_directory = self.case.output_dir
        custom_directory = custom.output_dir
        baseline_signature = implementation_signature("NSGA3")
        custom_signature = {**implementation_signature("IEMOEO"), "test_revision": "new"}
        with patch("iemoec_experiment.result_store.implementation_signature",
                   side_effect=lambda algorithm: custom_signature if algorithm == "IEMOEO" else baseline_signature):
            self.assertEqual(self.case.output_dir, baseline_directory)
            self.assertNotEqual(custom.output_dir, custom_directory)
        self.assertNotEqual(custom.output_dir,
                            replace(custom, iemoeo=IEMOEOConfig(block_size=2)).output_dir)

    def test_legacy_import_preserves_original_and_avoids_optimizer(self):
        legacy = replace(self.case, result_store=None, output_root=str(self.root / "old"))
        self.write_manifest([legacy])
        run_case(legacy)
        original_config = (legacy.output_dir / "config.json").read_bytes()
        candidates = legacy_result_candidates(self.root, [self.case])
        with patch("iemoec_experiment.runner.minimize", side_effect=AssertionError("重跑了旧结果")):
            result = run_case(self.case, legacy_candidates=candidates[task_key(self.case.to_dict())])
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reused_from"], str(legacy.output_dir))
        self.assertEqual((legacy.output_dir / "config.json").read_bytes(), original_config)
        self.assertEqual((legacy.output_dir / "final_population.csv").read_bytes(),
                         (self.case.output_dir / "final_population.csv").read_bytes())
        self.assertTrue(stored_result_complete(self.case))
        provenance = json.loads((self.case.output_dir / "storage_identity.json").read_text(encoding="utf-8"))
        self.assertFalse(provenance["legacy_implementation_verified"])

    def test_incompatible_or_incomplete_legacy_results_are_not_reused(self):
        legacy = replace(self.case, result_store=None, output_root=str(self.root / "old"))
        self.write_manifest([legacy])
        run_case(legacy)
        self.assertFalse(complete_result(legacy.output_dir, replace(self.case, max_fes=273)))
        metrics_path = legacy.output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["n_eval"] = 1
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
        self.assertFalse(complete_result(legacy.output_dir, self.case))
        result = run_case(self.case, legacy_candidates=(str(legacy.output_dir),))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(legacy_result_candidates(self.root, [replace(self.case, algorithm="IEMOEO")]))

    def test_missing_population_and_completion_marker_trigger_recovery(self):
        run_case(self.case)
        (self.case.output_dir / "final_population.csv").unlink()
        self.assertFalse(stored_result_complete(self.case))
        self.assertEqual(run_case(self.case)["status"], "completed")
        (self.case.output_dir / "storage_identity.json").unlink()
        self.assertEqual(run_case(self.case)["status"], "completed")

    def test_running_task_lock_prevents_duplicate_evaluation(self):
        directory = self.case.output_dir
        lock = directory.with_name(directory.name + ".lock")
        lock.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "锁目录"):
            run_case(self.case)

    def test_manifests_summaries_and_plots_resolve_shared_results(self):
        run_case(self.case)
        self.write_manifest([self.case])
        later_root = self.root / "later"
        later = replace(self.case, output_root=str(later_root))
        extra = replace(later, seed=8)
        self.write_manifest([later, extra])
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            load_rows(later_root)
        self.assertEqual(run_case(later)["status"], "skipped")
        run_case(extra)
        rows, validation = load_rows(later_root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(validation["completed"], 2)
        self.assertFalse(list(later_root.rglob("metrics.json")))
        self.assertEqual(len(result_files(later_root, "history.csv")), 2)
        validate_dataset(later_root, False)
        self.assertEqual(len(load_rows(Path(self.case.output_root))[0]), 1)

    def test_manifest_rejects_wrong_identity_and_reads_old_layout(self):
        run_case(self.case)
        manifest = self.write_manifest([self.case])
        manifest["tasks"][0]["result_identity"] = "wrong"
        manifest_path = Path(self.case.output_root) / "experiment_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "标识"):
            load_rows(Path(self.case.output_root))
        legacy = replace(self.case, result_store=None, output_root=str(self.root / "old"))
        run_case(legacy)
        self.write_manifest([legacy])
        self.assertEqual(len(load_rows(Path(legacy.output_root))[0]), 1)

    def test_cli_progress_and_reuse_across_run_names(self):
        command = [
            sys.executable, str(ROOT / "scripts/run.py"), "--preset", "smoke",
            "--algorithms", "NSGA3", "--evals-per-pop", "2", "--workers", "1",
            "--reference-points", "30", "--high-dim-hv-samples", "1000",
            "--output-root", str(self.root),
        ]
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        first = subprocess.run(command + ["--run-name", "cli_first"], capture_output=True,
                               encoding="utf-8", env=environment)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("[1/1][完成]", first.stdout)
        second = subprocess.run(command + ["--run-name", "cli_second"], capture_output=True,
                                encoding="utf-8", env=environment)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("[1/1][跳过]", second.stdout)


if __name__ == "__main__":
    unittest.main()
