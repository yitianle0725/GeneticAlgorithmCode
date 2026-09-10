from __future__ import annotations

import json
import os
from pathlib import Path

from .config import ExperimentCase


MANIFEST_SCHEMA_VERSION = 1
TASK_FIELDS = (
    "problem",
    "n_obj",
    "algorithm",
    "seed",
    "max_fes",
    "algorithm_schema_version",
    "algorithm_variant",
)


def task_key(data: dict) -> tuple[str, int, str, int]:
    return (
        str(data["problem"]).lower(),
        int(data["n_obj"]),
        str(data["algorithm"]).upper(),
        int(data["seed"]),
    )


def _task_from_case(case: ExperimentCase) -> dict:
    data = case.to_dict()
    return {field: data[field] for field in TASK_FIELDS}


def build_manifest(
    cases: list[ExperimentCase],
    metric_schema_version: int,
) -> dict:
    if not cases:
        raise ValueError("cannot build a manifest without experiment cases")
    roots = {Path(case.output_root).resolve() for case in cases}
    if len(roots) != 1:
        raise ValueError("all experiment cases must use the same output root")
    tasks = [_task_from_case(case) for case in cases]
    keys = [task_key(task) for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("experiment cases contain duplicate task keys")
    root = Path(cases[0].output_root)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_name": root.name,
        "metric_schema_version": int(metric_schema_version),
        "expected_task_count": len(tasks),
        "problems": sorted({task["problem"] for task in tasks}),
        "objectives": sorted({task["n_obj"] for task in tasks}),
        "seeds": sorted({task["seed"] for task in tasks}),
        "algorithms": sorted({task["algorithm"] for task in tasks}),
        "tasks": sorted(tasks, key=task_key),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise ValueError(
                f"{path} does not match the requested experiment configuration"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_manifest(root: Path) -> dict:
    path = root / "experiment_manifest.json"
    if not path.exists():
        raise RuntimeError(f"missing experiment manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported manifest schema in {path}")
    if manifest.get("run_name") != root.name:
        raise RuntimeError(
            f"manifest run_name={manifest.get('run_name')!r} does not match {root.name!r}"
        )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError(f"manifest tasks must be a list: {path}")
    keys = [task_key(task) for task in tasks]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"manifest contains duplicate task keys: {path}")
    if manifest.get("expected_task_count") != len(tasks):
        raise RuntimeError(f"manifest task count is inconsistent: {path}")
    dimensions = {
        "problems": sorted({str(task["problem"]).lower() for task in tasks}),
        "objectives": sorted({int(task["n_obj"]) for task in tasks}),
        "seeds": sorted({int(task["seed"]) for task in tasks}),
        "algorithms": sorted({str(task["algorithm"]).upper() for task in tasks}),
    }
    for field, expected in dimensions.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"manifest {field} do not match its tasks: {path}")
    if not isinstance(manifest.get("metric_schema_version"), int):
        raise RuntimeError(f"manifest metric schema is invalid: {path}")
    return manifest


def validate_result_rows(
    rows: list[dict],
    manifest: dict,
    allow_incomplete: bool = False,
) -> dict:
    expected = {task_key(task): task for task in manifest["tasks"]}
    actual: dict[tuple[str, int, str, int], dict] = {}
    for row in rows:
        key = task_key(row)
        if key in actual:
            raise RuntimeError(f"duplicate metrics result for {key}")
        actual[key] = row
    unexpected = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    if unexpected:
        raise RuntimeError(f"results contain tasks outside the manifest: {unexpected[:5]}")
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"results are incomplete: {len(missing)} missing tasks; "
            "use --allow-incomplete only for exploratory summaries"
        )
    metric_schema = manifest["metric_schema_version"]
    for key, row in actual.items():
        task = expected[key]
        if row.get("metric_schema_version") != metric_schema:
            raise RuntimeError(f"metric schema mismatch for {key}")
        for field in TASK_FIELDS[4:]:
            if row.get(field) != task[field]:
                raise RuntimeError(f"{field} mismatch for {key}")
    return {
        "expected": len(expected),
        "completed": len(actual),
        "missing": missing,
        "unexpected": unexpected,
    }
