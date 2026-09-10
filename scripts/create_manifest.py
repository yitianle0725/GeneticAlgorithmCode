#!/usr/bin/env python
"""Create a strict manifest for an existing experiment directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iemoec_experiment.config import ExperimentCase, IEMOECConfig  # noqa: E402
from iemoec_experiment.manifest import build_manifest, task_key, write_manifest  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    values = set()
    for token in text.split(","):
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return sorted(values)


def load_case(path: Path) -> ExperimentCase:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    for field in ("algorithm_variant", "algorithm_schema_version", "algorithm_label"):
        data.pop(field, None)
    data["iemoec"] = IEMOECConfig(**data["iemoec"])
    return ExperimentCase(**data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--problems", nargs="+", required=True)
    parser.add_argument("--objectives", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--metric-schema-version", type=int, required=True)
    args = parser.parse_args()

    cases = [load_case(path) for path in args.results.rglob("config.json")]
    by_key = {task_key(case.to_dict()): case for case in cases}
    if len(by_key) != len(cases):
        raise SystemExit("configuration directory contains duplicate task keys")
    expected_keys = {
        (problem.lower(), n_obj, algorithm.upper(), seed)
        for problem in args.problems
        for n_obj in args.objectives
        for algorithm in args.algorithms
        for seed in args.seeds
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        unexpected = sorted(set(by_key) - expected_keys)
        raise SystemExit(
            f"configuration mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    manifest = build_manifest(
        [by_key[key] for key in sorted(expected_keys)],
        args.metric_schema_version,
    )
    manifest["run_name"] = args.results.name
    write_manifest(args.results / "experiment_manifest.json", manifest)
    print(f"wrote manifest for {len(expected_keys)} tasks: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
