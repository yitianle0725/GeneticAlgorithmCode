#!/usr/bin/env python
"""Recalculate HV from saved populations for several normalized reference points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iemoec_experiment.manifest import load_manifest, task_key  # noqa: E402
from iemoec_experiment.metrics import MetricSuite  # noqa: E402
from iemoec_experiment.problems import make_problem  # noqa: E402


def read_population(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        objective_columns = sorted(
            (name for name in (reader.fieldnames or []) if name.startswith("f")),
            key=lambda name: int(name[1:]),
        )
        rows = list(reader)
        objectives = np.asarray(
            [[float(row[name]) for name in objective_columns] for row in rows],
            dtype=float,
        )
        violations = None
        if reader.fieldnames and "cv" in reader.fieldnames:
            violations = np.asarray(
                [[float(row["cv"])] for row in rows],
                dtype=float,
            )
        return objectives, violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--reference-points", nargs="+", type=float, default=[1.1, 1.5, 2.0]
    )
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.results)
    rows = []
    missing = []
    for task in manifest["tasks"]:
        problem_name, n_obj, algorithm, seed = task_key(task)
        directory = (
            args.results / problem_name.upper() / f"M{n_obj}"
            / algorithm / f"seed_{seed:03d}"
        )
        population_path = directory / "final_population.csv"
        metrics_path = directory / "metrics.json"
        config_path = directory / "config.json"
        required_paths = (population_path, metrics_path, config_path)
        if not all(path.exists() for path in required_paths):
            missing.append(task_key(task))
            continue
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        F, CV = read_population(population_path)
        problem = make_problem(problem_name, n_obj, config.get("n_var"))
        for reference_point in args.reference_points:
            suite = MetricSuite(
                problem,
                n_reference_points=config["reference_points"],
                hv_samples=args.samples,
                hv_reference_point=reference_point,
            )
            values = suite.calculate_hv(F, CV=CV)
            rows.append({
                "problem": problem_name,
                "n_obj": n_obj,
                "algorithm": algorithm,
                "seed": seed,
                "reference_point": reference_point,
                "hv_method": suite.hv_method,
                "hv_samples": args.samples if n_obj > 5 else 0,
                "hv_eligible_solution_count": values["hv_eligible_solution_count"],
                "hv": values["hv"],
            })

    output = args.output or args.results / "hv_audit.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"audited {len(rows)} task/reference-point pairs; missing tasks={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
