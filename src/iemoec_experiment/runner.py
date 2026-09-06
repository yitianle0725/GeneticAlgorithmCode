from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

from .config import ExperimentCase
from .factory import make_baseline, reference_directions
from .iemoec import IEMOECRunner
from .initialization import initialization_hash, shared_initial_decisions
from .metrics import METRIC_SCHEMA_VERSION, MetricSuite
from .problems import make_problem


def _json_dump(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, path)


class HistoryRecorder(Callback):
    def __init__(self, suite: MetricSuite, max_fes: int, n_points: int, include_hv: bool):
        super().__init__()
        self.suite = suite
        self.thresholds = np.unique(
            np.linspace(max_fes / n_points, max_fes, n_points, dtype=int)
        )
        self.include_hv = include_hv
        self.rows: list[dict] = []
        self.started = time.perf_counter()
        self._next = 0

    def record(self, n_eval: int, population) -> None:
        if population is None or len(population) == 0:
            return
        while self._next < len(self.thresholds) and n_eval >= self.thresholds[self._next]:
            checkpoint_fe = int(self.thresholds[self._next])
            values = self.suite.calculate(population.get("F"), include_hv=self.include_hv)
            self.rows.append({
                "fe": checkpoint_fe,
                "observed_fe": int(n_eval),
                "runtime_seconds": float(time.perf_counter() - self.started),
                "event": "checkpoint",
                "population_size": int(len(population)),
                **values,
            })
            self._next += 1

    def record_event(self, n_eval: int, population, event: str) -> dict:
        """记录检查点之外的重要算法状态，例如 IEMOEC 外层筛选。"""
        values = self.suite.calculate(population.get("F"), include_hv=self.include_hv)
        row = {
            "fe": int(n_eval),
            "observed_fe": int(n_eval),
            "runtime_seconds": float(time.perf_counter() - self.started),
            "event": event,
            "population_size": int(len(population)),
            **values,
        }
        self.rows.append(row)
        return values

    def finalize(self, n_eval: int, population, final_values: dict) -> None:
        """保证历史末行与最终 metrics 使用同一批目标值和指标。"""
        final_row = {
            "fe": int(n_eval),
            "observed_fe": int(n_eval),
            "runtime_seconds": float(time.perf_counter() - self.started),
            "event": "final",
            "population_size": int(len(population)),
            **final_values,
        }
        while self.rows and int(self.rows[-1]["fe"]) == int(n_eval):
            self.rows.pop()
        self.rows.append(final_row)

    def notify(self, algorithm):
        self.record(int(algorithm.evaluator.n_eval), algorithm.pop)


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        preferred = [
            "fe", "observed_fe", "runtime_seconds", "event", "population_size",
            "igd_plus", "gd_plus", "hv", "spacing", "direction_occupancy",
            "onvg", "nd_ratio",
        ]
        available = {key for row in rows for key in row}
        fieldnames = [key for key in preferred if key in available]
        fieldnames.extend(sorted(available - set(fieldnames)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_population(path: Path, population) -> None:
    X = np.asarray(population.get("X"), dtype=float)
    F = np.asarray(population.get("F"), dtype=float)
    columns = [f"x{i + 1}" for i in range(X.shape[1])] + [
        f"f{i + 1}" for i in range(F.shape[1])
    ]
    provenance = population.get("provenance")
    if provenance is not None:
        columns.append("provenance")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, values in enumerate(np.hstack([X, F])):
            row = values.tolist()
            if provenance is not None:
                row.append(provenance[index])
            writer.writerow(row)


def _is_complete(case: ExperimentCase) -> bool:
    config_path = case.output_dir / "config.json"
    metrics_path = case.output_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return False
    try:
        with config_path.open(encoding="utf-8") as handle:
            config_matches = json.load(handle) == case.to_dict()
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        return (
            config_matches
            and metrics.get("metric_schema_version") == METRIC_SCHEMA_VERSION
        )
    except (OSError, json.JSONDecodeError):
        return False


def run_case(case: ExperimentCase, force: bool = False) -> dict:
    case.validate()
    if _is_complete(case) and not force:
        return {"status": "skipped", "output_dir": str(case.output_dir), **case.to_dict()}

    output_dir = case.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, json.JSONDecodeError):
            previous = None
        previous_schema = (
            previous.get("algorithm_schema_version")
            if isinstance(previous, dict)
            else None
        )
        if previous_schema != case.algorithm_schema_version:
            raise RuntimeError(
                f"{output_dir} 已有不同 algorithm schema；禁止覆盖，请更换 --run-name"
            )
        if previous != case.to_dict():
            raise RuntimeError(
                f"{output_dir} 已有不同配置；禁止混写，请更换 --run-name"
            )
    _json_dump(config_path, case.to_dict())
    problem = make_problem(case.normalized_problem, case.n_obj, case.n_var)
    pop_size = len(reference_directions(case))
    initial_X = shared_initial_decisions(problem, pop_size, case.seed)
    initial_hash = initialization_hash(initial_X)
    suite = MetricSuite(
        problem,
        case.reference_points,
        case.high_dim_hv_samples,
        direction_directions=reference_directions(case),
    )
    history = HistoryRecorder(suite, case.max_fes, case.history_points, case.history_hv)
    started = time.perf_counter()

    extra = {}
    if case.normalized_algorithm == "IEMOEC":
        algorithm = IEMOECRunner(
            problem,
            case,
            initial_X=initial_X,
            on_checkpoint=lambda fe, pop: history.record(fe, pop),
            on_outer_selection=lambda fe, pop: history.record_event(
                fe,
                pop,
                "outer_selection",
            ),
        )
        population, outer_iterations = algorithm.run()
        n_eval = algorithm.n_eval
        pop_size = algorithm.pop_size
        extra["outer_iterations"] = outer_iterations
        extra["global_selection_count"] = algorithm.global_selection_count
        extra["origin_population_size"] = algorithm.n_origin
        extra["island_initialization"] = case.iemoec.island_initialization
        extra["island_fes_total"] = int(
            sum(row["island_fes"] for row in algorithm.outer_records)
        )
        extra["island_expansion_fes_total"] = int(
            sum(row["expansion_fes"] for row in algorithm.outer_records)
        )
        extra["island_evolution_fes_total"] = int(
            sum(row["island_evolution_fes"] for row in algorithm.outer_records)
        )
        extra["recombination_offspring_total"] = int(
            sum(row["recombination_offspring"] for row in algorithm.outer_records)
        )
    else:
        algorithm, pop_size, _ = make_baseline(case, initial_X=initial_X)
        if case.max_fes < pop_size:
            raise ValueError(f"max_fes={case.max_fes} 小于种群大小 {pop_size}")
        if case.max_fes % pop_size != 0:
            raise ValueError(
                f"为保证 baseline FE 完全一致，max_fes 必须是共同种群大小 {pop_size} 的倍数"
            )
        result = minimize(
            problem,
            algorithm,
            termination=("n_eval", case.max_fes),
            seed=case.seed,
            callback=history,
            verbose=False,
            save_history=False,
        )
        population = result.pop
        n_eval = int(result.algorithm.evaluator.n_eval)

    runtime = time.perf_counter() - started
    if n_eval != case.max_fes:
        raise RuntimeError(f"FE 预算违反：期望 {case.max_fes}，实际 {n_eval}")
    F = np.asarray(population.get("F"), dtype=float)
    if not np.all(np.isfinite(F)):
        raise RuntimeError("最终目标值含 NaN/Inf")
    final_values = suite.calculate(F, include_hv=True)
    history.finalize(n_eval, population, final_values)
    metrics = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "algorithm_schema_version": case.algorithm_schema_version,
        "algorithm_variant": case.algorithm_variant,
        "algorithm": case.normalized_algorithm,
        "algorithm_label": case.algorithm_label,
        "problem": case.normalized_problem,
        "n_obj": case.n_obj,
        "seed": case.seed,
        "max_fes": case.max_fes,
        "n_eval": n_eval,
        "population_size": int(len(population)),
        "reference_population_size": int(pop_size),
        "runtime_seconds": float(runtime),
        "hv_method": suite.hv_method,
        "initialization_hash": initial_hash,
        **final_values,
        **extra,
    }
    _write_history(output_dir / "history.csv", history.rows)
    if case.normalized_algorithm == "IEMOEC":
        _write_history(output_dir / "iemoec_diagnostics.csv", algorithm.outer_records)
    _write_population(output_dir / "final_population.csv", population)
    _json_dump(output_dir / "metrics.json", metrics)
    return {"status": "completed", "output_dir": str(output_dir), **metrics}
