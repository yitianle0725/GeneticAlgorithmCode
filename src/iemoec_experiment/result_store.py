"""按算法保存结果，在不同实验批次之间复用已完成的同配置任务。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import uuid
from functools import cache
from pathlib import Path

from .config import ExperimentCase, default_ref_partitions
from .manifest import TASK_FIELDS, task_key
from .metrics import metric_schema_version_for_problem
from .result_lock import result_lock


CUSTOM_ALGORITHMS = {"IEMOEC", "IEMOEO"}


def comparable_config(config: dict) -> dict:
    """排除批次位置，以及公共基线根本不使用的 IEMOEC 参数。"""
    values = dict(config)
    values.pop("output_root", None)
    values.pop("result_store", None)
    if values["algorithm"] not in CUSTOM_ALGORITHMS:
        values.pop("iemoec", None)
    return values


@cache
def implementation_signature(algorithm: str) -> dict:
    """IEMOEO 自身的修改只影响 IEMOEO，不使成熟基线失效。"""
    modules = [
        "factory", "problems", "initialization", "normalization",
        "constraints", "metrics", "runner",
    ]
    if algorithm == "IEMOEO":
        modules.extend(["iemoeo", "iemoeo_diagnostics"])
    elif algorithm == "IEMOEC":
        modules.extend([
            "iemoec", "principle", "directions", "directional_memory", "source_budget",
        ])
    directory = Path(__file__).parent
    sources = {
        name: hashlib.sha256((directory / f"{name}.py").read_bytes()).hexdigest()
        for name in modules
    }
    versions = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for package in ("numpy", "scipy", "pymoo", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {"sources": sources, "versions": versions}


def result_identity(case: ExperimentCase) -> str:
    values = {
        "config": comparable_config(case.to_dict()),
        "implementation": implementation_signature(case.normalized_algorithm),
        "metric_schema": metric_schema_version_for_problem(case.normalized_problem),
        "reference_partitions": case.ref_partitions or default_ref_partitions(case.n_obj),
    }
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stored_result_directory(case: ExperimentCase) -> Path:
    return (
        Path(case.result_store)
        / case.normalized_algorithm / case.normalized_problem.upper()
        / f"M{case.n_obj}" / f"seed_{case.seed:03d}"
        / f"FE{case.max_fes}_{result_identity(case)[:16]}"
    )


def complete_result(directory: Path, case: ExperimentCase) -> bool:
    required = ["config.json", "metrics.json", "history.csv", "final_population.csv"]
    if case.normalized_algorithm == "IEMOEO":
        required.extend(["iemoeo_diagnostics.csv", "baseline_solution.json"])
    elif case.normalized_algorithm == "IEMOEC":
        required.append("iemoec_diagnostics.csv")
        if case.iemoec.variant in ("principle", "s4"):
            required.append("lineage_audit.csv")
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        expected = case.to_dict()
        return (
            comparable_config(config) == comparable_config(expected)
            and metrics.get("metric_schema_version")
            == metric_schema_version_for_problem(case.normalized_problem)
            and metrics.get("n_eval") == case.max_fes
            and all(metrics.get(field) == expected[field] for field in TASK_FIELDS)
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def stored_result_complete(case: ExperimentCase) -> bool:
    directory = case.output_dir
    try:
        identity = json.loads((directory / "storage_identity.json").read_text(encoding="utf-8"))
        return identity["identity"] == result_identity(case) and complete_result(directory, case)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def legacy_result_candidates(root: Path, cases: list[ExperimentCase]) -> dict[tuple, tuple[str, ...]]:
    """只读取旧批次清单建立索引，避免为每个 seed 递归扫描全部结果。"""
    wanted = {
        task_key(case.to_dict()): case.max_fes for case in cases
        if case.normalized_algorithm not in CUSTOM_ALGORITHMS
    }
    candidates: dict[tuple, list[str]] = {}
    if not root.exists() or not wanted:
        return {}
    for manifest_path in sorted(root.glob("*/experiment_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for task in manifest["tasks"]:
                key = task_key(task)
                if "result_path" in task or wanted.get(key) != task.get("max_fes"):
                    continue
                problem, n_obj, algorithm, seed = key
                directory = (
                    manifest_path.parent / problem.upper() / f"M{n_obj}"
                    / algorithm / f"seed_{seed:03d}"
                )
                candidates.setdefault(key, []).append(str(directory))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {key: tuple(paths) for key, paths in candidates.items()}


def _write_identity(case: ExperimentCase, imported_from: str | None = None) -> None:
    from .runner import _json_dump

    _json_dump(case.output_dir / "storage_identity.json", {
        "identity": result_identity(case),
        "implementation": implementation_signature(case.normalized_algorithm),
        "imported_from": imported_from,
        # 历史基线没有源码/环境摘要，迁入时依据 schema 和完整配置核对。
        "legacy_implementation_verified": imported_from is None,
    })


def run_stored_case(
    case: ExperimentCase, force: bool, verbose_fe: bool,
    legacy_candidates: tuple[str, ...],
) -> dict:
    from .runner import _execute_case, _json_dump

    directory = case.output_dir
    skipped = {"status": "skipped", "output_dir": str(directory), **case.to_dict()}
    if not force and stored_result_complete(case):
        return skipped
    with result_lock(directory):
        if not force and stored_result_complete(case):
            return skipped
        if not force and not directory.exists() and case.normalized_algorithm not in CUSTOM_ALGORITHMS:
            for source in legacy_candidates:
                if not complete_result(Path(source), case):
                    continue
                staging = directory.with_name(directory.name + f".import_{uuid.uuid4().hex}")
                try:
                    shutil.copytree(source, staging)
                    _json_dump(staging / "config.json", case.to_dict())
                    os.replace(staging, directory)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
                _write_identity(case, source)
                return {**skipped, "reused_from": source}
        # 中途失败时，不能让旧的完成标记掩盖本次未完成的写入。
        (directory / "storage_identity.json").unlink(missing_ok=True)
        result = _execute_case(case, force, verbose_fe)
        if not complete_result(directory, case):
            raise RuntimeError(f"运行结果不完整，不能标记为可复用：{directory}")
        _write_identity(case)
        return result
