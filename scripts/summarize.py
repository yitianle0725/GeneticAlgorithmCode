#!/usr/bin/env python
"""汇总多种子结果并执行配对 Wilcoxon、Holm 与 Friedman 检验。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iemoec_experiment.metrics import METRIC_SCHEMA_VERSION  # noqa: E402
from iemoec_experiment.manifest import load_manifest, validate_result_rows  # noqa: E402


METRICS = (
    "igd_plus", "gd_plus", "hv", "spacing", "direction_occupancy",
    "feasible_direction_coverage", "feasible_ratio", "min_cv", "mean_cv",
    "nd_ratio", "runtime_seconds",
)
LOWER_IS_BETTER = {
    "igd_plus", "gd_plus", "spacing", "min_cv", "mean_cv",
    "runtime_seconds",
}
ALGORITHM_LABELS = {
    "NSGA2": "NSGA-II",
    "NSGA3": "NSGA-III",
    "MOEAD": "MOEA/D-TCH",
    "MOEADPBI": "MOEA/D-PBI",
    "RVEA": "RVEA",
    "AGEMOEA2": "AGE-MOEA2",
    "AGEMOEA2STABLE": "AGE-MOEA2-Stable",
    "IEMOEC": "IEMOEC",
}


def load_rows(root: Path, allow_incomplete: bool = False) -> tuple[list[dict], dict]:
    manifest = load_manifest(root)
    rows = []
    for path in root.rglob("metrics.json"):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("metric_schema_version") != manifest["metric_schema_version"]:
            raise RuntimeError(
                f"{path} does not match manifest metric schema "
                f"{manifest['metric_schema_version']}"
            )
        row["path"] = str(path.parent)
        rows.append(row)
    validation = validate_result_rows(rows, manifest, allow_incomplete)
    validation["run_name"] = manifest["run_name"]
    validation["metric_schema_version"] = manifest["metric_schema_version"]
    return rows, validation


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def vargha_delaney_a12(
    target: np.ndarray,
    competitor: np.ndarray,
    lower_is_better: bool,
) -> float:
    """返回目标算法优于竞争算法的概率，平局计 0.5。"""
    differences = target[:, None] - competitor[None, :]
    wins = differences < 0 if lower_is_better else differences > 0
    ties = differences == 0
    return float((np.sum(wins) + 0.5 * np.sum(ties)) / differences.size)


def a12_magnitude(value: float) -> str:
    distance = abs(value - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    available = {key for row in rows for key in row}
    fieldnames = list(rows[0])
    fieldnames.extend(sorted(available - set(fieldnames)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["problem"], row["n_obj"], row["algorithm"])].append(row)
    output = []
    for (problem, n_obj, algorithm), values in sorted(groups.items()):
        item = {
            "problem": problem,
            "n_obj": n_obj,
            "algorithm": algorithm,
            "algorithm_label": values[0].get(
                "algorithm_label",
                ALGORITHM_LABELS.get(algorithm, algorithm),
            ),
            "n": len(values),
        }
        for metric in METRICS:
            data = np.asarray([
                value[metric]
                for value in values
                if value.get(metric) is not None
            ], dtype=float)
            if not len(data):
                continue
            item[f"{metric}_n_valid"] = len(data)
            item[f"{metric}_mean"] = float(np.mean(data))
            item[f"{metric}_std"] = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
            item[f"{metric}_median"] = float(np.median(data))
            item[f"{metric}_iqr"] = float(np.percentile(data, 75) - np.percentile(data, 25))
        output.append(item)
    return output


def paired_tests(rows: list[dict], target: str, alpha: float) -> list[dict]:
    lookup = {(r["problem"], r["n_obj"], r["algorithm"], r["seed"]): r for r in rows}
    instances = sorted({(r["problem"], r["n_obj"]) for r in rows})
    algorithms = sorted({r["algorithm"] for r in rows if r["algorithm"] != target})
    tests = []
    for metric in ("igd_plus", "hv"):
        for problem, n_obj in instances:
            for algorithm in algorithms:
                seeds = sorted({
                    r["seed"] for r in rows
                    if r["problem"] == problem and r["n_obj"] == n_obj
                    and (problem, n_obj, target, r["seed"]) in lookup
                    and (problem, n_obj, algorithm, r["seed"]) in lookup
                    and lookup[(problem, n_obj, target, r["seed"])].get(metric)
                    is not None
                    and lookup[(problem, n_obj, algorithm, r["seed"])].get(metric)
                    is not None
                })
                if not seeds:
                    continue
                x = np.asarray([lookup[(problem, n_obj, target, s)][metric] for s in seeds])
                y = np.asarray([lookup[(problem, n_obj, algorithm, s)][metric] for s in seeds])
                if np.allclose(x, y):
                    p_value = 1.0
                else:
                    p_value = float(wilcoxon(x, y, alternative="two-sided").pvalue)
                tests.append({
                    "metric": metric, "problem": problem, "n_obj": n_obj,
                    "target": target, "competitor": algorithm, "n_pairs": len(seeds),
                    "target_median": float(np.median(x)),
                    "competitor_median": float(np.median(y)),
                    "p_value": p_value,
                    "a12_target_superiority": vargha_delaney_a12(
                        x,
                        y,
                        metric in LOWER_IS_BETTER,
                    ),
                })
    adjusted = holm_adjust([test["p_value"] for test in tests])
    for test, p_adjusted in zip(tests, adjusted):
        test["p_holm"] = p_adjusted
        test["a12_magnitude"] = a12_magnitude(
            test["a12_target_superiority"]
        )
        if p_adjusted >= alpha:
            symbol = "="
        else:
            target_better = test["target_median"] < test["competitor_median"]
            if test["metric"] not in LOWER_IS_BETTER:
                target_better = not target_better
            symbol = "+" if target_better else "-"
        test["target_result"] = symbol
    return tests


def friedman_report(rows: list[dict]) -> dict:
    algorithms = sorted({row["algorithm"] for row in rows})
    instances = sorted({(row["problem"], row["n_obj"]) for row in rows})
    lookup = {
        (row["problem"], row["n_obj"], row["algorithm"], row["seed"]): row
        for row in rows
    }
    means = defaultdict(dict)
    common_seed_counts = {}
    excluded_run_blocks = 0
    for problem, n_obj in instances:
        seed_sets = [
            {
                row["seed"] for row in rows
                if row["problem"] == problem
                and row["n_obj"] == n_obj
                and row["algorithm"] == algorithm
            }
            for algorithm in algorithms
        ]
        common_seeds = set.intersection(*seed_sets) if seed_sets else set()
        common_seeds = {
            seed
            for seed in common_seeds
            if all(
                lookup[(problem, n_obj, algorithm, seed)].get("igd_plus")
                is not None
                for algorithm in algorithms
            )
        }
        all_seeds = set.union(*seed_sets) if seed_sets else set()
        common_seed_counts[f"{problem}-M{n_obj}"] = len(common_seeds)
        excluded_run_blocks += len(all_seeds - common_seeds)
        for algorithm in algorithms:
            values = [
                lookup[(problem, n_obj, algorithm, seed)]["igd_plus"]
                for seed in sorted(common_seeds)
            ]
            if common_seeds:
                means[(problem, n_obj)][algorithm] = float(np.mean(values))
    complete = [instance for instance in instances if len(means[instance]) == len(algorithms)]
    ranks = {algorithm: [] for algorithm in algorithms}
    for instance in complete:
        values = [means[instance][algorithm] for algorithm in algorithms]
        for algorithm, rank in zip(algorithms, rankdata(values, method="average")):
            ranks[algorithm].append(float(rank))
    report = {
        "metric": "igd_plus", "blocks": len(complete), "algorithms": algorithms,
        "average_ranks": {algorithm: float(np.mean(value)) if value else None for algorithm, value in ranks.items()},
        "common_seed_counts": common_seed_counts,
        "excluded_incomplete_seed_blocks": excluded_run_blocks,
        "missing_values_imputed": False,
    }
    if len(algorithms) >= 3 and len(complete) >= 2:
        samples = [[means[instance][algorithm] for instance in complete] for algorithm in algorithms]
        statistic, p_value = friedmanchisquare(*samples)
        report.update(statistic=float(statistic), p_value=float(p_value))
    else:
        report.update(statistic=None, p_value=None, note="至少需要 3 个算法和 2 个完整问题块")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="例如 results/pilot")
    parser.add_argument("--target", default="IEMOEC")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow missing manifest tasks for exploratory summaries",
    )
    args = parser.parse_args()
    try:
        rows, validation = load_rows(args.results, args.allow_incomplete)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not rows:
        raise SystemExit(f"未在 {args.results} 找到 metrics.json")
    write_csv(args.results / "summary.csv", summarize(rows))
    write_csv(args.results / "wilcoxon_holm.csv", paired_tests(rows, args.target, args.alpha))
    with (args.results / "friedman.json").open("w", encoding="utf-8") as handle:
        json.dump(friedman_report(rows), handle, ensure_ascii=False, indent=2)
    with (args.results / "summary_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
    print(f"已汇总 {len(rows)} 次独立运行: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
