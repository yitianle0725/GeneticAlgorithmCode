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


METRICS = ("igd_plus", "gd_plus", "hv", "spacing", "nd_ratio", "runtime_seconds")
LOWER_IS_BETTER = {"igd_plus", "gd_plus", "spacing", "runtime_seconds"}
ALGORITHM_LABELS = {
    "NSGA2": "NSGA-II",
    "NSGA3": "NSGA-III",
    "MOEAD": "MOEA/D-TCH",
    "IEMOEC": "IEMOEC",
}


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("metrics.json"):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            raise RuntimeError(
                f"{path} 使用旧指标定义；请重新运行对应实验以生成 GD+ 和 pymoo Spacing"
            )
        row["path"] = str(path.parent)
        rows.append(row)
    return rows


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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
            "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
            "n": len(values),
        }
        for metric in METRICS:
            data = np.asarray([value[metric] for value in values], dtype=float)
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
                })
    adjusted = holm_adjust([test["p_value"] for test in tests])
    for test, p_adjusted in zip(tests, adjusted):
        test["p_holm"] = p_adjusted
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
    means = defaultdict(dict)
    for problem, n_obj in instances:
        for algorithm in algorithms:
            values = [r["igd_plus"] for r in rows if r["problem"] == problem and r["n_obj"] == n_obj and r["algorithm"] == algorithm]
            if values:
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
    args = parser.parse_args()
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"未在 {args.results} 找到 metrics.json")
    write_csv(args.results / "summary.csv", summarize(rows))
    write_csv(args.results / "wilcoxon_holm.csv", paired_tests(rows, args.target, args.alpha))
    with (args.results / "friedman.json").open("w", encoding="utf-8") as handle:
        json.dump(friedman_report(rows), handle, ensure_ascii=False, indent=2)
    print(f"已汇总 {len(rows)} 次独立运行: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
