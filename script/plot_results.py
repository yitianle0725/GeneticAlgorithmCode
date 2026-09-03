#!/usr/bin/env python
"""实验完成后统一绘制收敛曲线、箱线图和平行坐标图。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def convergence(root: Path, output: Path) -> None:
    groups = defaultdict(list)
    for path in root.rglob("history.csv"):
        with (path.parent / "config.json").open(encoding="utf-8") as handle:
            cfg = json.load(handle)
        rows = read_csv(path)
        if rows:
            groups[(cfg["problem"], cfg["n_obj"], cfg["algorithm"])].append((cfg, rows))
    for (problem, n_obj, algorithm), runs in groups.items():
        count = min(len(rows) for _, rows in runs)
        curves = np.asarray([[float(row["igd_plus"]) for row in rows[:count]] for _, rows in runs])
        x = np.asarray([float(row["fe"]) for row in runs[0][1][:count]])
        median = np.median(curves, axis=0)
        low, high = np.percentile(curves, [25, 75], axis=0)
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(x, median, label=algorithm)
        plt.fill_between(x, low, high, alpha=0.25, label="IQR")
        plt.xlabel("Function Evaluations")
        plt.ylabel("IGD+")
        plt.title(f"{problem.upper()} M={n_obj}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"convergence_{problem}_M{n_obj}_{algorithm}.png", dpi=180)
        plt.close()


def boxplots(root: Path, output: Path) -> None:
    groups = defaultdict(lambda: defaultdict(list))
    for path in root.rglob("metrics.json"):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        groups[(row["problem"], row["n_obj"])][row["algorithm"]].append(row["igd_plus"])
    for (problem, n_obj), values in groups.items():
        algorithms = sorted(values)
        plt.figure(figsize=(7.2, 4.4))
        plt.boxplot([values[a] for a in algorithms], tick_labels=algorithms, showmeans=True)
        plt.ylabel("Final IGD+")
        plt.title(f"{problem.upper()} M={n_obj}")
        plt.tight_layout()
        plt.savefig(output / f"boxplot_{problem}_M{n_obj}.png", dpi=180)
        plt.close()


def parallel_coordinates(root: Path, output: Path) -> None:
    records = []
    for path in root.rglob("final_population.csv"):
        config_path = path.parent / "config.json"
        with config_path.open(encoding="utf-8") as handle:
            cfg = json.load(handle)
        rows = read_csv(path)
        objective_columns = [key for key in rows[0] if key.startswith("f")]
        F = np.asarray([[float(row[key]) for key in objective_columns] for row in rows])
        records.append((path, cfg, objective_columns, F))
    bounds = {}
    for _, cfg, _, F in records:
        key = (cfg["problem"], cfg["n_obj"])
        bounds.setdefault(key, []).append(F)
    bounds = {
        key: (np.min(np.vstack(values), axis=0), np.max(np.vstack(values), axis=0))
        for key, values in bounds.items()
    }
    for path, cfg, objective_columns, F in records:
        low, high = bounds[(cfg["problem"], cfg["n_obj"])]
        normalized = (F - low) / np.maximum(high - low, 1e-12)
        plt.figure(figsize=(max(7, cfg["n_obj"] * 0.65), 4.5))
        for line in normalized[:300]:
            plt.plot(range(cfg["n_obj"]), line, alpha=0.16, linewidth=0.7)
        plt.xticks(range(cfg["n_obj"]), objective_columns)
        plt.ylabel("Shared normalized objective (display only)")
        plt.title(f"{cfg['problem'].upper()} M={cfg['n_obj']} {cfg['algorithm']} seed={cfg['seed']}")
        plt.tight_layout()
        plt.savefig(output / f"parallel_{cfg['problem']}_M{cfg['n_obj']}_{cfg['algorithm']}_s{cfg['seed']:03d}.png", dpi=180)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--kind", choices=["all", "convergence", "boxplot", "parallel"], default="all")
    args = parser.parse_args()
    output = args.results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if args.kind in ("all", "convergence"):
        convergence(args.results, output)
    if args.kind in ("all", "boxplot"):
        boxplots(args.results, output)
    if args.kind in ("all", "parallel"):
        parallel_coordinates(args.results, output)
    print(f"图片已保存到 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
