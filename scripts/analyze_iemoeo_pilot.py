"""审计 IE-MOEO pilot，生成可复现的描述统计、机制诊断和 Markdown 报告。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from iemoec_experiment.manifest import result_files
from summarize import ALGORITHM_LABELS, friedman_report, load_rows, paired_tests, write_csv


def median(rows: list[dict], metric: str) -> float:
    return float(statistics.median(row[metric] for row in rows))


def descriptive_comparisons(groups: dict, target: str) -> list[dict]:
    """这里的胜负仅是中位数方向，不能当成统计显著的胜负。"""
    algorithms = sorted(next(iter(groups.values())))
    comparisons = []
    for algorithm in algorithms:
        if algorithm == target:
            continue
        for metric in ("igd_plus", "hv"):
            counts = Counter()
            for group in groups.values():
                x, y = median(group[target], metric), median(group[algorithm], metric)
                better = x < y if metric == "igd_plus" else x > y
                counts["equal" if x == y else "better" if better else "worse"] += 1
            comparisons.append({"metric": metric, "competitor": algorithm,
                                **{name: counts[name] for name in ("better", "equal", "worse")}})
    return comparisons


def analyze(root: Path, report_path: Path) -> dict:
    rows, validation = load_rows(root)
    if not rows or not any(row["algorithm"] == "IEMOEO" for row in rows):
        raise ValueError("该批次没有 IE-MOEO 结果")
    if any(row.get("n_eval") != row["max_fes"] for row in rows):
        raise ValueError("存在未耗尽声明预算的结果")
    if any(not np.isfinite(row[metric]) for row in rows for metric in ("igd_plus", "hv")):
        raise ValueError("存在非有限的 IGD+/HV")
    failures_path = root / "failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else None
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["problem"], row["n_obj"])][row["algorithm"]].append(row)
    algorithms = sorted({row["algorithm"] for row in rows})
    labels = dict(ALGORITHM_LABELS)
    if {row["algorithm_variant"] for row in rows if row["algorithm"] == "IEMOEC"} == {"s3_elite"}:
        labels["IEMOEC"] = "S3-Elite"
    if any(set(group) != set(algorithms) for group in groups.values()):
        raise ValueError("存在缺少对比算法的场景")

    ranks = defaultdict(list)
    firsts = Counter()
    scene_rows = []
    rank_matrix = []
    for (problem, n_obj), group in sorted(groups.items()):
        values = [median(group[algorithm], "igd_plus") for algorithm in algorithms]
        scene_ranks = rankdata(values)
        rank_matrix.append(scene_ranks)
        for algorithm, value, rank in zip(algorithms, values, scene_ranks):
            ranks[algorithm].append(float(rank))
            firsts[algorithm] += value == min(values)
            current = group[algorithm]
            scene_rows.append({
                "problem": problem, "n_obj": n_obj, "algorithm": algorithm,
                "seeds": len(current), "median_rank": float(rank),
                **{f"{metric}_median": median(current, metric) for metric in (
                    "igd_plus", "gd_plus", "hv", "direction_occupancy", "population_size",
                )},
            })
    mean_rank_report = friedman_report(rows)
    ranking = [{
        "algorithm": algorithm,
        "mean_based_rank": mean_rank_report["average_ranks"][algorithm],
        "median_based_rank": statistics.mean(ranks[algorithm]),
        "median_first_scenarios": int(firsts[algorithm]),
    } for algorithm in sorted(algorithms, key=lambda name: mean_rank_report["average_ranks"][name])]
    comparisons = descriptive_comparisons(groups, "IEMOEO")
    tests = paired_tests(rows, "IEMOEO", 0.05)
    ie_rows = [row for row in rows if row["algorithm"] == "IEMOEO"]
    budget = Counter()
    for row in ie_rows:
        budget.update(row["evaluation_breakdown"])
    total_fes = sum(row["n_eval"] for row in ie_rows)
    if sum(budget.values()) != total_fes:
        raise ValueError("IE-MOEO FE 分账不平衡")
    diagnostics = []
    for path in result_files(root, "iemoeo_diagnostics.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            diagnostics.extend(csv.DictReader(handle))
    total_representatives = sum(int(row["representative_count"]) for row in diagnostics)
    qualified = sum(int(row["qualified_parent_count"]) for row in diagnostics)
    rounds_rejecting = sum(int(row["qualified_parent_count"]) < int(row["representative_count"])
                          for row in diagnostics)
    objective_rows = []
    for n_obj in sorted({row["n_obj"] for row in rows}):
        current = [row for row in ie_rows if row["n_obj"] == n_obj]
        report = friedman_report([row for row in rows if row["n_obj"] == n_obj])
        objective_rows.append({
            "n_obj": n_obj, "iemoeo_mean_based_rank": report["average_ranks"]["IEMOEO"],
            "runs": len(current), "zero_hv_runs": sum(row["hv"] == 0 for row in current),
            "zero_hv_with_eligible_points": sum(row["hv"] == 0 and row["hv_eligible_solution_count"] > 0
                                                for row in current),
            "direction_occupancy_median": median(current, "direction_occupancy"),
        })
    diagnostic_report = {
        "runs": len(ie_rows), "rounds": len(diagnostics),
        "rounds_with_rejection": rounds_rejecting,
        "rejection_round_fraction": rounds_rejecting / len(diagnostics),
        "parent_rejection_fraction": 1 - qualified / total_representatives,
        "fallback_rounds": sum(row["parent_fallback"] == "True" for row in diagnostics),
        "runs_all_parents_accepted": sum(row["parent_acceptance_rate"] == 1 for row in ie_rows),
        "final_population_min": min(row["population_size"] for row in ie_rows),
        "final_population_max": max(row["population_size"] for row in ie_rows),
        "total_fes": total_fes,
        "fe_breakdown": dict(budget),
        "fe_fraction": {name: value / total_fes for name, value in budget.items()},
        "recorded_algorithm_seconds": sum(row["algorithm_runtime_seconds"] for row in ie_rows),
    }
    write_csv(output / "ranking.csv", ranking)
    write_csv(output / "scenario_medians.csv", scene_rows)
    write_csv(output / "descriptive_comparisons.csv", comparisons)
    write_csv(output / "objectives.csv", objective_rows)
    snapshot = {
        "validation": validation, "failures": failures,
        "friedman_on_scenario_means": mean_rank_report,
        "diagnostics": diagnostic_report,
        "minimum_raw_pairwise_p": min(row["p_value"] for row in tests),
        "significant_pairwise_results": sum(row["target_result"] != "=" for row in tests),
    }
    (output / "audit.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(10, max(7, len(groups) * 0.24)))
    matrix = np.asarray(rank_matrix)
    plot = axis.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(algorithms))
    axis.set_xticks(range(len(algorithms)), [labels[a] for a in algorithms], rotation=35, ha="right")
    axis.set_yticks(range(len(groups)), [f"{p.upper()} M{m}" for p, m in sorted(groups)])
    axis.set_title("IGD+ median ranks by scenario (lower is better)")
    figure.colorbar(plot, ax=axis, label="Rank")
    figure.tight_layout()
    figure.savefig(output / "igd_rank_heatmap.png", dpi=160)
    plt.close(figure)

    lines = [f"# {root.name} 实验结果分析", "", "## 审计与结论", "",
        f"- 完成 {validation['completed']}/{validation['expected']} 个任务，失败记录：{len(failures) if failures is not None else '未提供'}。",
        f"- {len(groups)} 个问题×目标数场景，{len(algorithms)} 个算法；各任务实际 FE 与预算相同。",
        "- 现有结果不支持 IE-MOEO 具有整体性能优势；正面个例需要独立确认。",
        "- S3-Elite 仅作为前期性能参照，不能作为独立进化的单因素对照。", "",
        "## 排名", "", "均值排名与现有 Friedman 输出一致；中位数排名用于稳健的描述性比较。", "",
        "| 算法 | 均值排名 | 中位数排名 | 中位数第一场景数 |", "|---|---:|---:|---:|"]
    for row in ranking:
        lines.append(f"| {labels[row['algorithm']]} | {row['mean_based_rank']:.4f} | "
                     f"{row['median_based_rank']:.4f} | {row['median_first_scenarios']} |")
    lines.extend(["", f"Friedman χ²={mean_rank_report['statistic']:.4f}，p={mean_rank_report['p_value']:.6g}。"
                  "这只是总体差异，不能确定哪一对算法显著不同。", "", "## 对各对手的描述性比较", "",
                  "下面统计 IE-MOEO 的场景中位数更好/相同/更差，**不是显著性胜平负**。", "",
                  "| 对手 | IGD+ 更好/相同/更差 | HV 更好/相同/更差 |", "|---|---|---|"])
    for algorithm in algorithms:
        current = [row for row in comparisons if row["competitor"] == algorithm]
        if current:
            descriptions = ["/".join(str(row[k]) for k in ("better", "equal", "worse")) for row in current]
            lines.append(f"| {labels[algorithm]} | {' | '.join(descriptions)} |")
    lines.extend(["", "## 为什么显著性表全是平局", "",
        "本批次每个场景只有 5 个 seed。无零差值时，双侧精确 Wilcoxon 符号秩检验的最小 p 值为 "
        "2/2⁵=0.0625，本来就无法低于 0.05；再对多个比较进行 Holm 校正后更不可能显著。",
        f"实际最小原始 p={snapshot['minimum_raw_pairwise_p']:.4f}，校正后显著比较数为 "
        f"{snapshot['significant_pairwise_results']}。不能据此声称等效或没有差别。", "",
        "## 目标数与覆盖", "", "| M | IE-MOEO 均值排名 | 零 HV / 运行数 | 有参考框内点但 MC HV 为零 | 方向占用率中位数 |",
        "|---:|---:|---:|---:|---:|"])
    for row in objective_rows:
        lines.append(f"| {row['n_obj']} | {row['iemoeo_mean_based_rank']:.4f} | "
                     f"{row['zero_hv_runs']}/{row['runs']} | {row['zero_hv_with_eligible_points']} | "
                     f"{row['direction_occupancy_median']:.4f} |")
    lines.extend(["", "高维 HV 使用固定 Monte Carlo 采样，零值可能来自参考框外或有限采样分辨率，不能等同于完全没有优化。", "",
        "## 机制是否真正触发", "",
        f"- {len(diagnostics)} 轮中，只有 {rounds_rejecting} 轮出现亲本拒绝，占 {rounds_rejecting / len(diagnostics):.2%}。",
        f"- 按代表数量加权的拒绝率为 {diagnostic_report['parent_rejection_fraction']:.2%}；"
        f"{diagnostic_report['runs_all_parents_accepted']}/{len(ie_rows)} 次运行的亲本入选率恒为 1。",
        f"- 保底补足出现 {diagnostic_report['fallback_rounds']} 轮。邻域检查只在已评价代表间比较支配，不能认证局部极值。",
        f"- 最终非支配集大小范围为 {diagnostic_report['final_population_min']}–{diagnostic_report['final_population_max']}，"
        "对照算法通常输出 N 个个体，不能把结果写成所有算法输出规模相同。", "",
        "| FE 来源 | 次数 | 比例 |", "|---|---:|---:|"])
    for name, value in budget.items():
        lines.append(f"| {name} | {value} | {value / total_fes:.2%} |")
    lines.extend(["", "默认仅局部进化一代，各岛共用同一标量化权重；重建消耗较多 FE、组合消耗较少 FE。"
        "这些是待验证的解释，不能直接认定为性能落后的因果原因。", "",
        "## 48 个场景的 IE-MOEO 表现", "",
        "| 场景 | IGD+ 中位数 | 排名 | HV 中位数 | GD+ 中位数 | 方向占用率 | 最终解数中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for row in scene_rows:
        if row["algorithm"] == "IEMOEO":
            lines.append(f"| {row['problem'].upper()} M{row['n_obj']} | {row['igd_plus_median']:.6g} | "
                         f"{row['median_rank']:.1f} | {row['hv_median']:.6g} | {row['gd_plus_median']:.6g} | "
                         f"{row['direction_occupancy_median']:.4f} | {row['population_size_median']:.0f} |")
    lines.extend(["", "## 结论边界", "",
        "WFG4 M10 的中位数排名第一是值得追踪的开发集现象，不能当成全面优势。"
        "本批次 seeds 1–5 已经用于分析和方案设计；确认实验冻结后使用 seeds 31–60。",
        "普通 pilot 包含历史指标回调且使用任务并行，记录时间不用于宣称内核速度或岛内并行优势。",
        f"派生数据位于 `{output.as_posix()}`；运行 `python scripts/analyze_iemoeo_pilot.py {root.as_posix()}` 可以重建。", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = args.report or ROOT / "reports" / f"{args.results.name}_analysis.md"
    audit = analyze(args.results, report)
    print(f"完成审计：{audit['validation']['completed']} 个结果；报告：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
