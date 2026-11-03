"""按预登记对照汇总 IE-MOEO 机制结果，区分主效应、交互和激活诊断。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from iemoec_experiment.manifest import result_files
from iemoec_experiment.problems import make_problem
from summarize import holm_adjust, load_rows, paired_tests, write_csv


def apply_global_holm(rows: list[dict], field: str = "p_holm") -> None:
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row[field] = adjusted


def evidence_scope(stage: str) -> str:
    if stage in ("confirm", "many_objective"):
        return "confirmation"
    return {"smoke": "implementation_only", "pilot": "development", "timing": "timing_diagnostic"}[stage]


def paired_log_ratio(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    """对数尺度降低不同问题数值量纲的影响；零值下限事先固定。"""
    return np.log(np.maximum(target, 1e-12)) - np.log(np.maximum(control, 1e-12))


def scene_by_seed(rows: list[dict], problem: str, n_obj: int) -> dict:
    return {row["seed"]: row for row in rows
            if row["problem"] == problem and row["n_obj"] == n_obj}


def decision_diversity(row: dict) -> float:
    """保存解集中的归一化决策 RMS 距离，固定采样规则，最多 2000 对。"""
    directory = Path(row["path"])
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    problem = make_problem(row["problem"], row["n_obj"], config.get("n_var"))
    with (directory / "final_population.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) < 2:
        return 0.0
    X = np.asarray([[float(record[f"x{i + 1}"]) for i in range(problem.n_var)] for record in records])
    X = (X - problem.xl) / (problem.xu - problem.xl)
    rng = np.random.default_rng(20261102 + row["seed"])
    count = min(2000, len(X) * (len(X) - 1))
    left = rng.integers(len(X), size=count)
    right = rng.integers(len(X) - 1, size=count)
    right += right >= left
    return float(np.mean(np.sqrt(np.mean((X[left] - X[right]) ** 2, axis=1))))


def interaction_rows(arms: dict[str, list[dict]], interactions: list[dict]) -> list[dict]:
    reports = []
    for interaction in interactions:
        names = interaction["cells"]
        if not all(name in arms for name in names):
            continue
        lookup = {name: {(r["problem"], r["n_obj"], r["seed"]): r for r in arms[name]} for name in names}
        keys = set(lookup[names[0]])
        if any(set(lookup[name]) != keys for name in names):
            raise ValueError("交互实验的场景和 seeds 不完整配对")
        for problem, n_obj in sorted({(p, m) for p, m, _ in keys}):
            seeds = sorted(s for p, m, s in keys if p == problem and m == n_obj)
            arrays = [np.asarray([lookup[name][(problem, n_obj, seed)]["igd_plus"] for seed in seeds])
                      for name in names]
            difference = paired_log_ratio(arrays[0], arrays[1]) - paired_log_ratio(arrays[2], arrays[3])
            p_value = 1.0 if np.all(difference == 0) else float(wilcoxon(difference).pvalue)
            reports.append({
                "interaction": interaction["id"], "problem": problem, "n_obj": n_obj,
                "n_pairs": len(seeds), "median_log_interaction": float(np.median(difference)),
                "p_value": p_value,
            })
    apply_global_holm(reports)
    return reports


def diagnostic_summary(name: str, rows: list[dict], rounds: list[dict]) -> dict:
    def total(field: str) -> int:
        return sum(int(row.get(field) or 0) for row in rounds)

    budget = sum(row["n_eval"] for row in rows)
    return {
        "variant": name, "runs": len(rows), "rounds": len(rounds),
        "rounds_rejecting_parents": sum(int(r["qualified_parent_count"]) < int(r["representative_count"]) for r in rounds),
        "rounds_with_foreign_parents": sum(int(r["foreign_parent_count"]) > 0 for r in rounds),
        "fallback_rounds": sum(r["parent_fallback"] == "True" for r in rounds),
        "combination_fe_fraction": sum(r["evaluation_breakdown"]["combination"] for r in rows) / budget,
        "expansion_fe_fraction": sum(r["evaluation_breakdown"]["initial_expansion"] + r["evaluation_breakdown"]["later_expansions"] for r in rows) / budget,
        "combination_reallocated_fes": total("combination_reallocated_fes"),
        "combination_budget_requested": total("combination_budget_requested"),
        "extra_local_generation_calls": total("extra_local_generation_calls"),
        "audited_pair_count": total("audited_pair_count"),
        "near_f_far_x_pair_count": total("near_f_far_x_pair_count"),
        "near_f_far_x_both_retained": total("near_f_far_x_both_retained"),
        "near_f_far_x_endpoints_retained": total("near_f_far_x_endpoints_retained"),
        "crowding_rounds": sum(r.get("survival_method") == "rank_crowding" for r in rounds),
        "representative_slots": total("representative_count"),
        "unique_representative_slots": total("unique_representative_count"),
        "representative_island_slots": total("representative_source_islands"),
        "schema2_diagnostics_available": all("representative_selection" in r for r in rounds),
        "rounds_check_changed_parent_pool": sum(r.get("parent_pool_changed_by_check") == "True" for r in rounds),
        "combination_parent_pairs": total("combination_parent_pairs"),
        "combination_distinct_parent_pairs": total("combination_distinct_parent_pairs"),
        "combination_parent_copies_before_mutation": total("combination_parent_copies_before_mutation"),
        "combination_parent_copies_after_mutation": total("combination_parent_copies_after_mutation"),
        "combination_block_observations": total("combination_block_observations"),
        "combination_intact_blocks_before_mutation": total("combination_intact_blocks_before_mutation"),
        "combination_intact_blocks_after_mutation": total("combination_intact_blocks_after_mutation"),
        "combination_unique_candidates": total("combination_unique_candidates"),
        "combination_children_selected": total("combination_children_selected"),
        "combination_children_nondominated": total("combination_children_nondominated"),
        "schema3_diagnostics_available": bool(rounds) and all("parent_check_mode" in r for r in rounds),
    }


def contrast_overview(tests: list[dict]) -> list[dict]:
    """分开报告场景中位数方向与校正后的显著胜负，避免将数值优势当作确认。"""
    groups = {}
    for row in tests:
        groups.setdefault((row["contrast"], row["metric"]), []).append(row)
    overview = []
    for (contrast, metric), rows in groups.items():
        direction = -1 if metric == "igd_plus" else 1
        differences = [direction * (r["target_median"] - r["competitor_median"]) for r in rows]
        relative = [100 * difference / abs(row["competitor_median"])
                    for difference, row in zip(differences, rows) if row["competitor_median"] != 0]
        overview.append({
            "contrast": contrast, "metric": metric, "claim": rows[0]["claim"],
            "target": rows[0]["target"], "control": rows[0]["competitor"],
            "scenes": len(rows),
            "median_better": sum(value > 0 for value in differences),
            "median_equal": sum(value == 0 for value in differences),
            "median_worse": sum(value < 0 for value in differences),
            "median_relative_improvement_percent": float(np.median(relative)) if relative else None,
            "relative_change_scenes": len(relative),
            "significant_better": sum(row["target_result"] == "+" for row in rows),
            "significant_worse": sum(row["target_result"] == "-" for row in rows),
            "min_p_holm": min(row["p_holm"] for row in rows),
            "evidence_scope": rows[0]["evidence_scope"],
        })
    return overview


def interaction_overview(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["interaction"], []).append(row)
    return [{
        "interaction": name, "scenes": len(group),
        "negative": sum(row["median_log_interaction"] < 0 for row in group),
        "zero": sum(row["median_log_interaction"] == 0 for row in group),
        "positive": sum(row["median_log_interaction"] > 0 for row in group),
        "significant_negative": sum(row["median_log_interaction"] < 0 and row["p_holm"] < .05 for row in group),
        "significant_positive": sum(row["median_log_interaction"] > 0 and row["p_holm"] < .05 for row in group),
        "min_p_holm": min(row["p_holm"] for row in group),
        "interpretation": "负值表示前两组的相对效应更有利；不自动代表两个主效应均有益",
    } for name, group in groups.items()]


def summarize_study(root: Path) -> dict:
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    plan = schedule["plan"]
    arms = {}
    activation = []
    run_metrics = []
    for command in schedule["commands"]:
        name = command["variant"]
        arm_root = root / name
        rows, validation = load_rows(arm_root, allow_incomplete=True)
        if validation["expected"] != command["tasks"]:
            raise ValueError(f"{name} 的清单任务数量与登记不符")
        if validation["missing"]:
            missing = ", ".join(f"{p.upper()} M{m} seed={seed}" for p, m, _, seed in validation["missing"])
            raise RuntimeError(
                f"机制组 {name} 未完成：{validation['completed']}/{command['tasks']}；缺失：{missing}。\n"
                f"失败详情：{arm_root / 'failures.json'}\n"
                "请在原机制运行命令中增加 --retry-incomplete --execute 补齐后再汇总；"
                "不跳过缺失任务生成完整统计。"
            )
        if any(row.get("n_eval") != row["max_fes"] for row in rows):
            raise ValueError(f"{name} 存在实际 FE 与预算不一致的结果")
        for row in rows:
            directory = Path(row["path"])
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            identity = json.loads((directory / "storage_identity.json").read_text(encoding="utf-8"))
            if config["iemoeo"] != command["iemoeo_config"]:
                raise ValueError(f"{name} 的 IE-MOEO 参数不符合登记")
            if identity["implementation"] != schedule["implementation"]:
                raise ValueError(f"{name} 的源码/依赖与冻结记录不同")
            # 只在分析内使用机制组名称，原 metrics.json 的算法名保持 IEMOEO。
            row["algorithm"] = name
            row["decision_diversity"] = decision_diversity(row)
            run_metrics.append({"variant": name, **{key: row[key] for key in (
                "problem", "n_obj", "seed", "max_fes", "igd_plus", "hv", "gd_plus",
                "direction_occupancy", "decision_diversity", "population_size",
                "algorithm_runtime_seconds",
            )}})
        arms[name] = rows
        rounds = []
        for path in result_files(arm_root, "iemoeo_diagnostics.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rounds.extend(csv.DictReader(handle))
        activation.append(diagnostic_summary(name, rows, rounds))

    tests = []
    for contrast in plan["contrasts"]:
        target, control = contrast["target"], contrast["control"]
        if target not in arms or control not in arms:
            continue
        target_keys = {(r["problem"], r["n_obj"], r["seed"], r["max_fes"]) for r in arms[target]}
        control_keys = {(r["problem"], r["n_obj"], r["seed"], r["max_fes"]) for r in arms[control]}
        if target_keys != control_keys:
            raise ValueError(f"{contrast['id']} 未按相同场景、seed 和 FE 配对")
        current = paired_tests(arms[target] + arms[control], target, 0.05)
        for row in current:
            row["contrast"] = contrast["id"]
            row["claim"] = contrast["claim"]
            row["evidence_scope"] = evidence_scope(schedule["stage"])
            if row["n_pairs"] == 1:
                row["a12_magnitude"] = "not_interpreted_single_seed"
            target_group = scene_by_seed(arms[target], row["problem"], row["n_obj"])
            control_group = scene_by_seed(arms[control], row["problem"], row["n_obj"])
            seeds = sorted(target_group)
            x = np.asarray([target_group[s][row["metric"]] for s in seeds])
            y = np.asarray([control_group[s][row["metric"]] for s in seeds])
            better = x < y if row["metric"] == "igd_plus" else x > y
            row["paired_win_fraction"] = float(np.mean(better + 0.5 * (x == y)))
            row["paired_median_difference"] = float(np.median(x - y))
            row["median_log_igd_ratio"] = float(np.median(paired_log_ratio(x, y))) if row["metric"] == "igd_plus" else None
            tests.append(row)
    # 对本套件全部登记对照、场景、IGD+/HV 联合校正，不能按结果拆分检验族。
    apply_global_holm(tests)
    for row in tests:
        if row["p_holm"] >= 0.05:
            row["target_result"] = "="
        else:
            better = row["target_median"] < row["competitor_median"]
            if row["metric"] == "hv":
                better = row["target_median"] > row["competitor_median"]
            row["target_result"] = "+" if better else "-"
    interactions = interaction_rows(arms, plan["interactions"])
    output = root / "mechanism_summary"
    output.mkdir(exist_ok=True)
    write_csv(output / "registered_contrasts.csv", tests)
    write_csv(output / "contrast_overview.csv", contrast_overview(tests))
    write_csv(output / "interactions.csv", interactions)
    write_csv(output / "interaction_overview.csv", interaction_overview(interactions))
    write_csv(output / "activation.csv", activation)
    write_csv(output / "run_metrics.csv", run_metrics)
    summary = {
        "stage": schedule["stage"], "suite": schedule["suite"],
        "registration_sha256": schedule["registration_sha256"],
        "completed": len(run_metrics), "contrasts": len(tests),
        "significant_contrasts": sum(row["target_result"] != "=" for row in tests),
        "interpretation": "smoke/pilot 仅用于实现和开发诊断，不能宣称机制已获确认" if schedule["stage"] in ("smoke", "pilot")
                          else "按登记检验解释，报告正面、负面与不确定结果；不可将不显著当成等效",
        "secondary_metrics": "方向覆盖、决策多样性、GD+ 和解集大小只作描述；输出大小不等时需等基数补充分析",
        "performance_contribution_verified": None if evidence_scope(schedule["stage"]) == "confirmation" else False,
    }
    (output / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="例如 results/iemoeo_mechanisms_v2/pilot/attribution，兼容旧版登记")
    args = parser.parse_args()
    try:
        summary = summarize_study(args.study)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"汇总未完成：{exc}", file=sys.stderr)
        return 2
    print(f"已汇总 {summary['completed']} 个机制任务：{args.study / 'mechanism_summary'}")
    print(summary["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
