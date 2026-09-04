#!/usr/bin/env python
"""IEMOEC 统一实验入口（pymoo 0.6.2）。"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from iemoec_experiment.config import (  # noqa: E402
    ExperimentCase,
    IEMOECConfig,
    SUPPORTED_ALGORITHMS,
)
from iemoec_experiment.factory import reference_directions  # noqa: E402
from iemoec_experiment.runner import run_case  # noqa: E402


PRESETS = {
    "smoke": {
        "problems": ["dtlz2"],
        "objectives": [3],
        "seeds": [1],
        "evals_per_pop": 20,
    },
    "pilot": {
        "problems": ["dtlz1", "dtlz2", "dtlz3", "dtlz4"],
        "objectives": [3, 5, 10],
        "seeds": list(range(1, 6)),
        "evals_per_pop": 200,
    },
    "formal": {
        "problems": [
            "dtlz1", "dtlz2", "dtlz3", "dtlz4",
            "wfg1", "wfg2", "wfg4", "wfg9",
        ],
        "objectives": [3, 5, 8, 10, 15],
        "seeds": list(range(1, 31)),
        "evals_per_pop": 400,
    },
}


def parse_int_set(text: str) -> list[int]:
    result = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            if end < start:
                raise argparse.ArgumentTypeError(f"非法范围: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    if not result:
        raise argparse.ArgumentTypeError("集合不能为空")
    return sorted(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 pymoo 的 NSGA-II/III、MOEA/D 与 IEMOEC 公平对比实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=["smoke", "pilot", "formal", "custom"], default="smoke")
    parser.add_argument("--algorithms", nargs="+", choices=SUPPORTED_ALGORITHMS, default=list(SUPPORTED_ALGORITHMS))
    parser.add_argument("--problems", nargs="+", help="例如 dtlz2 wfg1")
    parser.add_argument("--objectives", type=parse_int_set, help="例如 3,5,8,10,15")
    parser.add_argument("--seeds", type=parse_int_set, help="例如 1-5 或 1,3,7")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-fes", type=int, help="显式共同 FE；必须是各 M 共同种群大小的倍数")
    budget.add_argument("--evals-per-pop", type=int, help="推荐：预算=参考方向数×该倍数")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--run-name", help="results 下的实验名；固定名称便于断点续跑")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--history-points", type=int, default=20)
    parser.add_argument("--history-hv", action="store_true", help="在检查点计算 HV（高维实验不建议）")
    parser.add_argument("--reference-points", type=int, default=1000)
    parser.add_argument("--high-dim-hv-samples", type=int, default=20000)
    parser.add_argument("--force", action="store_true", help="覆盖相同目录中的已有结果")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iemoec-survival", choices=["nsga3", "rank"], default="nsga3")
    parser.add_argument("--iemoec-crowding", action="store_true")
    parser.add_argument("--no-recombination", action="store_true")
    parser.add_argument("--retain-island-state", action="store_true")
    parser.add_argument("--fixed-island-definitions", action="store_true")
    parser.add_argument("--island-population", type=int, default=20)
    parser.add_argument("--inner-generations-early", type=int, default=1)
    parser.add_argument("--inner-generations-late", type=int, default=1)
    parser.add_argument("--partners-per-elite", type=int, default=2)
    return parser


def resolve_cases(args) -> list[ExperimentCase]:
    preset = PRESETS.get(args.preset, {})
    problems = args.problems or preset.get("problems")
    objectives = args.objectives or preset.get("objectives")
    seeds = args.seeds or preset.get("seeds")
    evals_per_pop = args.evals_per_pop or preset.get("evals_per_pop", 200)
    if not problems or not objectives or not seeds:
        raise ValueError("custom 模式必须提供 --problems、--objectives 和 --seeds")
    run_name = args.run_name or args.preset
    output_root = str(Path(args.output_root) / run_name)
    iemoec = IEMOECConfig(
        island_population=args.island_population,
        inner_generations_early=args.inner_generations_early,
        inner_generations_late=args.inner_generations_late,
        partners_per_elite=args.partners_per_elite,
        outer_survival=args.iemoec_survival,
        use_crowding=args.iemoec_crowding,
        enable_recombination=not args.no_recombination,
        retain_island_state=args.retain_island_state,
        fixed_island_definitions=args.fixed_island_definitions,
    )
    cases = []
    for problem in problems:
        for n_obj in objectives:
            probe = ExperimentCase("NSGA2", problem, n_obj, seeds[0], 1)
            pop_size = len(reference_directions(probe))
            max_fes = args.max_fes or (pop_size * evals_per_pop)
            if max_fes % pop_size:
                raise ValueError(
                    f"M={n_obj} 的共同种群大小是 {pop_size}，max_fes={max_fes} 不是其倍数"
                )
            for algorithm in args.algorithms:
                for seed in seeds:
                    cases.append(ExperimentCase(
                        algorithm=algorithm,
                        problem=problem,
                        n_obj=n_obj,
                        seed=seed,
                        max_fes=max_fes,
                        output_root=output_root,
                        history_points=args.history_points,
                        history_hv=args.history_hv,
                        reference_points=args.reference_points,
                        high_dim_hv_samples=args.high_dim_hv_samples,
                        iemoec=iemoec,
                    ))
    return cases


def main() -> int:
    args = build_parser().parse_args()
    try:
        cases = resolve_cases(args)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    print(f"实验任务: {len(cases)} | workers={args.workers}")
    for case in cases[:8]:
        print(
            f"  {case.normalized_algorithm:7s} {case.normalized_problem.upper():8s} "
            f"M={case.n_obj:2d} seed={case.seed:03d} MaxFEs={case.max_fes}"
        )
    if len(cases) > 8:
        print(f"  ... 其余 {len(cases) - 8} 个任务")
    if args.dry_run:
        return 0

    failures = []
    completed = skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_case, case, args.force): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            label = f"{case.normalized_problem.upper()} M{case.n_obj} {case.normalized_algorithm} seed={case.seed}"
            try:
                result = future.result()
                if result["status"] == "skipped":
                    skipped += 1
                    print(f"[跳过] {label}")
                else:
                    completed += 1
                    print(f"[完成] {label} FE={result['n_eval']} {result['runtime_seconds']:.2f}s")
            except Exception as exc:  # 单任务失败不能中断整个正式批次
                failures.append({"case": case.to_dict(), "error": repr(exc)})
                print(f"[失败] {label}: {exc}", file=sys.stderr)

    root = Path(cases[0].output_root)
    root.mkdir(parents=True, exist_ok=True)
    failure_path = root / "failures.json"
    with failure_path.open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    print(f"完成={completed} 跳过={skipped} 失败={len(failures)} | {root}")
    return 1 if failures else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
