"""生成或执行已登记的 IE-MOEO 机制实验，复用 scripts/run.py 的配置和执行核心。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run import build_parser, resolve_cases
from iemoec_experiment.result_store import implementation_signature, stored_result_complete, case_implementation_signature

DEFAULT_PLAN = ROOT / "experiments" / "iemoeo_mechanisms_v3.json"


def build_commands(plan: dict, stage: str, suite: str, output_root: Path,
                   result_store: Path, workers: int) -> list[dict]:
    configuration = plan["stages"][stage]
    if configuration["timing_only"] and workers != 1:
        raise ValueError("干净计时必须使用 --workers 1")
    commands = []
    common_arguments = list(plan.get("common_arguments", []))
    if configuration["timing_only"]:
        common_arguments = [argument for argument in common_arguments if argument != "--iemoeo-selection-audit"]
    for name in plan["suites"][suite]:
        variant = plan["variants"][name]
        algorithm = variant.get("algorithm", "IEMOEO")
        arguments = [
            "--preset", "custom", "--algorithms", algorithm,
            "--problems", *configuration["problems"],
            "--objectives", configuration["objectives"], "--seeds", configuration["seeds"],
            "--evals-per-pop", str(configuration["evals_per_pop"]),
            "--output-root", str(output_root / stage / suite),
            "--result-store", str(result_store), "--run-name", name,
            "--workers", str(workers), *common_arguments, *variant["arguments"],
        ]
        if configuration["timing_only"]:
            arguments.append("--timing-only")
        # 同一解析器验证 flags、预算和配置，避免另写一套实验业务逻辑。
        cases = resolve_cases(build_parser().parse_args(arguments))
        case_template = cases[0].to_dict()
        case_template.pop("output_root")
        commands.append({
            "variant": name, "label": variant["label"], "arguments": arguments,
            "tasks": len(cases), "total_fes": sum(case.max_fes for case in cases),
            f"{'iemoeo' if algorithm == 'IEMOEO' else 'iemoec'}_config": cases[0].to_dict()["iemoeo" if algorithm == "IEMOEO" else "iemoec"],
            "case_template": case_template,
            "implementation": case_implementation_signature(cases[0]),
        })
    return commands


def write_schedule(path: Path, plan: dict, stage: str, suite: str, commands: list[dict]) -> None:
    # worker 数可以调整；科学配置和源码在同一登记目录中不可悄悄改变。
    registration = {
        "plan": plan, "stage": stage, "suite": suite,
        "implementation": commands[0].get("implementation", implementation_signature("IEMOEO")),
        "resolved_configurations": [{key: command[key] for key in (
            "variant", "tasks", "total_fes", "case_template",
        )} for command in commands],
    }
    if any(command.get("implementation", registration["implementation"]) != registration["implementation"] for command in commands):
        registration["arm_implementations"] = {command["variant"]: command["implementation"] for command in commands}
    encoded = json.dumps(registration, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["registration_sha256"] != digest:
            raise ValueError("方案或实现已经改变，请使用新的 --output-root 保留本次冻结记录")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"registration_sha256": digest, **registration, "commands": commands,
               "total_tasks": sum(command["tasks"] for command in commands),
               "total_fes": sum(command["total_fes"] for command in commands)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def incomplete_commands(commands: list[dict]) -> list[dict]:
    """只筛执行任务，不修改完整冻结清单、参数或实验组。"""
    pending = []
    for command in commands:
        cases = resolve_cases(build_parser().parse_args(command["arguments"]))
        if any(not stored_result_complete(case) for case in cases):
            pending.append(command)
    return pending


def main(default_plan: Path = DEFAULT_PLAN, default_suite: str = "core") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=default_plan)
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirm", "many_objective", "timing"], default="pilot")
    parser.add_argument("--suite", default=default_suite, help="方案中的套件名，例如 core/parents/blocks/attribution")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--result-store", type=Path, default=ROOT / "results" / "algorithms")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--execute", action="store_true", help="实际执行；省略时仅登记和显示清单")
    parser.add_argument("--retry-incomplete", action="store_true", help="只续跑未完成机制组中的未完成任务，保留原冻结清单")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.suite not in plan["suites"]:
        parser.error(f"该方案的 suite 可选：{', '.join(plan['suites'])}")
    args.output_root = args.output_root or ROOT / "results" / plan["name"]
    workers = args.workers if args.workers is not None else (1 if args.stage == "timing" else 4)
    if workers < 1:
        parser.error("workers 必须至少为 1")
    try:
        commands = build_commands(plan, args.stage, args.suite, args.output_root, args.result_store, workers)
        schedule = args.output_root / args.stage / args.suite / "schedule.json"
        write_schedule(schedule, plan, args.stage, args.suite, commands)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"阶段={args.stage} 套件={args.suite} 任务={sum(c['tasks'] for c in commands)} "
          f"FE={sum(c['total_fes'] for c in commands)} workers={workers}", flush=True)
    for command in commands:
        print(f"  {command['variant']}: {command['tasks']} 项 | {command['label']}", flush=True)
    print(f"冻结清单：{schedule}", flush=True)
    if args.retry_incomplete:
        all_count = len(commands)
        commands = incomplete_commands(commands)
        print(f"续跑机制组：{len(commands)}/{all_count}，已完成组直接复用", flush=True)
    if not args.execute:
        print("已登记，尚未启动计算。增加 --execute 执行此阶段。")
        return 0
    failed = []
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] 开始机制组 {command['variant']}", flush=True)
        retry_arguments = ["--only-incomplete"] if args.retry_incomplete else []
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"),
                                 *command["arguments"], *retry_arguments])
        if result.returncode:
            failed.append(command["variant"])
    print(f"机制组完成：{len(commands) - len(failed)}/{len(commands)}；失败组：{failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
