#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键实验启动脚本 — NSGA-II / NSGA-III / MOEA/D 多目标优化实验

用法:
  # 交互模式（推荐新手）
  python script/run.py

  # 命令行模式
  python script/run.py --mode DTLZ --algo NSGA3 --M 8
  python script/run.py --mode DTLZ --algo MOEAD --M 5
  python script/run.py --mode DTLZ --algo NSGA3 --M 3 --problem 2
  python script/run.py --mode ZDT  --algo NSGA2 --problem 4
  python script/run.py --mode ZDT  --algo MOEAD

支持的问题类型:
  DTLZ — DTLZ1~10，M 目标 (3/5/8/10/15)
  ZDT  — ZDT1~6，2 目标

支持的算法:
  NSGA2 — 非支配排序遗传算法 II (拥挤度距离)
  NSGA3 — 非支配排序遗传算法 III (参考向量小生境)
  MOEAD — 基于分解的多目标进化算法 (Tchebycheff)
"""

import sys
import os
import argparse
import random
import datetime
import numpy as np

# 确保 src 目录在 path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from config import GAConfig
from moec_algorithm import MultiObjMOECAbilene
from batch_summary import generate_batch_summary


# ──────────────────────────── 外观 ────────────────────────────
HEADER = """
╔══════════════════════════════════════════════════════╗
║   NSGA-II / NSGA-III / MOEA/D  多目标优化实验平台    ║
╚══════════════════════════════════════════════════════╝"""

MODE_INFO = {
    "DTLZ": {"desc": "DTLZ 标准测试集 (1~10)", "nobj": "3/5/8/10/15"},
    "ZDT":  {"desc": "ZDT 双目标测试集 (1~6)", "nobj": "2 (固定)"},
}

ALGO_INFO = {
    "NSGA3": "参考向量小生境 (推荐 >=3 目标)",
    "NSGA2": "拥挤度距离 (推荐 2 目标)",
    "MOEAD": "Tchebycheff 分解 + 邻域交配 (论文对比基线)",
}


# ──────────────────────────── 核心执行 ────────────────────────────
def run_single(cfg: GAConfig, problem_type: str, problem_id: int,
               algo: str, batch_root: str, scale_scheme: str = None) -> None:
    """执行单次实验"""
    random.seed(123)
    np.random.seed(123)

    if problem_type == "DTLZ":
        ga = MultiObjMOECAbilene(
            dt_id=problem_id, dtlz_M=cfg.NOBJ, algo_type=algo,
            root_output_dir=batch_root, scale_scheme=scale_scheme
        )
    elif problem_type == "ZDT":
        ga = MultiObjMOECAbilene(
            zdt_id=problem_id, algo_type=algo,
            root_output_dir=batch_root
        )
    else:
        raise ValueError(f"未知问题类型: {problem_type}")

    ga.run()


def run_batch(problem_type: str, algo: str, nobj: int,
              single_problem: int = None,
              shared_batch_root: str = None,
              scale_scheme: str = None) -> None:
    """批量实验入口

    目录结构:
      output/{algo}/batch_{ts}/
        {problem_type}_M{nobj}/
          {problem_type}{pid}_M{nobj}_{algo}[_Scale{x}]/
    """
    # 顶层批次目录（可由 run.sh 统一传入，保证同一批实验在同一时间戳下）
    if shared_batch_root:
        batch_root = shared_batch_root
    else:
        batch_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir_name = f"batch_{batch_time}"
        batch_root = os.path.join(PROJECT_ROOT, "output", algo, batch_dir_name)

    # M 级目录: output/{algo}/batch_ts/DTLZ_M5/
    # batch_root 已含 algo，此处不再重复添加
    m_root = os.path.join(batch_root, f"{problem_type}_M{nobj}")
    os.makedirs(m_root, exist_ok=True)

    cfg = GAConfig(nobj=nobj, scale_scheme=scale_scheme)

    scale_tag = f" Scale={scale_scheme}" if scale_scheme and scale_scheme != "A" else ""
    print(f"\n{'='*60}")
    print(f"  实验配置: {problem_type} | {algo} | M={nobj}{scale_tag}")
    print(f"  种群={cfg.POPSIZE} | 代数={cfg.MAXGENS} | "
          f"参考点划分={cfg.REF_DIV}")
    print(f"  输出目录: {m_root}")
    print(f"{'='*60}\n")

    if problem_type == "DTLZ":
        problem_ids = [single_problem] if single_problem else list(range(1, 11))
    elif problem_type == "ZDT":
        problem_ids = [single_problem] if single_problem else list(range(1, 7))
    else:
        raise ValueError(f"未知问题类型: {problem_type}")

    for pid in problem_ids:
        label = f"{problem_type}{pid}"
        print(f"{'─'*50}")
        print(f"  >>> {label}_{algo}  开始")
        print(f"{'─'*50}")
        run_single(cfg, problem_type, pid, algo, batch_root=m_root, scale_scheme=scale_scheme)
        print(f"  <<< {label}_{algo}  完成\n")

    # 汇总
    print(f"\n{'='*60}")
    print("  全部实验完成，生成汇总…")
    if problem_type == "DTLZ":
        generate_batch_summary(batch_root, nobj, algo, scale_scheme=scale_scheme)
    print(f"  结果保存在: {batch_root}")
    print(f"{'='*60}\n")


# ──────────────────────────── 交互模式 ────────────────────────────
def interactive_mode():
    """逐步引导式交互"""
    print(HEADER)

    # Step 1: 问题类型
    print("\n【问题类型】")
    mode_keys = list(MODE_INFO.keys())
    for i, key in enumerate(mode_keys, 1):
        info = MODE_INFO[key]
        print(f"  {i}. {key:<6} — {info['desc']}  (目标维度: {info['nobj']})")
    while True:
        try:
            choice = input("\n请选择 (1-2, 默认 1=DTLZ): ").strip()
            if choice == "":
                choice = "1"
            idx = int(choice) - 1
            if 0 <= idx < len(mode_keys):
                problem_type = mode_keys[idx]
                break
            print("  ⚠ 输入无效，请重新选择")
        except ValueError:
            print("  ⚠ 请输入数字")

    # Step 2: 算法
    print("\n【算法选择】")
    algo_keys = list(ALGO_INFO.keys())
    for i, key in enumerate(algo_keys, 1):
        print(f"  {i}. {key:<6} — {ALGO_INFO[key]}")
    while True:
        try:
            choice = input("\n请选择 (1-3, 默认 1=NSGA3): ").strip()
            if choice == "":
                choice = "1"
            idx = int(choice) - 1
            if 0 <= idx < len(algo_keys):
                algo = algo_keys[idx]
                break
            print("  ⚠ 输入无效，请重新选择")
        except ValueError:
            print("  ⚠ 请输入数字")

    # Step 3: 目标维度 M (仅 DTLZ)
    nobj = None
    if problem_type == "DTLZ":
        print("\n【目标维度 M】")
        print("  1. M=3    2. M=5    3. M=8    4. M=10    5. M=15")
        while True:
            choice = input("请选择 (1-5, 默认 3=M=8): ").strip()
            if choice == "":
                nobj = 8
                break
            try:
                m_map = {1: 3, 2: 5, 3: 8, 4: 10, 5: 15}
                nobj = m_map.get(int(choice))
                if nobj:
                    break
                print("  ⚠ 输入无效")
            except ValueError:
                print("  ⚠ 请输入数字")
    elif problem_type == "ZDT":
        nobj = 2

    # Step 4: 单问题 or 批量
    single_problem = None
    if problem_type == "DTLZ":
        print("\n【运行范围】")
        print("  1. 批量运行全部 DTLZ1~10")
        print("  2. DTLZ1  3. DTLZ2  ...  10. DTLZ9  11. C-DTLZ2")
        choice = input("请选择 (默认 1=全跑): ").strip()
        if choice and choice != "1":
            try:
                pid = int(choice) - 1
                if 1 <= pid <= 10:
                    single_problem = pid
            except ValueError:
                pass
    elif problem_type == "ZDT":
        print("\n【运行范围】")
        print("  1. 批量运行全部 ZDT1~6")
        for i in range(1, 7):
            print(f"  {i+1}. 只跑 ZDT{i}")
        choice = input("请选择 (默认 1=全跑): ").strip()
        if choice and choice != "1":
            try:
                pid = int(choice) - 1
                if 1 <= pid <= 6:
                    single_problem = pid
            except ValueError:
                pass

    # Step 5: 确认
    label = f"{problem_type}{single_problem}" if single_problem else problem_type
    print(f"\n{'─'*50}")
    print(f"  确认: {label} | {algo} | M={nobj}")
    confirm = input("  开始执行? (Y/n): ").strip().lower()
    if confirm and confirm != "y":
        print("  已取消")
        return

    run_batch(problem_type, algo, nobj, single_problem)


# ──────────────────────────── CLI 入口 ────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NSGA-II/III/MOEAD 多目标优化实验一键启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python script/run.py --mode DTLZ --algo NSGA3 --M 8
  python script/run.py --mode DTLZ --algo MOEAD --M 5
  python script/run.py --mode DTLZ --algo NSGA3 --M 3 --problem 2
  python script/run.py --mode ZDT  --algo NSGA2 --problem 4
  python script/run.py -i          # 交互模式
        """
    )
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="交互模式（逐步引导选择参数）")
    parser.add_argument("--mode", type=str, choices=["DTLZ", "ZDT"],
                        default="DTLZ", help="问题类型 (默认: DTLZ)")
    parser.add_argument("--algo", type=str, choices=["NSGA2", "NSGA3", "MOEAD"],
                        default="NSGA3", help="算法 (默认: NSGA3)")
    parser.add_argument("--M", type=int, choices=[3, 5, 8, 10, 15],
                        help="目标维度 (仅 DTLZ 需要，默认: 8)")
    parser.add_argument("--problem", type=int,
                        help="问题编号 (仅跑单个问题，不传则批量全跑)")
    parser.add_argument("--batch-root", type=str, default=None,
                        help="父级批次目录（run.sh 统一传入，保证同时间戳）")
    parser.add_argument("--scale", type=str, choices=["A", "B", "C", "D"], default=None,
                        help="V-C 尺度缩放方案 (A=无缩放/对照, B=f1=1其余10, C=奇偶交替, D=递增幂)")
    parser.add_argument("--seed", type=int, default=123,
                        help="随机种子 (默认: 123)")

    args = parser.parse_args()

    # 无参数 / -i → 交互模式
    if args.interactive or len(sys.argv) == 1:
        interactive_mode()
        return

    # CLI 模式参数校验
    if args.mode == "DTLZ":
        nobj = args.M if args.M else 8
    else:  # ZDT
        nobj = 2

    single_problem = None
    if args.problem is not None:
        if args.mode == "DTLZ" and not (1 <= args.problem <= 10):
            print(f"⚠ DTLZ 问题编号范围: 1~10，收到 {args.problem}")
            sys.exit(1)
        if args.mode == "ZDT" and not (1 <= args.problem <= 6):
            print(f"⚠ ZDT 问题编号范围: 1~6，收到 {args.problem}")
            sys.exit(1)
        single_problem = args.problem

    run_batch(args.mode, args.algo, nobj, single_problem,
              shared_batch_root=args.batch_root, scale_scheme=args.scale)


if __name__ == "__main__":
    main()
