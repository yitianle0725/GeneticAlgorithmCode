#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 4 个算法的 IGD 指标"""
import os, sys, re, glob
from collections import defaultdict

BATCHES = {
    "NSGA2":  "output/NSGA2/batch_20260809_162311",
    "NSGA3":  "output/NSGA3/batch_20260809_162320",
    "MOEAD":  "output/MOEAD/batch_20260809_162340",
    "IEMOEC": "output/IEMOEC/batch_20260809_201220",
}

def extract_metrics(batch_root):
    """返回 {(M, problem): {'igd': ..., 'gd': ..., 'hv': ..., 'sp': ..., 'onvg': ...}}"""
    results = {}
    if not os.path.isdir(batch_root):
        return results
    for m_dir in sorted(os.listdir(batch_root)):
        m_path = os.path.join(batch_root, m_dir)
        if not os.path.isdir(m_path):
            continue
        M = int(m_dir.split("_M")[-1])
        for run_dir in sorted(os.listdir(m_path)):
            log = os.path.join(m_path, run_dir, "ga_log.txt")
            if not os.path.exists(log):
                continue
            # 从目录名提取 problem: DTLZ1_M3_IEMOEC → 1
            prob = int(run_dir.split("_")[0].replace("DTLZ", ""))
            with open(log) as f:
                content = f.read()
            m = list(re.finditer(
                r'IGD:(\S+)\s+GD:(\S+)\s+HV[*\s]*:(\S+)\s+SP:(\S+)\s+ONVG:\s*(\S+)',
                content
            ))
            if m:
                last = m[-1]
                results[(M, prob)] = {
                    'igd': float(last.group(1)),
                    'gd': float(last.group(2)),
                    'hv': float(last.group(3)),
                    'sp': float(last.group(4)),
                    'onvg': int(last.group(5)),
                }
    return results

# 收集所有算法的数据
all_data = {}
for algo, batch in BATCHES.items():
    all_data[algo] = extract_metrics(batch)

# 按 DTLZ 问题分组打印对比表
PROBLEMS = [1, 2, 3, 4, 10]
M_LIST = [3, 5, 8, 10, 15]
PROBLEM_NAMES = {1: "DTLZ1 (线性)", 2: "DTLZ2 (凹面)", 3: "DTLZ3 (多峰)", 4: "DTLZ4 (偏置)", 10: "C-DTLZ2 (凸面)"}

ALGOS = ["NSGA2", "NSGA3", "MOEAD", "IEMOEC"]

for prob in PROBLEMS:
    print(f"\n{'='*100}")
    print(f"  {PROBLEM_NAMES[prob]}")
    print(f"{'='*100}")
    header = f"{'M':>4s}"
    for algo in ALGOS:
        header += f"  {algo:>12s}"
    print(header)
    print("-" * 100)

    for M in M_LIST:
        key = (M, prob)
        row = f"{M:4d}"
        best_igd = float('inf')
        best_algo = None
        igds = {}
        for algo in ALGOS:
            if algo in all_data and key in all_data[algo]:
                igd = all_data[algo][key]['igd']
                igds[algo] = igd
                if igd < best_igd:
                    best_igd = igd
                    best_algo = algo
            else:
                igds[algo] = None

        for algo in ALGOS:
            if igds[algo] is not None:
                marker = " *" if algo == best_algo else "  "
                row += f"  {igds[algo]:>10.4f}{marker}"
            else:
                row += f"  {'-':>12s}"
        print(row)
    print(f"  (* = 最优 IGD)")

# 汇总：各算法获胜次数
print(f"\n{'='*100}")
print("  各算法最优 IGD 次数统计")
print(f"{'='*100}")
win_count = defaultdict(int)
for prob in PROBLEMS:
    for M in M_LIST:
        key = (M, prob)
        best_igd = float('inf')
        best_algo = None
        for algo in ALGOS:
            if algo in all_data and key in all_data[algo]:
                igd = all_data[algo][key]['igd']
                if igd < best_igd:
                    best_igd = igd
                    best_algo = algo
        if best_algo:
            win_count[best_algo] += 1
            print(f"  DTLZ{prob}_M{M}: {best_algo} (IGD={best_igd:.4f})")

print(f"\n  总计 (25 组):")
for algo in ALGOS:
    print(f"    {algo}: {win_count[algo]} 胜")
