# compare_algorithms.py
"""
算法对比模块 — 读取 output/ 下所有 batch_summary.csv，按 M 分别生成对比表。

用法:
  # 自动对比所有已有结果
  python src/compare_algorithms.py

  # 指定对比某几个 M
  python src/compare_algorithms.py --M 3 5 8

  # 只输出 CSV，不打印终端表格
  python src/compare_algorithms.py --csv-only

输出:
  output/comparison/DTLZ_M3_comparison.csv   ← 每个 M 一张对比表
  output/comparison/DTLZ_M5_comparison.csv
  ...
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

# 确保 src 在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tool import get_project_root


# ──────────────────────────── 数据收集 ────────────────────────────
def _latest_batch(algo_dir: str) -> str | None:
    """返回 algo 目录下最新的 batch 子目录名"""
    batches = sorted(
        [d for d in os.listdir(algo_dir)
         if d.startswith("batch_") and os.path.isdir(os.path.join(algo_dir, d))],
        reverse=True
    )
    return batches[0] if batches else None


def collect_all_results(output_root: str = None) -> dict:
    """扫描 output/ 目录，返回嵌套字典:
    {
        M: {
            algo: {
                dtlz_id: {"IGD": ..., "GD": ..., "HV": ..., "SP": ..., "ONVG": ...}
            }
        }
    }

    每个 algo 只取最新的 batch。
    """
    if output_root is None:
        output_root = os.path.join(get_project_root(), "output")

    results: dict[int, dict[str, dict[int, dict[str, float | int]]]] = defaultdict(dict)

    for algo in os.listdir(output_root):
        algo_path = os.path.join(output_root, algo)
        if not os.path.isdir(algo_path):
            continue

        batch_name = _latest_batch(algo_path)
        if batch_name is None:
            continue
        batch_path = os.path.join(algo_path, batch_name)

        for item in os.listdir(batch_path):
            if not item.startswith("DTLZ_M"):
                continue
            csv_path = os.path.join(batch_path, item, "batch_summary.csv")
            if not os.path.isfile(csv_path):
                continue

            # 解析 M 值: "DTLZ_M5" → 5
            try:
                M = int(item.split("_M")[1])
            except (IndexError, ValueError):
                continue

            # 读取 CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dtlz_id = int(row.get("DTLZ_ID", 0))
                    # 跳过空行
                    igd_raw = row.get("IGD", "").strip()
                    if not igd_raw:
                        continue
                    results[M].setdefault(algo, {})[dtlz_id] = {
                        "IGD": float(igd_raw),
                        "GD": float(row["GD"]),
                        "HV": float(row["HV"]),
                        "SP": float(row["SP"]),
                        "ONVG": int(row["ONVG"]),
                    }

    return dict(results)


# ──────────────────────────── 终端输出 ────────────────────────────
def _best_marker(values: dict[str, float], key=lambda v: v) -> str:
    """返回每个 algo 的值+标记字符串，最优值加 '*'"""
    best = key(min(values.values(), key=key))  # 找最小 key 的结果
    lines = []
    for algo, val in values.items():
        mark = " *" if key(val) == best else ""
        if isinstance(val, float):
            if abs(val) < 1e-6:
                s = f"{val:.4f}{mark}"
            else:
                s = f"{val:.6f}{mark}"
        else:
            s = f"{val}{mark}"
        lines.append(s)
    return lines


def print_comparison(results: dict, M_list: list[int] = None):
    """终端打印所有 M 的对比表"""
    if M_list is None:
        M_list = sorted(results.keys())

    METRICS = ["IGD", "GD", "HV", "SP", "ONVG"]
    metric_down = {"IGD": True, "GD": True, "HV": False, "SP": True, "ONVG": False}
    # ↓ = 越小越好, ↑ = 越大越好
    metric_arrow = {"IGD": "↓", "GD": "↓", "HV": "↑", "SP": "↓", "ONVG": "↑"}

    for M in M_list:
        if M not in results:
            continue
        data = results[M]
        algos = sorted(data.keys())
        dtlz_ids = sorted(set().union(*(data[a].keys() for a in algos)))

        print(f"\n{'='*80}")
        print(f"  M = {M}  算法对比")
        print(f"{'='*80}")

        for pid in dtlz_ids:
            print(f"\n  ── DTLZ{pid} ──")
            header = f"  {'指标':<8}"
            for a in algos:
                header += f"{a:<22}"
            print(header)
            print(f"  {'-'*8}{'-'*22*len(algos)}")

            for metric in METRICS:
                vals = {}
                for a in algos:
                    if pid in data[a]:
                        vals[a] = data[a][pid][metric]
                if not vals:
                    continue

                arrow = metric_arrow[metric]
                row = f"  {metric+arrow:<8}"

                if metric_down[metric]:  # 越小越好
                    best = min(vals.values())
                else:  # 越大越好
                    best = max(vals.values())

                for a in algos:
                    v = vals.get(a)
                    if v is None:
                        row += f"{'-':<22}"
                    else:
                        mark = " *" if v == best else ""
                        if metric == "ONVG":
                            row += f"{v}{mark:<21}"
                        elif abs(v) < 1e-6:
                            row += f"{v:.4f}{mark:<17}"
                        else:
                            row += f"{v:.6f}{mark:<16}"
                print(row)

        # 汇总：每个算法在该 M 下的平均 IGD
        print(f"\n  ── 汇总 (平均 IGD ↓) ──")
        summary = f"  {'':<8}"
        for a in algos:
            igds = [data[a][pid]["IGD"] for pid in dtlz_ids if pid in data[a] and "IGD" in data[a][pid]]
            avg = sum(igds) / len(igds) if igds else float("nan")
            best_avg = min(
                (sum(data[aa][pid]["IGD"] for pid in dtlz_ids if pid in data[aa] and "IGD" in data[aa][pid])
                 / max(1, len([p for p in dtlz_ids if p in data[aa] and "IGD" in data[aa][p]])))
                for aa in algos
            )
            mark = " *" if avg == best_avg else ""
            summary += f"{avg:.6f}{mark:<16}"
        print(summary)

    print(f"\n{'='*80}")
    print("  * = 最优值   ↓ = 越小越好   ↑ = 越大越好")
    print(f"{'='*80}\n")


# ──────────────────────────── CSV 导出 ────────────────────────────
def export_comparison_csv(results: dict, output_dir: str = None, M_list: list[int] = None):
    """每个 M 导出一张对比 CSV，同时生成一张汇总 CSV"""
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "output", "comparison")
    os.makedirs(output_dir, exist_ok=True)

    if M_list is None:
        M_list = sorted(results.keys())

    METRICS = ["IGD", "GD", "HV", "SP", "ONVG"]

    # ── 单 M 表 ──
    for M in M_list:
        if M not in results:
            continue
        data = results[M]
        algos = sorted(data.keys())
        dtlz_ids = sorted(set().union(*(data[a].keys() for a in algos)))

        csv_path = os.path.join(output_dir, f"DTLZ_M{M}_comparison.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # 双层表头: 指标 + 算法
            header1 = ["DTLZ_ID"]
            for metric in METRICS:
                for a in algos:
                    header1.append(f"{metric}_{a}")
            writer.writerow(header1)

            for pid in dtlz_ids:
                row = [pid]
                for metric in METRICS:
                    for a in algos:
                        if pid in data[a] and metric in data[a][pid]:
                            row.append(data[a][pid][metric])
                        else:
                            row.append("")
                writer.writerow(row)

        print(f"[OK] {csv_path}")

    # ── 总汇总表: 每个 (M, DTLZ_ID, Metric) 一行 ──
    summary_path = os.path.join(output_dir, "all_comparison.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "DTLZ_ID", "Metric", "Algo", "Value"])
        for M in sorted(M_list):
            if M not in results:
                continue
            data = results[M]
            algos = sorted(data.keys())
            dtlz_ids = sorted(set().union(*(data[a].keys() for a in algos)))
            for pid in dtlz_ids:
                for metric in METRICS:
                    for a in algos:
                        if pid in data[a] and metric in data[a][pid]:
                            writer.writerow([M, pid, metric, a, data[a][pid][metric]])

    print(f"[OK] {summary_path}")


# ──────────────────────────── 入口 ────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="多算法实验结果对比工具"
    )
    parser.add_argument("--M", type=int, nargs="+",
                        help="指定对比的 M 值，不传则自动检测所有")
    parser.add_argument("--csv-only", action="store_true",
                        help="只导出 CSV，不打印终端表格")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="CSV 输出目录 (默认 output/comparison/)")
    args = parser.parse_args()

    print("扫描实验数据...")
    results = collect_all_results()

    if not results:
        print("未找到任何 batch_summary.csv，请先跑实验。")
        return

    M_list = args.M if args.M else sorted(results.keys())
    print(f"找到 M = {M_list}")

    algos = set()
    for M in M_list:
        if M in results:
            algos.update(results[M].keys())
    print(f"算法: {sorted(algos)}")

    if not args.csv_only:
        print_comparison(results, M_list)

    export_comparison_csv(results, args.output_dir, M_list)


if __name__ == "__main__":
    main()
