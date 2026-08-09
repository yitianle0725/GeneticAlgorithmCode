# batch_summary.py
import os
import csv
import sys
import re
from path_tool import get_project_root

def extract_final_metrics(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"缺失日志文件: {log_file_path}")
        return None
    final_line = None
    with open(log_file_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        # 倒序找同时包含IGD和ONVG的行
        for line in reversed(all_lines):
            line_strip = line.strip()
            if "IGD:" in line_strip and "ONVG:" in line_strip:
                final_line = line_strip
                break
    if final_line is None:
        print(f"{log_file_path} 未找到完整指标行")
        return None

    metric_dict = {"IGD": "", "GD": "", "HV": "", "SP": "", "ONVG": ""}
    # 正则: 指标名 (可带 * 标记) : 数字
    # 日志格式: "HV*:0.1234" 或 "HV :0.1234"
    pattern = r"\b(IGD|GD|HV|SP|ONVG)\*?\s*:\s*([0-9.]+)"
    res_list = re.findall(pattern, final_line)
    for name, num_str in res_list:
        try:
            if name == "ONVG":
                metric_dict[name] = int(num_str)
            else:
                metric_dict[name] = float(num_str)
        except ValueError:
            metric_dict[name] = ""
    return metric_dict

def generate_batch_summary(batch_path, M=None, algo=None, scale_scheme=None):
    """从新目录结构读取全部 DTLZ 问题的最终指标，输出 CSV

    目录结构:
      {batch_path}/DTLZ_M{M}/DTLZ{id}_M{M}_{algo}[_Scale{x}]/ga_log.txt
    （batch_path 已含 algo 目录，如 output/NSGA3/batch_20260806_230000）
    """
    # 处理路径：batch_path 可以是绝对路径或相对 output/ 的文件夹名
    if not os.path.isabs(batch_path) and not batch_path.startswith("output"):
        root = get_project_root()
        batch_path = os.path.join(root, "output", batch_path)
    if not os.path.isdir(batch_path):
        print(f"批次目录不存在: {batch_path}")
        return

    # 单独运行脚本时可手动输入参数
    if M is None or algo is None:
        if len(sys.argv) >= 4:
            M = int(sys.argv[2])
            algo = sys.argv[3]
            print(f"手动传入参数 M={M}, algo={algo}")
        else:
            print("使用说明：")
            print("1. run.sh 批量运行自动传参，无需操作；")
            print("2. 单独执行：python batch_summary.py batch_20260806_225600 5 NSGA3")
            return

    scale_tag = f"_Scale{scale_scheme}" if (scale_scheme and scale_scheme != "A") else ""
    table_data = []
    for dtlz_id in range(1, 11):
        sub_folder = f"DTLZ{dtlz_id}_M{M}_{algo}{scale_tag}"
        log_path = os.path.join(batch_path, f"DTLZ_M{M}", sub_folder, "ga_log.txt")
        res = extract_final_metrics(log_path)
        if res is None:
            row = [dtlz_id, "", "", "", "", ""]
        else:
            row = [
                dtlz_id,
                res["IGD"],
                res["GD"],
                res["HV"],
                res["SP"],
                res["ONVG"]
            ]
        table_data.append(row)

    csv_dir = os.path.join(batch_path, f"DTLZ_M{M}")
    os.makedirs(csv_dir, exist_ok=True)
    csv_out = os.path.join(csv_dir, "batch_summary.csv")
    header = ["DTLZ_ID", "IGD", "GD", "HV", "SP", "ONVG"]
    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table_data)
    print(f"\n[OK] 汇总表已输出：{csv_out}")
    print("-" * 70)
    print(f"当前批次配置：M={M}，算法={algo}")
    print(f"{'DTLZ':<6}{'IGD':<12}{'GD':<12}{'HV':<10}{'SP':<10}{'ONVG':<6}")
    for row in table_data:
        d = row[0]
        igd = f"{row[1]:.6f}" if isinstance(row[1], float) else "-"
        gd = f"{row[2]:.6f}" if isinstance(row[2], float) else "-"
        hv = f"{row[3]:.4f}" if isinstance(row[3], float) else "-"
        sp = f"{row[4]:.4f}" if isinstance(row[4], float) else "-"
        onvg = row[5] if isinstance(row[5], int) else "-"
        print(f"{d:<6}{igd:<12}{gd:<12}{hv:<10}{sp:<10}{onvg:<6}")

if __name__ == "__main__":
    # 单独读取历史批次直接在这里改参数
    # 格式: batch_{ts} M algo
    batch_name = "batch_20260806_225600"
    generate_batch_summary(batch_name, 3, "NSGA3")