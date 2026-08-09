#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取 IEMOEC 批次所有实验的最终指标（IGD/GD/HV/SP/ONVG）"""
import os, sys, re

BATCH = sys.argv[1]
HEADER = f"{'Problem':<30s} {'M':>3s} {'IGD':>10s} {'GD':>10s} {'HV':>10s} {'SP':>10s} {'ONVG':>6s}"
print(HEADER)
print("-" * 85)

for m_dir in sorted(os.listdir(BATCH)):
    m_path = os.path.join(BATCH, m_dir)
    if not os.path.isdir(m_path):
        continue
    M = m_dir.split("_M")[-1]
    for run_dir in sorted(os.listdir(m_path)):
        full = os.path.join(m_path, run_dir)
        log = os.path.join(full, "ga_log.txt")
        if not os.path.exists(log):
            continue
        with open(log) as f:
            content = f.read()
        # 直接正则提取最后一行 IGD/GD/HV/SP/ONVG
        matches = list(re.finditer(
            r'IGD:(\S+)\s+GD:(\S+)\s+HV[*\s]*:(\S+)\s+SP:(\S+)\s+ONVG:\s*(\S+)',
            content
        ))
        if matches:
            m = matches[-1]  # 最后一个匹配
            igd, gd, hv, sp, onvg = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        else:
            igd = gd = hv = sp = onvg = "-"
        print(f"{run_dir:<30s} {M:>3s} {igd:>10s} {gd:>10s} {hv:>10s} {sp:>10s} {onvg:>6s}")
