#!/usr/bin/env bash
# ============================================================
#  一键实验脚本 — 编辑下方配置，然后直接 ./script/run.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# ──────────── Python 路径 ────────────
if [ -f "D:/develop/Anaconda/python.exe" ]; then
    PYTHON="D:/develop/Anaconda/python.exe"
else
    PYTHON="python"
fi

# ============================================================
#  实验配置（改这里！）
# ============================================================

# 算法: NSGA2 / NSGA3 / MOEAD / IEMOEC
ALGO="IEMOEC"

# 问题类型: DTLZ / ZDT（IEMOEC 仅支持 DTLZ）
MODE="DTLZ"

# 目标维度列表（空格分隔）
# ┌────────┬──────────┬──────────────┬──────────────────────┐
# │   M    │ 参考点数 │ 岛内总评估数 │ 单 DTLZ 预计耗时     │
# ├────────┼──────────┼──────────────┼──────────────────────┤
# │   3    │    91    │   ~60,000    │  ~2-3 分钟           │
# │   5    │   210    │  ~100,000    │  ~5-8 分钟           │
# │   8    │   120    │  ~160,000    │  ~10-15 分钟          │
# │  10    │   220    │  ~220,000    │  ~20-30 分钟          │
# │  15    │   120    │  ~310,000    │  ~40-60 分钟          │
# └────────┴──────────┴──────────────┴──────────────────────┘
# 5 个 M × 5 个 DTLZ = 25 组，串行总耗时约 6-12 小时
# 建议分两批跑: M_LIST="3 5" 先跑(约30分钟), M_LIST="8 10 15" 后跑(约5-10小时)
M_LIST="3 5 8 10 15"

# 问题编号列表（空格分隔，留空 = 全跑 DTLZ1~9）
#   PROBLEM_IDS="2"          只跑 DTLZ2（快速验证，1 个问题）
#   PROBLEM_IDS="1 2 3 4"    论文标准对比集 DTLZ1~4
#   PROBLEM_IDS="1 2 3 4 10" 论文标准 + C-DTLZ2 凸PF
#   PROBLEM_IDS=""           全跑 DTLZ1~9
PROBLEM_IDS="1 2 3 4 10"

# ──────────── IE_MOEC 参数调优（可选）────────────
# 如需覆盖 config.py 中的默认值，修改 src/config.py 中的以下参数:
#   N_ORIGIN=None           起源种群大小（None=自动 POPSIZE）
#   N_ISLANDS=None          孤岛数量（None=自动 2×NOBJ，含 M 极端+M 折中）
#   ISLAND_POPSIZE=20       孤岛子种群大小
#   ISLAND_GENS_EARLY=30    岛内演化代数（聚合阶段）
#   ISLAND_GENS_LATE=70     岛内演化代数（Pareto 阶段）
#   MAX_OUTER_GENS=20       外循环最大代数
#   SWITCH_RATIO=0.4        聚合→Pareto 切换比例
#   ELITES_PER_ISLAND=3     每岛精英输出数
#   PF_EXPAND_RATIO=2.0     PF 扩展倍数

# ──────────── V-C 尺度缩放实验（论文 Section V-C）────────────
# SCALE_LIST: 空格分隔的缩放方案列表，留空 = 不启用
#   A = 无缩放（对照组）  B = f1=1 其余=10
#   C = 奇偶交替 1/10    D = 递增幂 1,1,10,100,...
# 示例:
#   SCALE_LIST="B"          → 只跑 Scale B
#   SCALE_LIST="A B C D"    → 四个方案全跑
#   SCALE_LIST=""           → 不启用缩放实验（默认标准模式）
SCALE_LIST=""

# ============================================================
#  执行（下面不用改）
# ============================================================

# 统一时间戳，同一批次所有 M 共享
BATCH_TIME=$(date +"%Y%m%d_%H%M%S")
BATCH_ROOT="output/${ALGO}/batch_${BATCH_TIME}"

echo "============================================"
echo "  算法: $ALGO"
echo "  问题: $MODE"
echo "  M 列表: $M_LIST"
if [ -n "$SCALE_LIST" ]; then
    echo "  缩放方案: $SCALE_LIST  (V-C Scaled DTLZ)"
fi
if [ -n "$PROBLEM_IDS" ]; then
    echo "  问题编号: $PROBLEM_IDS"
else
    echo "  范围: 批量全跑"
fi
echo "  批次目录: $BATCH_ROOT"
echo "============================================"
echo ""

for M in $M_LIST; do
    if [ -n "$SCALE_LIST" ]; then
        for SCALE in $SCALE_LIST; do
            echo ">>>>  M=${M}  $ALGO  Scale=${SCALE}  <<<<"
            if [ -n "$PROBLEM_IDS" ]; then
                for PID in $PROBLEM_IDS; do
                    echo "  --- ${MODE}${PID} ---"
                    "$PYTHON" script/run.py --mode "$MODE" --algo "$ALGO" --M "$M" \
                        --problem "$PID" --batch-root "$BATCH_ROOT" --scale "$SCALE"
                done
            else
                "$PYTHON" script/run.py --mode "$MODE" --algo "$ALGO" --M "$M" \
                    --batch-root "$BATCH_ROOT" --scale "$SCALE"
            fi
            echo ""
        done
    else
        echo ">>>>  M=${M}  $ALGO  <<<<"
        if [ -n "$PROBLEM_IDS" ]; then
            for PID in $PROBLEM_IDS; do
                echo "  --- ${MODE}${PID} ---"
                "$PYTHON" script/run.py --mode "$MODE" --algo "$ALGO" --M "$M" \
                    --problem "$PID" --batch-root "$BATCH_ROOT"
            done
        else
            "$PYTHON" script/run.py --mode "$MODE" --algo "$ALGO" --M "$M" \
                --batch-root "$BATCH_ROOT"
        fi
        echo ""
    fi
done

echo "============================================"
echo "  全部完成！"
echo "  结果: $BATCH_ROOT"
echo "============================================"
