# IEMOEC 实验平台

本项目已重构为基于 **pymoo 0.6.2** 的可复现多目标/多目标数优化实验平台。

## 设计原则

- NSGA-II、NSGA-III、MOEA/D-TCH、RVEA、AGE-MOEA2、DTLZ、WFG、SBX、多项式变异、非支配排序和质量指标直接使用 pymoo。
- 四个算法在相同问题与目标数下共享种群规模，并严格使用相同 `MaxFEs`。
- 每个 problem、目标数和 seed 使用同一份缓存 `X_init`；candidate 会先评价完整 N，再构造 origin。
- 正式实验不使用 HV/IGD 早停，不提供 IEMOEC 专属的额外 PF 扩展。
- 低维（M≤5）计算精确 HV；高维使用固定公共采样点的 Monte Carlo HV。
- 运行时只在固定 FE 检查点记录历史；默认仅最终计算 HV。
- 每个独立任务可并行运行、自动跳过已完成配置、记录失败任务并单独重跑。

## 环境

```powershell
conda activate moo
python -c "import pymoo; print(pymoo.__version__)"
```

所需包记录在 `requirement.txt`。当前代码面向 `pymoo>=0.6.2,<0.7`。

## 实验层级

### 1. 正确性冒烟测试

DTLZ2、M=3、1 个种子、4 个算法，预算为 20 倍共同种群大小：

```powershell
python scripts/run.py --preset smoke --workers 4
```

### 2. 预实验

DTLZ1–4、M=3/5/10、5 个种子：

```powershell
python scripts/run.py --preset pilot --workers 6
```

### 3. 结构实验清单

DTLZ2/3/4/7、WFG1/2/4/9，M=3/5/10，5 个种子。首先只审核任务：

```powershell
python scripts/run.py --preset structure --iemoec-variant candidate --dry-run
```

### 4. 正式实验

DTLZ1–4、WFG1/2/4/9、M=3/5/8/10/15、30 个种子：

```powershell
python scripts/run.py --preset formal --workers 6
```

正式实验任务很多。建议先完成 smoke 和 pilot，确认参数后再启动。仅使用任务级并行，不要同时开启岛级多进程。

## 自定义实验

推荐用 `--evals-per-pop` 定义预算，它会自动得到各目标数下与共同种群规模整除的 `MaxFEs`：

```powershell
python scripts/run.py --preset custom `
  --algorithms NSGA2 NSGA3 MOEAD IEMOEC `
  --problems dtlz2 wfg1 `
  --objectives 3,5,10 `
  --seeds 1-5 `
  --evals-per-pop 200 `
  --workers 6 `
  --run-name my_pilot
```

也可显式传入 `--max-fes`，但它必须是每个目标数对应共同种群大小的整数倍。这样 pymoo 的代际 NSGA-II/III 与 MOEA/D 都能精确停在同一 FE。

常用参数：

- `--dry-run`：只展示任务，不执行。
- `--force`：重新运行完全相同的配置；任何配置差异都必须更换 `--run-name`。
- `--history-hv`：在历史检查点计算 HV；高维时不建议启用。
- `--run-name`：固定结果批次名，用于断点续跑。
- `--iemoec-variant`：选择 `v0`、`s1` 或 `candidate`，默认 `s1`。
- `--iemoec-survival`：选择 `rank`、`rank_crowding` 或 `nsga3`。
- `--iemoec-crowding`：启用拥挤度，供消融实验使用。
- `--no-recombination`：关闭跨岛组合，供消融实验使用。
- `--recombination-budget-ratio`：聚合阶段的重组预算比例，默认为 1.0。
- `--late-recombination-budget-ratio`：Pareto 阶段的重组预算比例，默认为 0.25。
- `--retain-island-state`：跨外循环继续演化已有岛种群，供消融实验使用。
- `--fixed-island-definitions`：固定岛权重但仍逐轮重建，用作状态保留的严格对照。
- `--origin-ratio`：起源种群占共同种群的比例，默认为 0.2。
- `--island-initialization`：覆盖 profile 的 `multi_ancestor` 或 `single_ancestor`。
- `--direction-neighbor-ancestors`：每个岛优先注入的方向邻近解数量，默认为 4。
- `--diverse-ancestors`：每个岛注入的决策空间差异解数量，默认为 2。
- `--island-direction-mode`：选择 `axis_random` 或 `reference_subset`。
- `--island-count-multiplier`：构造 `2M` 或 `4M` 个岛；尚无实验支持 4M 最优。
- `--outer-batch-ratio`：固定外批次相对共同种群 N 的比例。
- `--local-fe-ratio`、`--recombination-fe-ratio`：candidate 批次内 FE 分配，两者之和必须为 1。
- `--pairing-strategy`：`farthest_weight`、`nearest_weight`、`random`、`farthest_decision` 或 `none`。
- `--inner-generations-early`：前期每轮岛内演化代数，默认为 1。
- `--inner-generations-late`：后期每轮岛内演化代数，默认为 1。

## IEMOEC 版本架构

| variant | schema | 初始化 | 建岛 | 方向与 FE 调度 | 全局选择 |
|---|---:|---|---|---|---|
| `v0` | 0 | 仅评价小 origin | 单祖先 PM 扩岛 | axis/random、旧预算 | 旧双重选择 |
| `s1` | 1 | 仅评价小 origin | 全局池多祖先、0 FE | axis/random、旧预算 | 旧双重选择 |
| `candidate` | 2 | 评价公共完整 N | origin anchor + supporting founders | 参考方向子集、固定批次 | 每轮一次 survival |

默认 `s1` 保持已完成 pilot 使用的多祖先算法，不自动启用 candidate。candidate 每轮使用同一
ideal/nadir 归一化上下文，先按 X 去重，再执行一次可消融的外层 survival；origin 优先吸收
存活的岛方向代表，再用非支配等级和拥挤度补齐。`rank`、`rank_crowding`、`nsga3` 的输出标签
分别为 IEMOEC-Rank、IEMOEC-CD、IEMOEC-RD。

每个 `config.json` 都记录 `algorithm_schema_version`。不同 schema 不允许写入同一结果目录，
即使指定 `--force` 也必须更换 `--run-name`。未显式指定 run-name 时，目录名自动包含 variant。

## 输出结构

```text
results/my_pilot/
  DTLZ2/
    M5/
      IEMOEC/
        seed_007/
          config.json
          history.csv
          final_population.csv
          metrics.json
  failures.json
```

其中 `history.csv` 的 `fe` 是公共固定检查点，最终行与 `metrics.json` 使用完全相同的最终 F。
`final_population.csv` 保存决策、目标值和 provenance。candidate 的 `iemoec_diagnostics.csv` 还记录
founder 多样性、合并唯一率、local/recombination 后代与存活率、方向覆盖率和每轮固定 FE 批次。
主指标为 IGD+、HV，补充 GD+、pymoo Spacing、方向覆盖率和运行时间；历史默认不计算 HV。

## 汇总、统计与作图

```powershell
python scripts/summarize.py results/my_pilot
python scripts/plot_results.py results/my_pilot --kind all
```

汇总产物包括：

- `summary.csv`：mean、median、std、IQR；
- `wilcoxon_holm.csv`：配对 Wilcoxon、Holm 校正和 Vargha-Delaney A12 效应量；
- `friedman.json`：跨问题平均排名与 Friedman 检验；
- `figures/`：中位数/IQR 收敛曲线、最终 IGD+ 箱线图和平行坐标图。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖公共初始化、目标尺度不变性、多祖先 0 FE、方向子集、pairing、去重、固定 FE 调度、
严格 MaxFEs、DTLZ2/WFG1 M=3/5/10 集成、最终 history/metrics 一致性和 schema 防混写。

## 代码结构

```text
src/iemoec_experiment/
  config.py      实验与 IEMOEC 配置
  problems.py    pymoo 问题工厂及旧 C-DTLZ2 薄包装
  factory.py     pymoo baseline、参考方向与算子工厂
  initialization.py 公共初始决策向量
  normalization.py 单轮公共目标归一化
  directions.py 参考方向子集选择
  metrics.py     公共参考 PF、归一化和指标
  iemoec.py      IEMOEC 自定义核心
  runner.py      单任务执行、检查点和标准结果输出
scripts/
  run.py         批量实验/并行/断点续跑入口
  summarize.py   多种子统计检验
  plot_results.py 统一离线绘图
tests/
  test_experiment.py
```

旧的自实现 baseline、DTLZ、遗传算子和指标代码已经移除，避免与 pymoo 管线混用。Git 历史仍保留旧实现以供追溯。
