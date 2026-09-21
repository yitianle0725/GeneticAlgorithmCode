# IEMOEC：可复现的多目标进化优化实验平台

本项目基于 **Python 3.12+** 与 **pymoo 0.6.2**，用于研究和验证多目标极值组合算法
IEMOEC（Independent Evolution and Multi-objective Extremum Combination）。代码库同时提供
NSGA-II、NSGA-III、MOEA/D-TCH、MOEA/D-PBI、RVEA 和 AGE-MOEA2-Stable 等公共基线，
统一问题、初始化、种群规模、函数评价预算、指标和统计流程。

截至 2026-09-21，项目已经完成：

- 7 种算法、16 个 DTLZ/WFG 问题、5 个目标数、30 seeds、`400N` FE 的正式实验；
- `16800/16800` 个正式任务成功，0 failures；
- S2、S3-Memory、S3-Hybrid、S3-Elite 及核心机制消融；
- 忠实实现原始思想的独立 S4 原理验证版本（algorithm schema 10）；
- S3-Elite 运行热点优化，以及 2400 项单线程干净计时实验。

当前应区分两条研究线：

- **S3-Elite**：当前性能版本，用于与现有 many-objective 算法进行正式比较。
- **S4**：原理验证版本，用于审计“亲本选择、极值组合、独立进化”是否按原始文字实现；
  它不是 S3-Elite 的性能升级版。

## 1. 当前实验结论

### 1.1 S3-Elite 的正式表现

正式实验覆盖 DTLZ1–7、WFG1–9，`M=3/5/8/10/15`，每场景 30 seeds。
按 80 个“问题 × 目标数”场景的 IGD+ 中位数排名：

| 算法 | 平均排名 | 场景第一数 |
|---|---:|---:|
| RVEA | **2.700** | **29/80** |
| S3-Elite | **3.125** | 11/80 |
| NSGA-III | 3.150 | 7/80 |
| AGE-MOEA2-Stable | 3.462 | 11/80 |
| MOEA/D-PBI | 4.212 | 17/80 |
| NSGA-II | 5.662 | 2/80 |
| MOEA/D-TCH | 5.688 | 3/80 |

Friedman 检验为 `χ²=156.41, p≈3.4×10⁻³¹`。合理结论是：

> S3-Elite 具有明确竞争力，综合排名第二，与 NSGA-III 接近；它明显优于 NSGA-II 和
> MOEA/D-TCH，对 AGE-MOEA2-Stable、MOEA/D-PBI 总体占优，但尚未整体超过 RVEA。

S3-Elite 在 DTLZ5/6/7、WFG3/4 上表现突出；主要弱项是 DTLZ1–4、WFG6/9、M=3，
以及相对 NSGA-III、RVEA 不足的全局方向覆盖。完整分析见
[第二次 formal 实验报告](docs/第二次formal实验0920.md)。

### 1.2 运行效率优化

方向记忆候选缓存、批量方向标量化、向量化诊断和轻量个体复制已经实现，且未改变算法语义。
优化后的 `timing_optimized_s3_elite`：

- `2400/2400` 成功，0 failures；
- 与优化前正式结果逐任务比较，IGD+、GD+、HV、Spacing、方向覆盖等指标完全一致；
- 相对旧 formal 记录，算法累计时间由 `124.12 h` 降至 `39.80 h`；
- 与旧单线程干净计时的 27 个公共场景比较，单位 FE 几何平均加速约 `1.42×`，
  即单位 FE 耗时约下降 `29.8%`。

旧 formal 包含历史指标回调并可能存在多进程资源竞争，因此不能把 formal 记录中的约 3 倍差距
全部解释为代码优化。论文报告算法时间时，应使用 `--workers 1 --timing-only` 的干净计时结果。

### 1.3 三个研究问题目前的证据

| 研究问题 | 当前证据 | 当前判断 |
|---|---|---|
| 独立进化能否替代 Crowding Distance？ | S2、NoIsolation 及 A–I 控制实验 | 尚不能证明普遍替代；隔离与共享在部分总体结果上近似 |
| 极值组合是否改善全局探索？ | NoRecombination、不同 pairing、来源贡献与 HV | 重组对覆盖/HV 有积极证据，但不是普遍的 IGD+ 改善 |
| 独立子种群是否缓解 many-objective 选择压力？ | M=8/10/15、方向覆盖、S3 方向记忆实验 | 部分困难前沿有效，但 M15 和规则前沿仍不稳定 |

因此，现阶段可以主张“机制具有研究价值并形成有竞争力的算法”，不能主张三个问题都已被肯定证明，
也不能主张 S3-Elite 普遍优于全部强基线。

## 2. 环境安装

推荐使用独立 Conda 环境：

```powershell
conda create -n moo python=3.12
conda activate moo
python -m pip install -r requirement.txt
python -c "import pymoo; print(pymoo.__version__)"
```

主要依赖包括 NumPy、SciPy、Matplotlib、pymoo、Numba 和 Pydantic。
项目要求 `pymoo>=0.6.2,<0.7`；AGE-MOEA2 需要 `numba>=0.59`。

运行测试：

```powershell
python -m unittest discover -s tests -v
```

当前测试集包含 68 项测试，覆盖公共初始化、FE 审计、指标、S2/S3 结构、S4 封闭谱系、
有限邻域资格检查、坐标继承、schema 隔离和断点续跑。

## 3. 快速开始

所有批次都建议先运行 `--dry-run`，核对任务数、variant、schema、MaxFEs 和输出目录，
确认后再去掉 `--dry-run`。

以下命令运行 7 个算法的最小冒烟实验：

```powershell
python scripts/run.py --preset benchmark_smoke `
  --iemoec-variant s3_elite `
  --workers 4 `
  --run-name smoke_s3_elite `
  --dry-run
```

确认显示 7 项后正式运行：

```powershell
python scripts/run.py --preset benchmark_smoke `
  --iemoec-variant s3_elite `
  --workers 4 `
  --run-name smoke_s3_elite
```

运行结束会输出：

```text
完成=<数量> 跳过=<数量> 失败=<数量> | <结果目录>
```

相同配置和 run name 可以安全续跑，已经完成的任务会自动跳过。配置发生变化时必须更换
`--run-name`；即使指定 `--force`，也不允许把不同 algorithm schema 或 metric schema 混入同一目录。

## 4. 公平实验口径

平台遵循以下共同规则：

- baseline、IEMOEC 在相同问题和目标数下使用相同参考方向数 `N`；
- 每个 problem、M、seed 共享同一份确定性初始决策向量；
- 所有算法严格消耗相同 MaxFEs，不使用 IGD/HV 早停；
- 正式实验不为 IEMOEC 提供额外 PF 后处理；
- 低维计算精确 HV，高维使用固定公共采样的 Monte Carlo HV；
- 主指标为 IGD+ 和 HV，补充 GD+、Spacing、ONVG、非支配比例、方向覆盖和运行时间；
- 每个任务保存初始化哈希、实际 FE、种群规模、算法/指标 schema 和运行时间分解。

默认参考方向对应的公共种群规模与预算如下：

| M | N | pilot `200N` | formal `400N` |
|---:|---:|---:|---:|
| 2 | 100 | 20000 | 40000 |
| 3 | 91 | 18200 | 36400 |
| 5 | 210 | 42000 | 84000 |
| 8 | 120 | 24000 | 48000 |
| 10 | 220 | 44000 | 88000 |
| 15 | 120 | 24000 | 48000 |

推荐使用 `--evals-per-pop` 指定预算。显式使用 `--max-fes` 时，它必须能被相应的公共种群规模整除。

## 5. IEMOEC 版本谱系

| variant | schema | 作用与关键结构 |
|---|---:|---|
| `v0` | 0 | 旧版小 origin、单祖先扩岛和旧 FE 调度，仅供追溯 |
| `s1` | 1 | 多祖先建岛，保留旧调度，仅供结构对照 |
| `candidate` | 2 | 公共完整 N、固定批次和单次外层 survival |
| `s2` | 3 | axis/random、2M 子群、75/25 局部/重组预算、NSGA-III survival |
| `s2_no_isolation` | 4 | 使用共享父代池的去隔离控制 |
| `s3_memory` | 5 | 在 S2 上增加持久方向微种群 |
| `s3_hybrid` | 6 | 50/25/25 的隔离、共享、重组三来源繁殖 |
| `s3_elite` | 7 | 当前性能版本：S3-Hybrid + 方向收敛精英保护 |
| `s3` | 8 | 增加贡献驱动的自适应来源预算，实验上未取代 S3-Elite |
| `principle` | 9 | 第一版忠实原理实现；存在等待全部谱系合格的调度缺陷 |
| `s4` | 10 | 当前原理验证版：异步资格触发、多尺度有限邻域检查 |

统一 CLI 默认 variant 仍是 `s2`，目的是保持旧命令兼容。运行当前性能版本时必须显式写：

```text
--iemoec-variant s3_elite
```

### 5.1 S3-Elite 性能架构

S3-Elite 的每轮主要流程是：

1. 用公共完整种群初始化；
2. 按方向维护跨轮次的独立微种群记忆；
3. 将本轮 FE 按 50% 隔离繁殖、25% 共享繁殖、25% 跨方向重组分配；
4. 合并旧种群、方向记忆和三类后代并按决策向量去重；
5. 从前两个非支配层为各方向提取收敛精英；
6. 保护方向精英，再使用 NSGA-III survival 补齐公共种群；
7. 记录来源存活、方向改进、方向记忆周转和覆盖等诊断信息。

S3-Elite 是工程性能版本。它包含共享繁殖、方向记忆和参考方向 survival，因此不能被描述为对原始
“完全封闭独立进化流程”的逐字实现。

### 5.2 S4 原理验证架构

S4 单独实现原始思想：

1. 只随机生成约 `P=N/5` 个起源者；
2. 每个起源者单独扩展为带谱系编号的小种群；
3. 小种群之间不迁移、不共享亲本，各自在内部交叉、选择和变异；
4. 局部停滞后执行可审计的多尺度有限邻域检查；
5. 只有通过检查的不同谱系代表才有资格作为组合亲本；
6. 通过坐标继承和变异产生混血后代；
7. 只对本轮混血后代做非支配等级选择，形成下一轮起源者。

有限邻域检查只是指定半径和有限方向下的数值证据，不是导数、KKT 或数学极值证明；坐标继承也
不保证子代是驻点。详细约定、已知限制和审计字段见
[原理验证版本说明](docs/原理验证版本0914.md)。

## 6. 实验预设

| preset | 用途 | 默认规模 |
|---|---|---:|
| `smoke` | 最小正确性检查 | 1 problem × 1 M × 1 seed |
| `pilot` | DTLZ1–4 初步实验 | 4 × 3 × 5 |
| `benchmark_smoke` | 7 算法冒烟 | 7 项 |
| `benchmark_pilot` | 16 问题、M=3/5/10、7 算法 | 1680 项 |
| `structure` | 结构消融代表集 | 每算法/variant 120 项 |
| `mechanism` | 7 个机制代表问题 | 每算法/variant 105 项 |
| `s3_development` | S3 高维开发集 | 每 variant 100 项 |
| `formal` | 16 问题、5 个 M、30 seeds、7 算法 | 16800 项 |
| `custom` | 完全自定义 | 由命令参数决定 |

查看所有参数：

```powershell
python scripts/run.py --help
```

## 7. 正式实验复现

一次性复现全部正式实验会生成 16800 项：

```powershell
python scripts/run.py --preset formal `
  --iemoec-variant s3_elite `
  --workers 4 `
  --run-name formal_full_s3_elite `
  --dry-run
```

确认显示 `实验任务: 16800` 后去掉 `--dry-run`。为了断点管理和避免修改 IEMOEC 后重跑固定
baseline，也可以每种算法独立运行。例如：

```powershell
python scripts/run.py --preset formal `
  --algorithms IEMOEC `
  --iemoec-variant s3_elite `
  --workers 4 `
  --run-name formal_s3_elite_400n_schema5 `
  --dry-run
```

该命令应显示 2400 项。六个正式 baseline 使用相同形式，将 `--algorithms` 和 `--run-name`
分别替换为：

| `--algorithms` | 推荐 run name |
|---|---|
| `NSGA2` | `formal_nsga2_400n_schema5` |
| `NSGA3` | `formal_nsga3_400n_schema5` |
| `MOEAD` | `formal_moead_tch_400n_schema5` |
| `MOEADPBI` | `formal_moead_pbi_400n_schema5` |
| `RVEA` | `formal_rvea_400n_schema5` |
| `AGEMOEA2STABLE` | `formal_age_stable_400n_schema5` |

`MOEAD` 表示 Tchebycheff 分解；`MOEADPBI` 表示 `PBI(theta=5.0)`。
正式比较使用 `AGEMOEA2STABLE`，不能把旧 `AGEMOEA2` 结果混入其中。

## 8. 干净计时

干净计时必须单进程运行，避免任务之间争抢 CPU，并关闭算法执行期间的历史指标回调：

```powershell
python scripts/run.py --preset formal `
  --algorithms IEMOEC `
  --iemoec-variant s3_elite `
  --timing-only `
  --workers 1 `
  --run-name timing_optimized_s3_elite `
  --dry-run
```

`metrics.json` 会分别保存：

- `algorithm_runtime_seconds`；
- `metric_runtime_seconds`；
- `io_runtime_seconds`；
- `total_runtime_seconds`。

只有 `algorithm_runtime_seconds` 适合用于算法内核时间比较。普通多进程 formal 的时间适合估算实验成本，
不应作为严格的算法效率结论。

## 9. 自定义实验

```powershell
python scripts/run.py --preset custom `
  --algorithms NSGA3 RVEA AGEMOEA2STABLE IEMOEC `
  --problems dtlz2 dtlz5 wfg3 wfg9 `
  --objectives 3,5,8,10,15 `
  --seeds 31-35 `
  --evals-per-pop 400 `
  --iemoec-variant s3_elite `
  --workers 4 `
  --run-name s3_elite_confirmation_seeds31_35 `
  --dry-run
```

当前 1–30 seeds 已经用于发现 S3-Elite 的优势和弱点。若基于这些结果开发 S3.1，建议冻结算法后
使用新 seeds（例如 31–60）或独立问题集做确认实验，避免只在开发数据上报告改进。

常用参数：

- `--dry-run`：只生成并展示任务清单；
- `--force`：重跑完全相同的配置；
- `--run-name`：固定输出批次名，支持断点续跑；
- `--workers`：任务级并行数；
- `--evals-per-pop`：设置 `MaxFEs=N×倍数`；
- `--timing-only`：关闭历史指标回调；
- `--no-recombination`：关闭重组，用于机制消融；
- `--iemoec-survival`：选择 `rank`、`rank_crowding` 或 `nsga3`；
- `--pairing-strategy`：设置方向/决策空间配对策略；
- `--isolated-fe-ratio`、`--shared-fe-ratio`：覆盖 S3 来源预算；
- `--direction-memory-capacity`：覆盖方向微种群容量；
- `--principle-*`：设置 S4 局部代数、停滞、探测半径和容差。

S4 会拒绝 S2/S3 专属的共享繁殖、参考方向 survival、状态保留等参数，防止原理版本被旧机制污染。

## 10. 输出与审计

```text
results/<run-name>/
  experiment_manifest.json
  failures.json
  DTLZ2/
    M5/
      IEMOEC/
        seed_007/
          config.json
          metrics.json
          history.csv
          final_population.csv
          iemoec_diagnostics.csv
```

主要文件：

- `experiment_manifest.json`：批次声明的全部任务和 schema；
- `failures.json`：失败任务，完整成功时为 `[]`；
- `config.json`：规范化算法配置、variant、schema 和预算；
- `metrics.json`：最终指标、FE、初始化哈希和时间分解；
- `history.csv`：固定 FE 检查点的收敛历史；
- `final_population.csv`：最终决策变量、目标值和可用的 provenance；
- `iemoec_diagnostics.csv`：IEMOEC 每轮预算、来源贡献、覆盖和记忆诊断；
- `lineage_audit.csv`：S4 的谱系、资格检查、组合亲本与坐标继承审计。

汇总程序默认严格验证 manifest，缺少或出现额外任务都会报错。探索性不完整结果只能显式使用
`--allow-incomplete`，并会生成排除记录。

```powershell
python scripts/summarize.py results/my_run
python scripts/plot_results.py results/my_run --kind all
```

汇总产物包括描述统计、配对 Wilcoxon、Holm 校正、Vargha–Delaney A12、Friedman 检验和图表。

离线审计不同 HV 参考点，不需要重新运行算法：

```powershell
python scripts/audit_hv.py results/my_run `
  --reference-points 1.1 1.5 2.0 `
  --samples 200000
```

## 11. 项目结构

```text
src/iemoec_experiment/
  config.py               实验配置、版本 profile 与 schema
  problems.py             DTLZ/WFG 问题工厂
  factory.py              baseline、参考方向和公共算子
  initialization.py       跨算法公共初始化
  normalization.py        目标归一化
  directions.py           方向子集和方向目标
  directional_memory.py   S3 持久方向记忆
  source_budget.py        S3 固定/自适应来源预算
  iemoec.py               S2/S3 算法核心
  principle.py            principle/S4 原理验证核心
  metrics.py              参考前沿与质量指标
  manifest.py             实验任务清单
  runner.py               单任务执行、计时和标准输出
scripts/
  run.py                   统一批量实验入口
  summarize.py             统计检验与汇总
  plot_results.py          收敛、箱线和平行坐标图
  audit_hv.py              离线 HV 审计
  draw_iemoec_framework.py 算法框架图
tests/
  test_experiment.py       实验平台与 S2/S3 测试
  test_principle.py        principle/S4 测试
docs/
  第二次formal实验0920.md   当前正式实验报告
  实验改进方案0912.md      S3 设计与实验方案
  原理验证版本0914.md      S4 实现边界与审计说明
```

## 12. 研究与实现边界

- 旧自实现 baseline、DTLZ/WFG、遗传算子和指标已经移除，公共组件统一复用 pymoo；
- S3-Elite 的正式优势是经验结果，不构成极值组合理论的数学证明；
- S4 的有限邻域“极值资格”不能称为严格驻点或 KKT 证书；
- 当前研究对象是无约束、有限边界的连续多目标问题；
- formal 使用 `400N`，pilot 通常使用 `200N`，二者不能直接拼接 seeds；
- 修改算法行为必须提升 algorithm schema 并使用新 run name；纯等价性能优化可以保留 schema，
  但必须用固定 seed 回归确认最终种群和诊断记录不变。

下一阶段优先研究：修正精英保护后的方向占用、降低 M=3 退化、改善 DTLZ1–4 的规则前沿表现，
以及为 WFG6/9 引入对变量关联更友好的亲本选择。任何新版本都应先小规模机制验证，再使用独立 seeds
进行确认，而不是直接重跑并反复调参全部 formal。
