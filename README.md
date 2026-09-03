# IEMOEC 实验平台

本项目已重构为基于 **pymoo 0.6.2** 的可复现多目标/多目标数优化实验平台。

## 设计原则

- NSGA-II、NSGA-III、MOEA/D、DTLZ、WFG、SBX、多项式变异、非支配排序和质量指标直接使用 pymoo。
- 四个算法在相同问题与目标数下共享种群规模，并严格使用相同 `MaxFEs`。
- 随机种子从命令行一直传入算法；同一问题的四算法使用相同种子集合。
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

## 三级实验

### 1. 正确性冒烟测试

DTLZ2、M=3、1 个种子、4 个算法，预算为 20 倍共同种群大小：

```powershell
python script/run.py --preset smoke --workers 4
```

### 2. 预实验

DTLZ1–4、M=3/5/10、5 个种子：

```powershell
python script/run.py --preset pilot --workers 6
```

### 3. 正式实验

DTLZ1–4、WFG1/2/4/9、M=3/5/8/10/15、30 个种子：

```powershell
python script/run.py --preset formal --workers 6
```

正式实验任务很多。建议先完成 smoke 和 pilot，确认参数后再启动。仅使用任务级并行，不要同时开启岛级多进程。

## 自定义实验

推荐用 `--evals-per-pop` 定义预算，它会自动得到各目标数下与共同种群规模整除的 `MaxFEs`：

```powershell
python script/run.py --preset custom `
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
- `--force`：显式覆盖相同目录中的已有任务。
- `--history-hv`：在历史检查点计算 HV；高维时不建议启用。
- `--run-name`：固定结果批次名，用于断点续跑。
- `--iemoec-survival rank`：关闭外层 NSGA-III survival，供消融实验使用。
- `--iemoec-crowding`：启用拥挤度，供消融实验使用。
- `--no-recombination`：关闭跨岛组合，供消融实验使用。

若已有结果的 `config.json` 与新任务不同，程序会拒绝混写。应更换 `--run-name`，或确认后使用 `--force`。

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

其中 `history.csv` 的横轴始终是 FE；`metrics.json` 记录最终 IGD+、GD、HV、Spacing、ONVG、第一前沿比例、运行时间和实际 FE。

## 汇总、统计与作图

```powershell
python script/summarize.py results/my_pilot
python script/plot_results.py results/my_pilot --kind all
```

汇总产物包括：

- `summary.csv`：mean、median、std、IQR；
- `wilcoxon_holm.csv`：IEMOEC 与各 baseline 的配对 Wilcoxon 检验及 Holm 校正；
- `friedman.json`：跨问题平均排名与 Friedman 检验；
- `figures/`：中位数/IQR 收敛曲线、最终 IGD+ 箱线图和平行坐标图。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 DTLZ2 的 PF 方程、WFG 构造、参考方向规模、高维 HV 可复现性、四算法共同 FE、种子可复现性、断点跳过和配置冲突保护。

## 代码结构

```text
src/iemoec_experiment/
  config.py      实验与 IEMOEC 配置
  problems.py    pymoo 问题工厂及旧 C-DTLZ2 薄包装
  factory.py     pymoo baseline、参考方向与算子工厂
  metrics.py     公共参考 PF、归一化和指标
  iemoec.py      IEMOEC 自定义核心
  runner.py      单任务执行、检查点和标准结果输出
script/
  run.py         批量实验/并行/断点续跑入口
  summarize.py   多种子统计检验
  plot_results.py 统一离线绘图
tests/
  test_experiment.py
```

旧的自实现 baseline、DTLZ、遗传算子和指标代码已经移除，避免与 pymoo 管线混用。Git 历史仍保留旧实现以供追溯。
