# IE-MOEO v3：核心确认与邻域、分块诊断

## 已有证据与本次目的

2026-11-03 核对 v2 的 1350 项机制 pilot，全部完成且 FE、参数及源码冻结记录一致。
每个场景只有 5 个配对 seed，双侧精确 Wilcoxon 最小 p=0.0625，684 项登记主对照均未显著。
以下均为开发趋势，不能写成“机制已经显著成立”：

| 对照 | 18 个场景的 IGD+ 中位数更优/相等/更差 | 判断 |
|---|---|---|
| full / rank | 18/0/0 | 显式维护最值得确认 |
| full / no_combination | 16/0/2 | 全局组合相对局部预算转移值得确认 |
| full / local_random | 13/0/5 | 每岛最佳选择有正向趋势 |
| full / global_best | 15/0/3 | 按岛保留代表有正向趋势 |
| full / shared | 12/0/6 | 隔离有条件性效果，DTLZ3 三种目标数均为负面 |
| full / unchecked | 0/18/0 | 90 次最终种群文件完全相同，检查未体现贡献 |
| full / sbx；full / random_coord | 9/0/9；8/0/10 | 整块继承没有稳定优势 |

隔离×全局组合的对数差中差在 17/18 个场景为负；这表示组合存在时隔离的相对效应通常更有利，
不意味着所有场景的两个主效应均为正，也不是已经检出了显著协同。

## 邻域问题如何处理

旧 `representative_dominance` 只在代表之间查找邻近支配者；非支配排序往往已经排除被拒绝者。
v2 full 虽有 32/2934 轮出现拒绝，但没有一轮改变实际亲本池。仅调大旧半径无法解决这种冗余。

新增显式可选 `--parent-check-mode local_scalar`：

- 用本轮局部进化后的已评价联合池作参考，与代表共用归一化及目标权重。
- 计算按变量边界归一化的 RMS 距离；邻域内若存在标量值低至少 1e-12 的候选，则代表不合格。
- 相比支配检查，它也可能移除非支配的权衡代表，因此可能降低多样性；不保证有益。
- 不产生探测点、不增加 FE；仍沿用合格者不足两个时的原保底规则，并报告保底次数。
- 默认算法仍保留旧检查。新规则需要单独实验，不把它伪装成旧机制的同义修复。

parents 预先固定 RMS 半径 0.2/0.4，每个半径分别比较 checked/unchecked；每对仅改变 parent_check。
两种半径是开发候选，不是已调出的最优值。旧规则用欧氏距离，新规则用 RMS，半径数值不可直接比较。

先看 `rounds_check_changed_parent_pool` 是否大于零，再看 IGD+/HV。若只产生拒绝、被保底撤销，
或始终不改变最终亲本池，则不能解释为有效贡献。若影响亲本但质量变差，应保留负面结果并考虑关闭，
不通过弱化对照来制造优势。

## 整块继承问题如何处理

uniform 四块只是等分坐标，不保证与问题关联结构一致；组合后的逐坐标变异也可能破坏完整块。
本次保持原算子作基准，增加有归因能力的分组控制：

| 组 | 改动及作用 |
|---|---|
| full / uniform_2 | 原四块与均匀两块，检查块数影响 |
| known_2 / uniform_2 | 同为两块，比较位置/距离分组与坐标均分；两者块尺寸可能不同 |
| known_2 / shuffled_known_2 | 块数和每块尺寸完全相同，仅打乱坐标归属，检验结构信息的贡献 |
| known_2 / known_sbx | 相同配置，只改组合方法为 SBX |
| known_2 / known_coord | 相同配置，只改为逐坐标继承 |

shuffled_known 每次运行用独立 seed 固定一份坐标分组，不消耗算法搜索随机流；它只用变量位置分组信息，
不使用真实 Pareto 前沿或新增评价。known 分组需要问题知识，不能宣传为无先验、数学完全可分或普遍有效。

审计新增亲本索引对数、不同决策的亲本对数、变异前后亲本克隆数、来自任一亲本的完整块数、
组合候选数，以及候选与下一轮起源者、最终非支配候选集的交集计数。
这些是产生与保留诊断；缓存克隆也可能重叠，不应将交集计数直接当作净新增贡献。
若新分组仍不优于 SBX/逐坐标继承，就不应保留“整块继承普遍更好”的主张。

## 运行阶段与成本

本次本地验证：146 项单元/集成测试通过，v3 的 34 项冒烟全部完成且 FE 完整；旧八组的 16 个
schema 2 冒烟案例与当前对应配置的最终种群文件、IGD+/HV/GD+/覆盖和 FE 分账逐项一致。
NSGA2、NSGA3、MOEAD、MOEADPBI、IEMOEC 的缓存身份及实现签名与修改前一致。
回归明细在 `results/iemoeo_mechanisms_v3/smoke/attribution/schema2_regression.json`。

新增机制的实际冒烟观测如下，不能解释为性能确认：

| 诊断 | 实际观测 | 含义 |
|---|---|---|
| checked_02 | 8 轮中 7 轮拒绝、6 轮改变亲本池、无保底 | 新检查确实影响搜索；两例 IGD+ 都差于对应 unchecked，不提升为默认 |
| checked_04 | 8 轮均改变亲本池、5 轮保底，组合亲本索引对总数降到 8 | 大半径可能过度筛选，组合机会减少也是影响路径，不能只报拒绝率 |
| 原 full | 150 对、600 次块观测，变异后完整块 453 次 | 约 24.5% 的块观测被变异改变，完整继承和最终完整性需要区分 |
| known_2 | 300 次块观测，变异后完整块 183 次 | 两块方案约 39.0% 被改变；块尺寸不同不能直接比原始计数 |
| known_2 / shuffled_known_2 | 两例 IGD+ 都更差，DTLZ2 的 HV 更好、WFG9 的 HV 更差 | 没有证据把已知分组升级为默认，需要保留两指标和负面结果 |

不要看到新分支激活就宣称修复了性能贡献；若 pilot 延续负面趋势，关闭检查或使用普通组合也是合理结论。

v3 的 smoke 为 DTLZ2/WFG9、M3、seed1、20N。
pilot 为 DTLZ2/3/7、WFG3/6/9，M3/4/10，**新的开发 seeds6–15、200N**。
confirm 为同样 18 场景、**独立 seeds31–60、400N**；many_objective 则为 M5/8/15。
默认实验方案是 `experiments/iemoeo_mechanisms_v3.json`；v1/v2 文件、冻结记录和结果均保留。

| 套件 | 内容 | smoke | pilot | confirm |
|---|---|---:|---:|---:|
| core | full/rank/crowding、共享、组合开关、两种代表对照，共 8 组 | 16 | 1440 | 4320 |
| parents | full 加两对检查开关，共 5 组 | 10 | 900 | 2700 |
| blocks | 原配置、两种分组及匹配组合控制，共 7 组 | 14 | 1260 | 3780 |
| attribution | 三套去重后 17 组 | 34 | 3060 | 9180 |

pilot attribution 共 80,580,000 FE；confirm core 共 227,520,000 FE。
各套件独立登记时任务数包含相同锚点，但共享结果库只计算一次同配置任务。
这次 schema 3 为新 IE-MOEO 结果建立身份；公共基线的算法配置和实现不变，不因这次修改而失效。
pilot 的 10 seeds 主要用于激活、稳定性和参数开发，不是正式确认：即使没有零差值且方向完全一致，
双侧精确检验最小 p=0.001953125，仍不足以通过完整 attribution 的 576 项检验的首个 Holm 阈值。
不要为获得显著性而事后改成单侧检验或拆检验族；要检验贡献应使用预先登记的独立 30 seeds 确认阶段。

```powershell
# 功能冒烟；只应检验完整性、预算和开关，不判断机制优劣
python scripts/run_iemoeo_mechanisms.py --stage smoke --suite attribution --workers 4 --execute
python scripts/summarize_iemoeo_mechanisms.py results/iemoeo_mechanisms_v3/smoke/attribution

# 完整开发 pilot，包含核心、邻域及分块；失败或中断后可执行同一命令续跑
python scripts/run_iemoeo_mechanisms.py --stage pilot --suite attribution --workers 4 --execute
python scripts/summarize_iemoeo_mechanisms.py results/iemoeo_mechanisms_v3/pilot/attribution

# 个别任务失败时，只补未完成组和未完成 seed，保留原科学配置及冻结清单
python scripts/run_iemoeo_mechanisms.py --stage pilot --suite attribution --workers 4 --retry-incomplete --execute
```

资源有限时，分别执行 `--suite parents` 和 `--suite blocks`，用于解决当前两个疑点；
原核心已有 v2 pilot，可以在冻结当前原核心后直接安排 core 独立确认，不必为了阶段名称再做一遍开发。
若运行全 attribution，最终按全 attribution 汇总；不能挑分套件中较有利的 p 值。

```powershell
# 冻结当前原核心后，独立确认四项主张。不要一边看确认结果一边改参数。
python scripts/run_iemoeo_mechanisms.py --stage confirm --suite core --workers 4 --execute
python scripts/summarize_iemoeo_mechanisms.py results/iemoeo_mechanisms_v3/confirm/core
```

core 的 full 不包含新 local_scalar 或 known 分组。其确认只适用于原核心。
若在 pilot 中选择了新邻域规则或分组，要先冻结新的实验方案和完整消融组，以新 output-root 运行独立确认；
不能用旧 core 的确认结果为改过的完整算法背书。若已经查看 seeds31–60 的确认结果并据此修改算法，
后续新方案需要另用未查看的独立 seeds。

确认时报告全套登记对照、所有负面场景、IGD+/HV 与效应大小；主对照统一 Holm 校正，交互另为登记族。
`contrast_overview.csv` 分开列数值更优/相等/更差和显著胜负；零对照值不计算相对百分比。
解集大小不等时，方向覆盖与决策距离仅作辅助描述；若要主张固定输出容量下的多样性优势，还需等基数分析。
无组合控制转移名义额度，保证相同总 FE，不保证缓存命中后每阶段实际 FE 完全相同。

## 何时与其他算法比较

现在就能对**已冻结的当前 full**进行开发性比较，无需等所有机制为正。
此前 `iemoeo_pilot_200n_v1` 已经是横向 pilot；本次机制 pilot 与横向比较回答不同问题。
新候选配置应先通过功能冒烟和开发 pilot，再冻结配置进行正式横向比较及对应消融。

先复用现有 200N、seeds1–5 的六种公共基线做开发性比较，避免重跑 MOEAD。
本次逐条核对，六种基线各 240 项、共 1440 项旧结果都满足自动导入条件；该推荐批次只需计算新 IE-MOEO 的 240 项。

```powershell
$algorithms = @('NSGA2', 'NSGA3', 'MOEAD', 'MOEADPBI', 'RVEA', 'AGEMOEA2STABLE', 'IEMOEO')
python scripts/run.py --preset benchmark_pilot `
  --algorithms $algorithms `
  --objectives 3,4,10 --seeds 1-5 --evals-per-pop 200 `
  --iemoeo-selection-audit --workers 4 `
  --run-name iemoeo_v3_public_200n --dry-run
```

确认清单后移除 `--dry-run` 实际执行，再运行：

```powershell
python scripts/summarize.py results/iemoeo_v3_public_200n --target IEMOEO
```

同算法、函数、M、seed、预算、评价配置及实现匹配的结果直接复用，缺失任务补算；不要加 `--force`。
如需把 S3-Elite 也放进这个批次，在数组中加入 `IEMOEC`，同时传入 `--iemoec-variant s3_elite`。
改变算法列表时另取批次名，例如 `iemoeo_v3_with_s3_200n`，保留原冻结清单。
现有 S3-Elite 属于自定义算法旧布局，当前没有可核验的实现身份，按现有规则不会自动导入；
该选项会另算 240 项 S3-Elite，而不会重跑六种公共基线。旧 S3-Elite 结果仍保留为前期参照。
新预算 400N 或新的 seed 是不同任务，不能用旧 200N/其他 seed 冒充。
正式横向比较可以与核心消融并行安排，但必须事先冻结配置、问题、预算、指标和检验，至少 30 个独立 seed；
S3-Elite 始终只是性能参照，不能代替单因素归因对照。

低复杂度阶数、岛内并行加速、数学极值认证和网络/DRL 应用仍需单独研究，不由本轮质量结果推导。
