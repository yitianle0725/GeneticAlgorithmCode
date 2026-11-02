# IE-MOEO 16 次机制冒烟：证据边界与 schema 2 补充对照

## 数据范围

原数据为 `results/iemoeo_mechanisms_v1/smoke/core`：8 个机制组 × DTLZ2/WFG9 × M3 × seed 1，预算 20N=1820，每组只有两个场景、每场景一个 seed。16 次均成功，预算完整。

**已经验证的是执行、配置和部分开关的功能；尚无一项机制的普遍性能贡献得到统计确认。** 所有原始两两检验 p=1，单 seed 的 A12 即使为 0/1 也不能解释为充分的大小效应证据。本轮汇总保留 A12 数值，并把单 seed 的 magnitude 标为 `not_interpreted_single_seed`。

## 哪些已经验证，哪些还不完整

| 机制或主张 | 原 16 次证据 | 当前判断 |
|---|---|---|
| 隔离开关确实阻断跨岛交配 | full 的 8 轮均无外岛亲本；shared 的 6 轮均出现外岛亲本 | 功能已验证；不能推导优化优势 |
| 隔离改善质量或多样性 | DTLZ2 full/shared IGD+=0.296007/0.239132；WFG9 为 0.773531/0.622527。shared 的方向覆盖两例都更高，决策距离的方向则一好一坏 | 本次没有支持一致的质量/多样性优势；样本不足，不能宣称隔离普遍无效 |
| 参考方向维护有贡献 | full 相对 rank 两例 IGD+、HV 都更好；例如 DTLZ2 IGD+=0.296007/0.413692 | 有正向开发迹象，尚未确认；这也不支持“隔离已经替代显式多样性维护” |
| 邻域资格检查有贡献 | full 两例均未拒绝亲本，与 unchecked 的 IGD+/HV 相同 | 未激活，不能验证贡献；shared_rank 曾有一轮拒绝，说明该分支并非始终无效 |
| 整块继承优于 SBX | full 两例 IGD+ 比 sbx 低约 44.9% 和 21.5%，HV 也更高 | 值得优先确认，但不能从两次比较泛化 |
| 整块继承优于逐坐标继承 | full 两例 IGD+ 更低，但 WFG9 的 HV 略低于 random_coord | 指标并非一致占优，贡献未确认 |
| 隔离与其他机制协同 | 两个 2×2 的差中差已经可以计算，只有一个配对 seed，p=1 | 交互计算流程已验证，协同主张未确认 |
| 局部最佳代表优于随机/全局最佳代表 | v1 没有这两个对照，unchecked 仍选择每岛最佳代表 | 原实验无法归因；schema 2 已补实现 |
| 隔离可替代拥挤距离 | v1 的 rank 是末层随机截断，不是真实拥挤距离 | 原实验无法检验；schema 2 已补真实 CD 控制 |
| 全局组合本身有效 | SBX/逐坐标两组仍是重组，没有无全局组合的等总 FE 控制 | 原实验无法检验；schema 2 已补局部预算转移控制 |
| 数学极值、低复杂度阶数、岛内并行、网络/DRL 优势 | 这些实验没有对应认证、规模曲线、岛内并行或应用场景 | 仍不能验证，不能由标准函数质量或任务级 workers 推导 |

本轮只改变一个开关的对照估计的是整个搜索动态的总效应。原 full 与 shared 在同总 FE 下外层轮数、组合阶段实际 FE 也不同；这属于搜索动态的中介变化，不能把观测结果全部解释成一次交配算子的直接作用。

## 代码补充

生产默认仍为局部最佳代表、NSGA-III 参考方向维护和 structured 组合，新增选项仅在显式选择时作为控制。

1. `--representative-selection local_best/local_random/global_best`：每岛最佳、每岛随机或联合局部池标量最佳。联合池先按决策去重再取最佳，避免用重复克隆削弱全局适应度对照。目标槽位数等于非空岛数，预算未耗尽而独特候选不足时明确失败；预算尾部允许实际数量不足，并记录目标槽位与实际代表数，不额外评价补足。全局策略可能集中于少数岛，所以同时记录来源岛数与独特代表数。
2. `--iemoeo-survival rank_crowding`：调用 pymoo 的真实 `RankAndCrowding(crowding_func="cd")`。新增 crowding/shared_crowding，配合 rank/shared_rank 构成隔离×真实拥挤距离的 2×2。full 与 crowding 则比较参考方向和 CD。
3. `--combination-method none`：不执行显式全局组合，将本轮 `min(亲本索引对数, 组合上限, 剩余FE)` 的名义额度转给原交配规则的局部搜索。每次评价受额度限制；重复决策不计费，连续无进展会明确失败，不虚构 FE。局部搜索仍含 SBX；该控制不是“取消所有交叉”。
4. `--iemoeo-selection-audit`：在候选池中固定抽样至多 1024 对，按目标归一化 RMS≤0.02、边界归一化决策 RMS≥0.2 统计保留。使用独立随机流，不改变搜索随机数，不增加 FE。大池可能抽到重复对，统计是抽样描述而非独立样本检验。
5. 汇总新增实际拥挤距离轮数、代表来源、预算转移与候选对保留计数，并区分 implementation/development/confirmation/timing 证据范围。另记录 `parent_pool_changed_by_check`：资格检查可能只拒绝本来就不会被非支配排序和数量上限选中的个体，因此“有拒绝”仍不等于检查影响了组合亲本。

关闭组合的版本与完整版本保持相同总 FE。完整版本可能因缓存命中而实际组合 FE 低于名义额度；不宣称两组逐阶段实际 FE 完全相同。这个对照估计的是把计算资源放在全局组合或局部搜索的策略效果。

目标接近而决策远离的候选未被保留，不自动构成“错杀”：还受 Pareto 等级、选择名额和归一化影响。其保留统计是辅助证据；若要主张可替代拥挤距离，需要独立确认、事先登记实际意义/非劣界限，不能把不显著当成等效。

timing 阶段自动关闭选择审计，并要求 workers=1，避免把新增的候选对与亲本池诊断费用计入净计时；这仍不能证明复杂度阶数或岛内并行效率。

## 版本隔离与验证

- IE-MOEO 升为 algorithm schema 2；新方案为 `experiments/iemoeo_mechanisms_v2.json`。旧 v1 清单和结果保留，不重新贴上新版本标签。
- 新 attribution 冒烟包含 15 组 × 2 场景，共 30 次；其作用仍然是功能验证，不是新一轮性能结论。
- 原八组共 16 个案例与 schema 2 对应组回归核对，检查最终种群、IGD+/HV/GD+/覆盖、解集大小和 FE 分账。回归明细保存在新冒烟目录的 `schema1_regression.json`。
- baseline 的实现与使用的配置未改动；现有 NSGA2/NSGA3/MOEAD/MOEADPBI 结果已核对仍可复用。
- 辅助诊断的单元测试验证目标相同而决策不同的候选对计数，验证真实 CD 保留权衡两端、代表策略差异、无组合预算不超支，以及诊断不改变搜索随机流。

新 30 次功能冒烟全部完成，原八组的 16 个案例最终种群文件与关键指标、FE 分账逐项一致。新增诊断显示：

- crowding/shared_crowding 分别记录 8/6 轮真实 CD 选择，说明控制算子实际被调用。
- no_combination/shared_no_combination 的组合 FE 都为 0，分别有 150/100 FE 转给局部搜索，与各自名义额度相同；总预算仍为每次 1820。
- global_best 累计 152 个代表槽位均为不同决策，代表来源岛次数累计为 35，区别于 full 的 152。这验证了去重和来源控制，不证明该策略更优。
- radius_10 在 8 轮中出现 5 轮资格拒绝，但 `rounds_check_changed_parent_pool=0`，而 full 与其最终质量仍相同。筛掉的代表本来就没有进入最终亲本池，说明单独扩大半径未必能赋予检查实际作用。
- global_best 中有 1 轮检查改变了亲本池，表明这条影响路径在部分控制条件下存在；尚不能据此主张完整算法受益。
- 目标近/决策远的候选对保留计数有观测机会，但样本少、候选集不同、并非独立样本，仍不能确认“避免错杀”。

旧冻结记录和前期调试稿保留；当前有效实验目录为 `results/iemoeo_mechanisms_v2`，调试稿在 `results/iemoeo_mechanisms_v2_draft`。旧 schema 1 清单重建工具也保留已记录版本号和原共享路径，防止归档时被重新标成 schema 2。

## 下一步执行顺序

优先运行完整的 attribution pilot，避免只留下两例有利于块组合的结果。共 15 组 × 6 问题 × M3/4/10 × seeds 1–5 = **1350 项**，每项 200N；不运行其他七种算法。

```powershell
# 登记并核对任务数，不开始计算
python scripts/run_iemoeo_mechanisms.py --stage pilot --suite attribution --workers 4

# 开始机制 pilot；相同条件任务可断点续跑
python scripts/run_iemoeo_mechanisms.py --stage pilot --suite attribution --workers 4 --execute

# 全部组完成后汇总
python scripts/summarize_iemoeo_mechanisms.py results/iemoeo_mechanisms_v2/pilot/attribution
```

资源有限时可分套件执行 parents（450 项）、crowding（540 项）、combination（630 项），相同配置锚点自动复用。分套件的校正只覆盖该套件，不能挑其中最有利的 p 值冒充完整 attribution 的联合校正；完整 attribution 跑齐后应统一汇总一次。

pilot 仍只有 5 seeds，主要用于激活诊断和开发。先检查：full 的拒绝率是否仍为零；random/global 代表来源是否确实改变；crowding 轮数是否大于零；none 的组合 FE 是否为零且预算转移成功；候选对是否有足够观测机会。所有失败和负面场景都需保留。

冻结后用 seeds 31–60、400N 进行确认。完整 attribution 为 **8100 项**，先登记审查成本再决定执行，不自动启动：

```powershell
python scripts/run_iemoeo_mechanisms.py --stage confirm --suite attribution --workers 4
```

如在 pilot 后改变算法，使用新的 `--output-root` 保留冻结记录，再重新建立对应版本的 full 锚点。不要向 v1 目录补入 schema 2 结果。数学极值认证、并行效率和网络应用需要另行实现与实验，当前代码不会自动宣称这些主张已经成立。
