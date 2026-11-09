# 完整五因素后的下一步：清单修复与独立确认

本次共享库已有完整 3330 项结果，但 factorial 中 16 个直接复用的 screen 组没有创建批次清单。
原因是旧调度器在筛掉完整组后，才由实际运行的子进程写清单。修复后先为全部组登记结果引用，
再决定哪些任务需要计算；本次仅补清单，不修改计算结果，3330 项严格汇总已完成。
修复记录：`results/iemoeo_s3_attribution_v1/pilot/factorial/manifest_repair_validation.json`。

完整因子分析以每个 seed 内的 16 个匹配背景平均 log IGD+ 为单位，以下只是开发方向：

| 因素 | 有利/不利场景 | 各场景效应比的中位数 |
|---|---:|---:|
| 代表选择 P | 16/2 | 0.90256 |
| 局部隔离 I | 14/4 | 0.96944 |
| 精英保护 E | 16/2 | 0.97875 |
| 方向记忆 M | 11/7 | 0.98869 |
| 整块继承 C 相对 SBX | 10/8 | 0.99765 |

比值小于 1 为有利。不能把背景个数当独立种子，也不能把这些小效应全部写成已确认。
全部主对照、交互和边际效应在 5 个开发 seed 下都没有通过各自登记的 Holm 检验。

`f11011`（I=1/P=1/C=0/M=1/E=1）在 37 个开发组的平均中位数排名最好，
对 S3-Elite 的 IGD+ 中位数为 12 个场景更好、6 个更差，相对改善的场景中位数约 3.78%。
它的组合算子是 SBX，因此后续即使确认通过，也不能证明整块继承必要或数学极值资格有效。
这是从开发集选择的候选，不能用它的开发排名当作无偏确认。

下一步冻结该候选，采用新文件 `experiments/iemoeo_s3_confirmation_v1.json`。
使用同样的 18 场景、200N，但换独立 seeds61–90，确认同预算下的可重复贡献。
这是在查看确认数据前登记的新方案，不修改旧方案原有的 400N 计划，也不覆盖旧冻结记录。

九个组包括：候选、关闭隔离、随机代表、关闭记忆、关闭保护、记忆开启/关闭时的两种整块控制、
无组合控制及原 S3-Elite。这样可以保留 I/P/普通跨群体重组的归因，并检查整块与记忆的交互。
共 4860 项、136404000 FE；主对照 8×18×2=288 项联合 Holm，另有 18 项登记交互检验。
新 seed 是必要的新实验，不能由开发结果代替。所有场景与负面结果都保留，不在确认集上换候选或调参数。

```powershell
# 只登记科学配置，不开始长批次
python scripts/run_s3_attribution.py --plan experiments/iemoeo_s3_confirmation_v1.json --stage confirm --suite confirm_core --workers 4

# 实际独立确认；中断后执行同一命令续跑
python scripts/run_s3_attribution.py --plan experiments/iemoeo_s3_confirmation_v1.json --stage confirm --suite confirm_core --workers 4 --retry-incomplete --execute
python scripts/summarize_iemoeo_mechanisms.py results/iemoeo_s3_confirmation_v1/confirm/confirm_core
```

本轮只修复清单、完成汇总并登记下一方案，不自动启动 4860 项确认计算。
若独立确认支持候选，表述可围绕“固定方向局部搜索、代表择优与跨方向重组的协同策略”；
若收益依赖记忆/保护，则保留该成立条件。极值资格与整块继承目前仍应保留为待研究机制。
