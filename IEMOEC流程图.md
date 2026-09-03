# IEMOEC 当前实现流程

```text
ExperimentCase
  │  problem / M / seed / MaxFEs
  ▼
pymoo Problem + 公共参考方向
  │
  ├─ NSGA-II / NSGA-III / MOEA/D
  │    └─ pymoo 原生算法，运行到共同 MaxFEs
  │
  └─ IEMOEC
       │
       ├─ 小规模起源种群（默认约共同种群的 1/5）
       ├─ M 个极端岛 + M 个折中岛
       ├─ pymoo PM 扩增，各岛独立演化
       │    ├─ 前期：Tchebycheff 聚合选择
       │    └─ 后期：Pareto 选择
       ├─ 固定预算跨岛 SBX + PM 组合
       ├─ 外层环境选择（默认 NSGA-III survival）
       └─ 重复至 evaluator.n_eval == MaxFEs
```

## 与旧实现的关键差异

- 不再按“外循环代数”与 baseline 比较，而是统一函数评价次数。
- 不再对所有精英两两组合；每个精英仅选择固定数量的跨岛伙伴，且总后代受剩余 FE 限制。
- 不使用 HV/IGD 早停。
- 不执行 IEMOEC 专属的最终 PF 变异扩展。
- SBX、PM、非支配排序、拥挤度和 NSGA-III survival 均来自 pymoo。
- `rank`、`crowding`、`recombination` 可独立切换，用于机制消融。

## 指标路径

```text
最终解集
  ├─ pymoo IGD+ / GD
  ├─ 公共 ideal/nadir 归一化
  │    ├─ M≤5：pymoo 精确 HV
  │    └─ M>5：固定公共采样的 Monte Carlo HV
  ├─ Spacing / ONVG / ND ratio
  └─ 多种子 Wilcoxon-Holm + Friedman 排名
```

平行坐标图和收敛图仅在实验完成后生成，不进入算法计时。
