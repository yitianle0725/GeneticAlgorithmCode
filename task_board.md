# 任务板 — NSGA-III 论文实验补充

## 已完成 ✅

- [x] **标准 DTLZ1-4 测试**（论文 Section V-A）
  - M=3,5,8,10,15 五个目标维度
  - NSGA2 / NSGA3 / MOEAD 三算法对比
  - 结果：DTLZ2 M≥10 NSGA3 反超、DTLZ4 M≥8 NSGA3 碾压 → 复现成功

---

## 待做：按优先级排序

### 🔴 P0 — V-C 尺度缩放 DTLZ（最重要，必做）

**论文依据**：Deb & Jain (2014) Section V-C "Scaled Problems"

**核心理由**：
- 直接验证 NSGA-III **独家机制**：ASF 自适应归一化
- NSGA-II 的拥挤度、MOEA/D 的 TCH 均无尺度自适应能力
- 现实工程问题目标量级天然不统一（成本 vs 损耗 vs 工期）
- 是论文最重要的论证实验，审稿人认可度最高

**实现方案**：

```
在标准 DTLZ1-4 计算后，对目标向量乘以缩放因子：

  f_i' = f_i × scale_i

缩放方案（论文 Table IV）：
  方案A: scale = [1, 1, ..., 1]           （对照组，无缩放）
  方案B: scale = [1, 10, ..., 10]         （f1 正常，其余扩大 10 倍）
  方案C: scale = [10, 1, 10, 1, ...]      （奇偶交替，差 10 倍）
  方案D: scale = [1, 1, 10, 100, ...]     （逐目标递增，最大差 100 倍）
```

**实现步骤**：
1. `config.py` 增加 `SCALE_FACTORS` 参数（None 或 list）
2. `fitness_multi.py` 的 `eval_dtlz` 末尾乘以缩放因子
3. `metric_calc.py` 的 `get_dtlz_true_pf` 同样缩放参考 PF（否则 IGD 无意义）
4. `script/run.py` 增加 `--scale` 选项，遍历 scale 方案
5. `src/compare_algorithms.py` 增加对 scale 方案的分组对比

**预期结果**：
- NSGA3：不同缩放方案下 IGD 几乎不变（自适应归一化起作用）
- NSGA2/MOEAD：缩放越大 IGD 越差（尺度扭曲了拥挤度/权重语义）
- 对比图直观展示 NSGA3 的鲁棒性优势

---

### 🟡 P1 — V-D 凸帕累托前沿（次要，有空再做）

**论文依据**：Deb & Jain (2014) Section V-D "Convex Pareto-optimal Front"

**定位**：补齐测试场景，证明算法普适性

**实现方案**：

构造凸 PF 测试问题（论文 Equation 11）：

```
f_i = (1 + g) × (1 - cos(θ_1) × cos(θ_2) × ... × cos(θ_{M-2}) × sin(θ_{M-1}))
...
最后一项略有不同，产生凸前沿（而非 DTLZ2 的凹前沿）
```

或者在 DTLZ2 公式中将 `cos/sin` 替换为论文中的凸版本。

**需要改动的文件**：
- `test_dtlz.py` — 新增 `dtlz2_convex` 函数
- `metric_calc.py` — 新增对应的真实 PF 生成
- `fitness_multi.py` — 注册新问题

---

### ⚪ P2 — V-B 传统单目标对比（不实现）

**定位**：论文用 fmincon 串行标量化 vs 进化算法并行优化

**放弃理由**：
- 对比对象过时（串行数值优化 vs 并行进化优化，结论可预判）
- 已有 NSGA2 / MOEAD 双基线对比，审稿人完全认可
- 投入产出比极低

---

## 执行顺序

```
1. 实现 V-C 尺度缩放（预计改 3-4 个文件）
2. 跑实验：4 个 scale 方案 × 3 个算法 × 5 个 M × 4 个 DTLZ
3. 对比分析 → 作为核心章节图表
4. (可选) V-D 凸 PF 作为补充附录
```
