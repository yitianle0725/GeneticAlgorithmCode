# IE_MOEC 任务板 — 独立进化多目标极值组合算法

> **IE_MOEC** = **I**ndependently **E**volving **M**ulti-**O**bjective **E**xtremum **C**ombination
>
> 双层分治多目标进化算法：内层孤岛隔离搜索局部极值，外层极值重组拼接完整帕累托前沿

---

## 算法流程总览（6 阶段）

```
┌─────────────────────────────────────────────────┐
│  阶段①  初始化起源种群                            │
│  M 个目标维度，N 个初始个体（较小规模，N≈20）       │
│  可选：输入历史数据获取需求矩阵/链路信息             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  阶段②  孤岛分离与扩增                            │
│                                                   │
│  从当前种群中筛选各目标维度极值解（M 个）            │
│  解1(f1最优)  解2(f2最优)  ...  解M(fM最优)        │
│                                                   │
│  每个极值解作为一座孤岛的"祖先"                     │
│  各自通过局部加强搜索扩增为子种群                   │
│  （多次变异 / 局部搜索）                           │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  阶段③  孤岛独立演化（K = M 个岛并行）             │
│                                                   │
│  ┌────岛1────┐ ┌────岛2────┐   ┌────岛M────┐    │
│  │交叉(内部) │ │交叉(内部) │   │交叉(内部) │    │
│  │   ↓       │ │   ↓       │   │   ↓       │    │
│  │变异(多种  │ │变异(多种  │   │变异(多种  │    │
│  │ 策略)    │ │ 策略)    │   │ 策略)    │    │
│  │   ↓       │ │   ↓       │   │   ↓       │    │
│  │适应度计算 │ │适应度计算 │   │适应度计算 │    │
│  │(加权聚合)│ │(加权聚合)│   │(加权聚合)│    │
│  │   ↓       │ │   ↓       │   │   ↓       │    │
│  │精英保留   │ │精英保留   │   │精英保留   │    │
│  │   ↓       │ │   ↓       │   │   ↓       │    │
│  │选1个代表  │ │选1个代表  │   │选1个代表  │    │
│  └──────────┘ └──────────┘   └──────────┘    │
│                                                   │
│  岛间严格隔离：禁止交叉、禁止基因互通               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  阶段④  极值重组（跨岛基因交流）                   │
│                                                   │
│  K 个岛的精英个体混合 → 互相交叉(跨岛交叉)         │
│  → 产生混血子代 → 评估                            │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  阶段⑤  非支配排序 + 参考向量小生境筛选            │
│                                                   │
│  混血子代 → 非支配排序 → 前沿1/前沿2/前沿3         │
│  → NSGA-III 参考向量小生境选择                     │
│  → 选出下一轮各孤岛的起源者                        │
└──────────────────────┬──────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │  阶段⑥  判断收敛      │
           │  Yes → 输出 PF        │
           │  No  → 回到阶段②      │
           └───────────────────────┘
```

**外循环**：阶段②→③→④→⑤→⑥→②...（阶段①仅在最开始执行一次）

---

## 与现有代码的关系

| 现有模块 | 复用方式 |
|----------|----------|
| `individual.py` | ✅ 直接复用 Individual 类 |
| `test_dtlz.py` | ✅ 直接复用 DTLZ1~10 评估函数 |
| `fitness_multi.py` | ✅ 直接复用 DTLZ 评估 |
| `metric_calc.py` | ✅ 直接复用 IGD/GD/HV/SP/ONVG + 真实 PF |
| `visualizer.py` | ✅ 直接复用全部 7 种图表 |
| `config.py` | 🔧 扩展：新增 IE_MOEC 专用参数 |
| `moec_algorithm.py` | 🔧 抽离复用：SBX 交叉、多项式变异、非支配排序、NSGA-III 环境选择 |
| `script/run.py` | 🔧 扩展：新增 `--algo IEMOEC` 选项 |

**核心思路**：新建 `src/iemoec_algorithm.py`，复用现有基础设施，只实现 IE_MOEC 独有的双层搜索逻辑。

---

## 实施计划

### 🔴 Phase 1 — 基础架构（必做）

#### 任务 1.1：扩展配置 `config.py`

新增 IE_MOEC 专用参数类或扩展现有 `GAConfig`：

```python
# ──────── IE_MOEC 专用参数 ────────
self.N_ISLANDS = None          # 孤岛数量，None = 自动设为 NOBJ
self.ISLAND_POPSIZE = 20       # 每个孤岛的子种群大小
self.ISLAND_GENS = 50          # 岛内演化代数（每次外循环）
self.MAX_OUTER_GENS = 20       # 外循环最大代数
self.ISLAND_PXOVER = 0.9       # 岛内交叉概率
self.ISLAND_MUT_START = 0.1    # 岛内变异概率（较高，促进局部探索）
self.RECOMBINE_RATE = 0.8      # 极值重组交叉率
self.CONVERGENCE_SWITCH_RATIO = 0.6  # 前期加权聚合/后期Pareto切换点

# 前期加权聚合的权重生成策略
self.AGG_WEIGHT_MODE = "extreme"  # "extreme" / "uniform" / "random"
# extreme: 每个岛对其目标维度权重=1.0，其余=ε
# uniform: 均匀权重
# random: 每代随机生成
```

**文件**：[`src/config.py`](src/config.py)

---

#### 任务 1.2：创建孤岛类 `src/island.py`

```python
class Island:
    """单个孤岛 — 管理一个子种群的隔离演化"""
    
    def __init__(self, island_id, ancestor, objective_focus, config):
        self.id = island_id
        self.ancestor = ancestor        # 起源个体
        self.objective_focus = objective_focus  # 该岛专注的目标维度索引
        self.population = []            # 子种群
        self.elite = None               # 本轮产出精英
        self.history_best = None        # 历史最优
    
    def expand_from_ancestor(self, expansion_size):
        """从祖先个体扩增为子种群
        策略：对祖先进行 expansion_size 次变异，产生多样化后代
        """
    
    def evolve(self, generations, weight_mode):
        """岛内独立演化
        - 交叉：岛内个体间 SBX 交叉
        - 变异：多项式变异（可扩展多种策略）
        - 适应度：加权聚合（前期）/ 单目标（后期）
        - 精英保留：目标维度极值个体强制保留
        """
    
    def select_elite(self):
        """选出本岛代表个体（目标维度最优 + 考虑多样性）"""
    
    def update_ancestor(self, new_ancestor):
        """更新起源者（外循环结束后调用）"""
```

**文件**：新建 [`src/island.py`](src/island.py)

---

#### 任务 1.3：创建主算法类 `src/iemoec_algorithm.py`

```python
class IEMOEC:
    """IE_MOEC: 独立进化多目标极值组合算法
    
    双层架构：
      内层 — K 个孤岛各自独立演化，搜索局部极值
      外层 — 极值重组 + NSGA-III 筛选，拼接全局 PF
    """
    
    def __init__(self, problem_config):
        """初始化：问题类型(DTLZ/ZDT/MOP7)、目标维度、算法参数"""
    
    # ── 阶段① ──
    def initialize_origin_population(self):
        """初始化起源种群（小规模 N≈20）"""
    
    # ── 阶段② ──
    def island_separation_and_expansion(self):
        """孤岛分离：选出 M 个极值解 → 各自扩增为子种群"""
    
    # ── 阶段③ ──
    def island_parallel_evolution(self):
        """K 个孤岛并行独立演化（可用 multiprocessing）"""
    
    # ── 阶段④ ──
    def extreme_recombination(self):
        """极值重组：各岛精英跨岛交叉 → 混血子代池"""
    
    # ── 阶段⑤ ──
    def outer_selection(self):
        """外层选择：非支配排序 + NSGA-III 参考向量小生境 → 下一轮起源者"""
    
    # ── 阶段⑥ ──
    def check_convergence(self):
        """判断收敛：IGD 变化 < 阈值 或 外循环代数用尽"""
    
    # ── 辅助 ──
    def _weighted_aggregation_fitness(self, obj_vector, weights):
        """前期加权聚合适应度"""
    
    def _select_extremes(self, population, M):
        """从种群中选出各目标维度的极值个体"""
    
    def run(self):
        """外循环主入口 → 输出 PF + 指标曲线"""
```

**文件**：新建 [`src/iemoec_algorithm.py`](src/iemoec_algorithm.py)

**关键复用**（从 `moec_algorithm.py` 抽离或直接引用）：
- `_sbx_crossover()` → SBX 模拟二进制交叉
- `_randval()` / 多项式变异
- `fast_non_dominated_sort()` → 向量化非支配排序
- `nsga3_environment_selection()` → 外层筛选
- `normalize_population()` → ASF 自适应归一化
- `generate_reference_points()` → Das-Dennis 参考点

---

### 🟡 Phase 2 — 核心逻辑细化

#### 任务 2.1：孤岛扩增策略

从 1 个祖先扩增为 `ISLAND_POPSIZE` 个子代，需要多种策略：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **变异扩增** | 对祖先重复多项式变异，产生 `POPSIZE-1` 个变异后代 | 通用 |
| **局部采样** | 在祖先周围高斯采样 `ancestor + N(0, σ²)` | 连续变量 |
| **交叉扩增** | 如果已有多个个体，岛内交叉填充 | 后续外循环 |

**默认方案**：首轮用变异扩增，后续外循环用变异 + 上一轮遗留个体混合。

---

#### 任务 2.2：岛内适应度函数（两阶段）

**前期（加权聚合阶段，outer_gen / MAX_OUTER_GENS < SWITCH_RATIO）**：

```python
# 岛 i 专注目标维度 i
w = np.full(M, ε)    # ε = 1e-3 小权重
w[i] = 1.0           # 目标维度权重 = 1.0

# Tchebycheff 聚合（也可用加权和）
fitness = max(w[j] * abs(obj[j] - z_ideal[j]) for j in range(M))
```

**后期（Pareto 阶段）**：
- 岛内也用非支配排序 + 拥挤度/极值保留
- 但精英选择仍偏向本岛目标维度

---

#### 任务 2.3：极值重组交叉策略

K 个精英个体 → 产生足够多的混血子代（供 NSGA-III 筛选）：

```python
# 方案 A: 全配对交叉
for i in range(K):
    for j in range(i+1, K):
        offspring = sbx_crossover(elites[i], elites[j])
        pool.append(offspring)

# 方案 B: 每对产生多个子代
# 每个配对产生 n_offspring_per_pair 个子代

# 混血池大小 ≈ C(K,2) × 2，供 NSGA-III 筛选出 K 个新起源者
```

---

#### 任务 2.4：收敛策略切换

```
                    外循环进度
    0% ═══════════════╪══════════════════ 100%
                      │
    加权聚合适应度      │    非支配排序
    (单目标加速收敛)    │    (多目标精细筛选)
                      │
              SWITCH_RATIO = 0.6
```

切换点可配置，也可以基于 **IGD 停滞检测** 自动切换。

---

### ⚪ Phase 3 — 实验验证 & 对比

#### 任务 3.1：标准 DTLZ 测试

```
M = 3, 5, 8, 10, 15
问题: DTLZ1, DTLZ2, DTLZ3, DTLZ4
对比: IE_MOEC vs NSGA2 vs NSGA3 vs MOEAD
指标: IGD, GD, HV, SP, ONVG
```

#### 任务 3.2：V-C 尺度缩放测试

```
缩放方案: A, B, C, D（复用现有 --scale 机制）
预期: IE_MOEC 的极值分离 + ASF 归一化应对尺度差异
```

#### 任务 3.3：超参数敏感性分析

| 超参 | 测试范围 |
|------|----------|
| 孤岛数量 K | M/2, M, 2M |
| 岛种群大小 | 10, 20, 50 |
| 岛内迭代代数 | 20, 50, 100 |
| 外循环代数 | 10, 20, 50 |
| 切换比例 | 0.3, 0.5, 0.7 |

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新建** | `src/iemoec_algorithm.py` | IE_MOEC 主算法类（~400行） |
| **新建** | `src/island.py` | 孤岛类（~150行） |
| **修改** | `src/config.py` | 新增 IE_MOEC 参数块 |
| **修改** | `script/run.py` | 新增 `--algo IEMOEC` |
| **修改** | `script/run.sh` | ALGO 可选 IEMOEC |
| **可选** | `src/compare_algorithms.py` | 对比中包含 IEMOEC |
| **可选** | `tests/test_iemoec.py` | 单元测试 |

---

## 预期挑战 & 风险

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 孤岛数量 K 直接等于 NOBJ (M) | 🟡 | 高维时 K 很大但每岛个体少，需验证 M=15 是否可行 |
| 极值聚集：多个岛收敛到同一区域 | 🟡 | 加权聚合阶段每岛用不同权重向量，强制搜索不同方向 |
| PF 中部采样不足 | 🟡 | 重组阶段混血子代自然填充中部区域 |
| 外层 NSGA-III 参考点数量爆炸 | 🟢 | M≥10 时外层种群小（仅 K 个），参考点数量不是瓶颈 |
| 岛间负载不均 | 🟢 | 岛种群大小统一，迭代次数统一，负载自然均衡 |
| 与现有代码耦合 | 🟢 | 通过复用而非继承，保持独立性 |

---

## 执行顺序

```
Phase 1（基础架构）
  1.1 扩展 config.py       ← 先定参数
  1.2 新建 island.py       ← 孤岛类
  1.3 新建 iemoec_algorithm.py ← 主算法（复用 MultiObjMOECAbilene 组件）
  1.4 修改 run.py / run.sh  ← 一键启动

Phase 2（核心逻辑）
  2.1 孤岛扩增策略实现
  2.2 两阶段适应度切换
  2.3 极值重组实现
  2.4 收敛判定

Phase 3（实验验证）
  3.1 标准 DTLZ 对比
  3.2 V-C 尺度缩放测试
  3.3 超参数敏感性分析
```

---

## 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| K = ? | **K = M**（每目标一岛） | 图片指定"M个极值解"，每个专注一个目标维度 |
| 岛内选几个代表？ | **1 个** | 图片"每个岛选1个代表"，简化重组阶段 |
| 复用方式？ | **直接引用**（非继承） | 现有 `MultiObjMOECAbilene` 是单种群类，不适合继承；抽离静态方法更干净 |
| 前期适应度函数？ | **加权 Tchebycheff** | 与 MOEA/D 一致，单目标优化效率高 |
| 后期适应度函数？ | **非支配排序** | 与 NSGA-III 一致 |
| 并行实现？ | **先串行后并行** | Phase 1 串行验证正确性，Phase 2+ 加 `multiprocessing` |
