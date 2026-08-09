"""
孤岛模块 — IE_MOEC 算法的子种群隔离演化单元

每个孤岛专注一个目标维度，独立进行（μ+μ）演化策略搜索，
产出该目标维度上的极值个体。

同时提供 SBX 交叉和多项式变异算子（供外层重组复用）。
"""

import random
import numpy as np
from individual import Individual


# ===================== 进化算子（独立函数，供岛内 + 外层复用） =====================

def sbx_crossover(p1_gene, p2_gene, lower, upper, eta_c, nvars):
    """SBX 模拟二进制交叉

    参数：
        p1_gene, p2_gene: 父代基因
        lower, upper: 变量边界
        eta_c: SBX 分布指数（推荐 30）
        nvars: 变量维数

    返回：
        (c1_gene, c2_gene): 两个子代基因
    """
    c1, c2 = [0.0] * nvars, [0.0] * nvars
    for i in range(nvars):
        if random.random() <= 0.5:
            xi1, xi2 = p1_gene[i], p2_gene[i]
            if abs(xi1 - xi2) < 1e-10:
                c1[i], c2[i] = xi1, xi2
                continue
            x1 = min(xi1, xi2)
            x2 = max(xi1, xi2)
            r = random.random()
            if r < 0.5:
                beta = (2 * r) ** (1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - r))) ** (1 / (eta_c + 1))
            cc1 = 0.5 * ((x1 + x2) - beta * (x2 - x1))
            cc2 = 0.5 * ((x1 + x2) + beta * (x2 - x1))
            c1[i] = max(min(cc1, upper[i]), lower[i])
            c2[i] = max(min(cc2, upper[i]), lower[i])
        else:
            c1[i], c2[i] = p1_gene[i], p2_gene[i]
    return c1, c2


def polynomial_mutation(gene, lower, upper, mut_rate, eta_m, nvars):
    """多项式变异（返回新基因列表，不修改原基因）

    参数：
        gene: 原始基因
        lower, upper: 变量边界
        mut_rate: 每个变量的变异概率（论文推荐 1/n）
        eta_m: 多项式变异分布指数（推荐 20）
        nvars: 变量维数

    返回：
        mutated: 变异后的基因（新列表）
    """
    mutated = gene.copy()
    for i in range(nvars):
        if random.random() < mut_rate:
            y = mutated[i]
            low, high = lower[i], upper[i]
            delta1 = (y - low) / (high - low) if (high - low) > 1e-10 else 0.0
            delta2 = (high - y) / (high - low) if (high - low) > 1e-10 else 0.0
            r = random.random()
            if r <= 0.5:
                delta = (2 * r) ** (1 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1 / (eta_m + 1))
            y += delta * (high - low)
            mutated[i] = max(min(y, high), low)
    return mutated


# ===================== 孤岛类 =====================

class Island:
    """单个孤岛 — 管理一个子种群的隔离演化

    每个孤岛有一个「祖先」个体，祖先在某一目标维度上表现最优。
    孤岛通过扩增、独立演化、精英选择，产出该目标维度的极值解。

    属性：
        id: 孤岛编号
        ancestor: 起源个体（当前外循环轮次的祖先）
        objective_focus: 该岛专注的目标维度索引 (0 ~ M-1)
        config: GAConfig 配置对象
        popsize: 子种群大小
        population: 当前子种群列表
    """

    def __init__(self, island_id, ancestor, objective_focus, config, popsize=20,
                 weight_vector=None):
        self.id = island_id
        self.ancestor = ancestor
        self.objective_focus = objective_focus  # 可为 None（折中型孤岛）
        self.config = config
        self.popsize = popsize
        self.population = []
        # 自定义权重向量（折中型孤岛使用），None 则按 objective_focus 自动生成
        self.weight_vector = weight_vector
        # 岛类型标签
        if weight_vector is not None:
            self.is_balanced = True
        else:
            self.is_balanced = False

    # ── 阶段②：扩增 ──
    def expand_from_ancestor(self, eval_fn, expansion_mut_rate=0.15):
        """从祖先个体扩增为子种群

        策略：保留祖先 + 对祖先进行 (popsize-1) 次高变异率变异，
        产生差异化后代，覆盖祖先附近的搜索区域。

        参数：
            eval_fn: callable(ind) → 评估个体目标值（原地修改 ind.obj）
            expansion_mut_rate: 扩增阶段的变异率（较高，促进多样性）
        """
        self.population = [self.ancestor]
        nvars = self.config.NVARS
        for _ in range(self.popsize - 1):
            child = Individual(self.config)
            child.copy_from(self.ancestor)
            child.gene = polynomial_mutation(
                child.gene, child.lower, child.upper,
                expansion_mut_rate, self.config.ETA_M, nvars
            )
            eval_fn(child)
            self.population.append(child)

    # ── 阶段③：岛内一代演化 ──
    def evolve_generation(self, eval_fn, z_ideal, phase="aggregation"):
        """岛内（μ+μ）一代演化

        1. 随机配对所有个体
        2. SBX 交叉 + 多项式变异 → 产生 popsize 个子代
        3. 评估子代
        4. 父代 + 子代合并，按适应度选前 popsize 个

        参数：
            eval_fn: callable(ind) → 评估个体目标值
            z_ideal: 当前理想点 (M,) 数组
            phase: "aggregation"（加权聚合）或 "pareto"（非支配排序）
        """
        n = len(self.population)
        nvars = self.config.NVARS
        eta_c = self.config.ETA_C
        eta_m = self.config.ETA_M
        mut_rate = 1.0 / nvars  # 论文标准：p_m = 1/n

        # 1. 产生子代
        indices = list(range(n))
        random.shuffle(indices)
        offspring = []
        for k in range(0, n, 2):
            if k + 1 >= n:
                break
            p1 = self.population[indices[k]]
            p2 = self.population[indices[k + 1]]

            c1 = Individual(self.config)
            c2 = Individual(self.config)
            c1.copy_from(p1)
            c2.copy_from(p2)

            if random.random() < self.config.PXOVER:
                c1.gene, c2.gene = sbx_crossover(
                    p1.gene, p2.gene, p1.lower, p1.upper, eta_c, nvars
                )

            c1.gene = polynomial_mutation(
                c1.gene, c1.lower, c1.upper, mut_rate, eta_m, nvars
            )
            c2.gene = polynomial_mutation(
                c2.gene, c2.lower, c2.upper, mut_rate, eta_m, nvars
            )

            eval_fn(c1)
            eval_fn(c2)
            offspring.extend([c1, c2])

        # 补齐奇数情况
        if len(offspring) < n:
            extra = Individual(self.config)
            extra.copy_from(self.population[-1])
            extra.gene = polynomial_mutation(
                extra.gene, extra.lower, extra.upper, mut_rate, eta_m, nvars
            )
            eval_fn(extra)
            offspring.append(extra)

        # 2. 合并 → 选择
        merged = self.population + offspring

        if phase == "aggregation":
            self._select_by_aggregation(merged, z_ideal)
        else:
            self._select_by_pareto(merged)

    def _select_by_aggregation(self, merged, z_ideal):
        """加权 Tchebycheff 聚合选择

        极端岛：权重偏置本岛目标维度，聚焦极值。
        折中岛：使用自定义权重向量，搜索 trade-off 区域。
        """
        M = self.config.NOBJ
        if self.weight_vector is not None:
            w = self.weight_vector.copy()
        elif self.objective_focus is not None:
            w = np.full(M, self.config.AGG_EPSILON)
            w[self.objective_focus] = 1.0
        else:
            w = np.ones(M) / M  # fallback: 均匀权重

        # 计算每个个体的 Tchebycheff 值（越小越好）
        scores = []
        for ind in merged:
            obj = np.array(ind.obj)
            tch = np.max(w * np.abs(obj - z_ideal))
            scores.append(tch)

        # 按得分排序，取前 popsize
        sorted_pairs = sorted(zip(scores, merged), key=lambda x: x[0])
        self.population = [ind for _, ind in sorted_pairs[:self.popsize]]

    def _select_by_pareto(self, merged):
        """Pareto 阶段选择：非支配排序 + 方向偏置

        逐前沿选取，最后前沿按本岛偏好截断。
        极端岛按目标维度值排序，折中岛按 Tchebycheff 值排序。
        """
        from iemoec_algorithm import fast_non_dominated_sort

        M = self.config.NOBJ
        fronts = fast_non_dominated_sort(merged, M)
        new_pop = []
        k = 0
        while k < len(fronts) and len(new_pop) + len(fronts[k]) <= self.popsize:
            new_pop.extend(fronts[k])
            k += 1

        remain = self.popsize - len(new_pop)
        if remain > 0 and k < len(fronts):
            last_front = fronts[k]
            if self.is_balanced or self.objective_focus is None:
                # 折中岛：按 Tchebycheff 值排序
                w = self.weight_vector if self.weight_vector is not None else np.ones(M) / M
                # 需要 z_ideal，这里用前沿最小值
                F_last = np.array([[ind.obj[m] for m in range(M)] for ind in last_front])
                z_ideal_local = np.min(F_last, axis=0)
                scored = []
                for ind in last_front:
                    obj = np.array(ind.obj)
                    tch = np.max(w * np.abs(obj - z_ideal_local))
                    scored.append((tch, ind))
                scored.sort(key=lambda x: x[0])
                new_pop.extend([ind for _, ind in scored[:remain]])
            else:
                # 极端岛：按本岛目标维度值排序（越小越好）
                last_front_sorted = sorted(
                    last_front, key=lambda ind: ind.obj[self.objective_focus]
                )
                new_pop.extend(last_front_sorted[:remain])

        self.population = new_pop

    # ── 精英选择 ──
    def select_elite(self):
        """选出本岛代表个体（单精英，向后兼容）"""
        if self.is_balanced or self.objective_focus is None:
            # 折中岛：选 Tchebycheff 值最小的
            return self.select_elites(n=1)[0]
        else:
            return min(self.population,
                       key=lambda ind: ind.obj[self.objective_focus])

    def select_elites(self, n=3):
        """选出本岛 n 个代表性个体（多精英输出）

        策略：
          1. 目标维度最优个体（极端岛）或 Tchebycheff 最优（折中岛）
          2. 非支配前沿中拥挤度最大的个体（多样性代表）
          3. 随机扰动个体（避免全部精英同质化）
        """
        if len(self.population) <= n:
            return list(self.population)

        elites = []
        remaining = list(self.population)

        # 精英 1：偏好方向最优
        if self.is_balanced or self.objective_focus is None:
            M = self.config.NOBJ
            w = self.weight_vector if self.weight_vector is not None else np.ones(M) / M
            F_local = np.array([[ind.obj[m] for m in range(M)] for ind in remaining])
            z_local = np.min(F_local, axis=0)
            best = min(remaining, key=lambda ind:
                       np.max(w * np.abs(np.array(ind.obj) - z_local)))
        else:
            best = min(remaining, key=lambda ind: ind.obj[self.objective_focus])
        elites.append(best)
        remaining.remove(best)
        if len(elites) >= n:
            return elites

        # 精英 2：拥挤度最大的个体（多样性）
        self._assign_crowding(remaining)
        best_crowd = max(remaining, key=lambda ind: ind.crowd_dist)
        elites.append(best_crowd)
        remaining.remove(best_crowd)
        if len(elites) >= n:
            return elites

        # 精英 3：随机选取（从前 50% 中随机）
        if len(remaining) > 0:
            top_half = sorted(remaining,
                             key=lambda ind: (ind.rank, -ind.crowd_dist))[:max(1, len(remaining)//2)]
            elites.append(random.choice(top_half))

        return elites

    def _assign_crowding(self, pop):
        """为种群分配拥挤度距离"""
        M = self.config.NOBJ
        for ind in pop:
            ind.crowd_dist = 0.0
        if len(pop) <= 2:
            for ind in pop:
                ind.crowd_dist = float('inf')
            return
        for m in range(M):
            sorted_pop = sorted(pop, key=lambda x: x.obj[m])
            sorted_pop[0].crowd_dist = float('inf')
            sorted_pop[-1].crowd_dist = float('inf')
            obj_min = sorted_pop[0].obj[m]
            obj_max = sorted_pop[-1].obj[m]
            if abs(obj_max - obj_min) < 1e-10:
                continue
            for i in range(1, len(pop) - 1):
                dist = (sorted_pop[i + 1].obj[m] -
                        sorted_pop[i - 1].obj[m]) / (obj_max - obj_min)
                sorted_pop[i].crowd_dist += dist

    def update_ancestor(self, new_ancestor):
        """更新祖先（外循环结束后由 IEMOEC 调用）"""
        self.ancestor = new_ancestor
