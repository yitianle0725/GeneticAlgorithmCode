"""
IE_MOEC — 独立进化多目标极值组合算法

IEMOEC = Independently Evolving Multi-Objective Extremum Combination

双层分治架构：
  内层 — K 个孤岛各自隔离演化，搜索各目标维度的局部极值
  外层 — 极值重组 + NSGA-III 参考向量小生境筛选，拼接完整帕累托前沿

参考：
  - Deb & Jain (2014) "NSGA-III, Part I"
  - 对比.md — 独立进化 MOEC vs NSGA-III 对比分析
  - picture/独立进化算法.png — 算法流程图
"""

import math
import random
import os
import time
import datetime
import numpy as np

from config import GAConfig
from individual import Individual
from fitness_multi import eval_dtlz
from metric_calc import get_dtlz_true_pf, calc_all_metrics
from visualizer import GAVisualizer
from path_tool import get_project_root
from island import Island, sbx_crossover, polynomial_mutation


# ===================== 独立进化算子函数 =====================

def fast_non_dominated_sort(population, nobj):
    """向量化快速非支配排序（独立函数版）

    参数：
        population: 个体列表
        nobj: 目标维度数

    返回：
        fronts: 列表的列表，fronts[0] 为 Pareto 最优前沿
    """
    n = len(population)
    if n == 0:
        return []

    F = np.array([[ind.obj[m] for m in range(nobj)] for ind in population])

    # 支配矩阵：dom[i,j] = True 表示 i 支配 j
    le_all = np.ones((n, n), dtype=bool)
    lt_any = np.zeros((n, n), dtype=bool)
    for m in range(nobj):
        le_all &= (F[:, None, m] <= F[None, :, m])
        lt_any |= (F[:, None, m] < F[None, :, m])
    dom_matrix = le_all & lt_any

    dom_count = np.sum(dom_matrix, axis=0).astype(np.int32)
    dominates_whom = [list(np.where(dom_matrix[i])[0]) for i in range(n)]

    fronts = []
    assigned = np.zeros(n, dtype=bool)
    remaining = n
    while remaining > 0:
        current_front = np.where((dom_count == 0) & (~assigned))[0]
        if len(current_front) == 0:
            break
        assigned[current_front] = True
        remaining -= len(current_front)
        for i in current_front:
            for j in dominates_whom[i]:
                if not assigned[j]:
                    dom_count[j] -= 1
        fronts.append([population[i] for i in current_front])

    # 设置 rank
    for fi, front in enumerate(fronts):
        for ind in front:
            ind.rank = fi
    return fronts


def generate_das_dennis_points(M, p):
    """生成 Das-Dennis 结构化参考点

    参数：
        M: 目标维度
        p: 每维度划分份数 (REF_DIV)

    返回：
        ref_points: (H, M) 数组，H = C(M+p-1, p)
    """
    import itertools
    combinations = list(itertools.combinations(range(M + p - 1), M - 1))
    ref = np.zeros((len(combinations), M))
    for idx, comb in enumerate(combinations):
        prev = -1
        for i in range(M - 1):
            ref[idx, i] = comb[i] - prev - 1
            prev = comb[i]
        ref[idx, -1] = M + p - 2 - prev
        ref[idx] /= p
    return ref


def asf_normalize(population, nobj):
    """ASF 自适应归一化（独立函数版）

    步骤：
      1. 计算理想点 z_min，平移目标空间
      2. 对每个目标轴用 ASF 找到极值点
      3. 构建超平面求截距
      4. 用截距归一化

    参数：
        population: 个体列表
        nobj: 目标维度数

    返回：
        F_norm: (N, M) 归一化目标矩阵
        z_min: (M,) 理想点
    """
    M = nobj
    F = np.array([ind.obj for ind in population])
    z_min = np.min(F, axis=0)
    F_shift = F - z_min

    # ASF 极值点搜索
    extreme_points = np.zeros((M, M))
    for m in range(M):
        w = np.full(M, 1e-6)
        w[m] = 1.0
        asf_vals = np.max(F_shift / w, axis=1)
        extreme_idx = np.argmin(asf_vals)
        extreme_points[m] = F_shift[extreme_idx]

    # 超平面截距
    try:
        a = np.linalg.solve(extreme_points, np.ones(M))
        intercept = 1.0 / a
    except np.linalg.LinAlgError:
        intercept = np.max(F_shift, axis=0)

    intercept = np.maximum(intercept, np.max(F_shift, axis=0))
    intercept = np.maximum(intercept, 1e-10)

    scale = intercept.copy()
    scale[scale < 1e-10] = 1.0
    F_norm = F_shift / scale
    return F_norm, z_min


def nsga3_selection(merge_pop, n_select, nobj, ref_div, ref_points=None):
    """NSGA-III 环境选择（独立函数版）

    用于 IE_MOEC 外层筛选：从混血子代 + 原起源种群中选出下一轮起源者。

    参数：
        merge_pop: 合并种群
        n_select: 需要选出的个体数
        nobj: 目标维度数
        ref_div: Das-Dennis 划分参数
        ref_points: 预计算参考点（可选）

    返回：
        selected: 选中的 n_select 个个体
        ref_points: 参考点数组
    """
    fronts = fast_non_dominated_sort(merge_pop, nobj)

    # 逐前沿选取
    selected = []
    k = 0
    while k < len(fronts) and len(selected) + len(fronts[k]) <= n_select:
        selected.extend(fronts[k])
        k += 1

    remain = n_select - len(selected)
    if remain == 0:
        return selected, ref_points

    last_front = fronts[k]

    # 生成参考点
    if ref_points is None:
        ref_points = generate_das_dennis_points(nobj, ref_div)
    R = len(ref_points)

    # 归一化
    F_norm_all, _ = asf_normalize(merge_pop, nobj)
    merge_idx_map = {id(ind): i for i, ind in enumerate(merge_pop)}

    # ── 辅助：垂直距离 ──
    def _perp_dist(F_sub, ref_pts):
        dots = np.dot(F_sub, ref_pts.T)
        proj = dots[:, :, None] * ref_pts[None, :, :]
        diff = F_sub[:, None, :] - proj
        return np.linalg.norm(diff, axis=2)

    # ── 1. 统计已接受个体的小生境计数 ρ ──
    rho = np.zeros(R, dtype=int)
    if len(selected) > 0:
        sel_indices = [merge_idx_map[id(ind)] for ind in selected]
        F_sel = F_norm_all[sel_indices]
        dist_sel = _perp_dist(F_sel, ref_points)
        nearest_sel = np.argmin(dist_sel, axis=1)
        for r in nearest_sel:
            rho[int(r)] += 1

    # ── 2. 关联最后前沿 ──
    last_indices = [merge_idx_map[id(ind)] for ind in last_front]
    F_last = F_norm_all[last_indices]
    dist_last = _perp_dist(F_last, ref_points)
    nearest_last = np.argmin(dist_last, axis=1)

    niche_dict = {r: [] for r in range(R)}
    for local_i, ind in enumerate(last_front):
        r = int(nearest_last[local_i])
        niche_dict[r].append((local_i, ind))

    # ── 3. 小生境补选 ──
    exhausted = set()
    while len(selected) < n_select:
        candidates = [r for r in range(R)
                      if r not in exhausted and len(niche_dict[r]) > 0]
        if not candidates:
            # 所有方向耗尽，从最后前沿随机补
            remaining_in_last = [ind for ind in last_front
                                 if ind not in selected]
            needed = n_select - len(selected)
            selected.extend(remaining_in_last[:needed])
            break

        min_rho = min(rho[r] for r in candidates)
        ref_candidates = [r for r in candidates if rho[r] == min_rho]
        pick_r = random.choice(ref_candidates)

        ind_list = niche_dict[pick_r]
        best_item = min(ind_list, key=lambda x: dist_last[x[0], pick_r])
        sel_ind = best_item[1]
        selected.append(sel_ind)
        ind_list.remove(best_item)
        rho[pick_r] += 1
        if len(ind_list) == 0:
            exhausted.add(pick_r)

    return selected, ref_points


# ===================== 主算法类 =====================

class IEMOEC:
    """IE_MOEC: 独立进化多目标极值组合算法

    双层架构：
      内层 — K 个孤岛各自独立演化（阶段②③）
      外层 — 极值重组 + NSGA-III 筛选（阶段④⑤）
      迭代至收敛（阶段⑥）

    使用方式：
      algo = IEMOEC(dt_id=2, dtlz_M=8, scale_scheme="B")
      algo.run()
    """

    def __init__(self, dt_id=None, dtlz_M=3, scale_scheme=None,
                 root_output_dir=None, seed=123):
        # ── 问题配置 ──
        self.dt_id = dt_id
        self.dtlz_M = dtlz_M
        self.scale_scheme = scale_scheme if scale_scheme else "A"

        # ── 创建配置 ──
        self.config = GAConfig(nobj=dtlz_M, scale_scheme=scale_scheme)

        # ── IE_MOEC 参数（可从 config 读取，也可直接覆盖） ──
        self.n_origin = self.config.N_ORIGIN or self.config.POPSIZE  # None → 自动匹配 NSGA3 规模
        self.n_islands = self.config.N_ISLANDS or (2 * dtlz_M)       # K = 2M（M 极端 + M 折中）
        self.island_popsize = self.config.ISLAND_POPSIZE
        self.island_gens_early = self.config.ISLAND_GENS_EARLY
        self.island_gens_late = self.config.ISLAND_GENS_LATE
        self.max_outer_gens = self.config.MAX_OUTER_GENS
        self.switch_ratio = self.config.SWITCH_RATIO
        self.elites_per_island = self.config.ELITES_PER_ISLAND
        self.pf_expand_ratio = self.config.PF_EXPAND_RATIO

        # ── 问题初始化 ──
        self.config.VAR_LB = 0.0
        self.config.VAR_UB = 1.0
        self.config.NVARS = dtlz_M + 19  # DTLZ 标准: n = M + k, k=19

        # ── 参考 PF ──
        self.ref_pf = get_dtlz_true_pf(dt_id, M=dtlz_M, num_samples=1000)
        if self.scale_scheme != "A":
            self.ref_pf = self.ref_pf * self.config.scale_factors

        # ── NSGA-III 参考点缓存 ──
        self.ref_points = None

        # ── 状态 ──
        self.origin_population = []
        self.islands = []
        self.z_ideal = None  # 全局理想点
        self.outer_gen = 0

        # ── 记录 ──
        self.gen_record = []
        self.igd_record = []
        self.gd_record = []
        self.hv_record = []
        self.sp_record = []
        self.onvg_record = []

        # ── 早停 ──
        self.last_igd = None
        self.no_improve_cnt = 0

        # ── 输出 ──
        self.root_output_dir = root_output_dir
        self.log_file = None

        # ── 随机种子 ──
        random.seed(seed)
        np.random.seed(seed)

    # ===================== 阶段①：初始化起源种群 =====================
    def initialize_origin_population(self):
        """初始化起源种群（小规模，n_origin ≈ 20）

        随机生成决策变量，评估目标值。
        """
        self.origin_population = [
            Individual(self.config) for _ in range(self.n_origin)
        ]
        for ind in self.origin_population:
            for var_idx in range(self.config.NVARS):
                ind.gene[var_idx] = random.random()
        self._evaluate_pop(self.origin_population)
        # 初始非支配排序
        fast_non_dominated_sort(self.origin_population, self.config.NOBJ)

    # ===================== 阶段②：孤岛分离与扩增 =====================
    def island_separation_and_expansion(self):
        """从当前起源种群中选出极值解，各自扩增为孤岛子种群

        孤岛类型：
          - M 个**极端岛**：每目标维度选最优者，专注极值搜索
          - M 个**折中岛**：使用 Dirichlet 随机权重向量，搜索 trade-off 区域

        每个祖先通过高变异率扩增为大小为 island_popsize 的子种群。
        总计 n_islands = 2M 个孤岛。
        """
        M = self.config.NOBJ
        ancestors = []
        used = set()

        # ── 极端岛祖先：每目标维度选最优个体 ──
        for m in range(M):
            best_idx = None
            best_val = float('inf')
            for i, ind in enumerate(self.origin_population):
                if i in used:
                    continue
                if ind.obj[m] < best_val:
                    best_val = ind.obj[m]
                    best_idx = i
            if best_idx is not None:
                ancestors.append(('extreme', m, self.origin_population[best_idx]))
                used.add(best_idx)

        # 不够 M 个时从未选个体补
        if len(ancestors) < M:
            for i, ind in enumerate(self.origin_population):
                if i not in used and len(ancestors) < M:
                    ancestors.append(('extreme', len(ancestors), ind))
                    used.add(i)

        # ── 折中岛祖先：使用 Dirichlet 权重向量 ──
        # 每个折中岛有不同的权重分布，覆盖 PF 的不同 trade-off 区域
        balanced_ancestors = []
        for m in range(M):
            # 生成 Dirichlet 随机权重向量
            w = np.random.dirichlet(np.ones(M))
            # 从起源种群非支配前沿中选与权重最匹配的个体
            front0 = [ind for ind in self.origin_population if ind.rank == 0]
            if not front0:
                front0 = self.origin_population
            # 使用 Tchebycheff 匹配度选择折中岛祖先
            F_all = np.array([ind.obj for ind in front0])
            z_local = np.min(F_all, axis=0)
            best_balanced = min(front0, key=lambda ind:
                                np.max(w * np.abs(np.array(ind.obj) - z_local)))
            balanced_ancestors.append(('balanced', m, best_balanced, w))

        # 合并祖先：先极端岛，后折中岛
        all_ancestors = ancestors + [
            (a[0], a[1], a[2]) for a in balanced_ancestors
        ]
        all_weights = [None] * len(ancestors) + [a[3] for a in balanced_ancestors]

        # 确保有 n_islands 个岛（截断或补齐）
        if len(all_ancestors) > self.n_islands:
            all_ancestors = all_ancestors[:self.n_islands]
            all_weights = all_weights[:self.n_islands]
        while len(all_ancestors) < self.n_islands:
            idx = random.randrange(len(self.origin_population))
            all_ancestors.append(('extra', len(all_ancestors) % M,
                                  self.origin_population[idx]))
            all_weights.append(None)

        # 创建孤岛并扩增
        self.islands = []
        for i, ((island_type, obj_focus, ancestor), w) in enumerate(
                zip(all_ancestors, all_weights)):
            island = Island(
                island_id=i,
                ancestor=ancestor,
                objective_focus=obj_focus if island_type == 'extreme' else None,
                config=self.config,
                popsize=self.island_popsize,
                weight_vector=w
            )
            island.expand_from_ancestor(eval_fn=self._evaluate_single)
            self.islands.append(island)

    # ===================== 阶段③：孤岛独立演化 =====================
    def island_parallel_evolution(self):
        """K 个孤岛并行独立演化

        每个孤岛运行（μ+μ）演化策略。
        前期（聚合阶段）用 island_gens_early 代 + Tchebycheff 聚合选优，
        后期（Pareto 阶段）用 island_gens_late 代 + 非支配排序选优。
        """
        # 判断阶段
        progress = self.outer_gen / max(1, self.max_outer_gens)
        phase = "aggregation" if progress < self.switch_ratio else "pareto"

        # 根据阶段选择演化代数
        gens = self.island_gens_early if phase == "aggregation" else self.island_gens_late

        # 更新全局理想点（所有孤岛共享）
        all_obj = []
        for island in self.islands:
            for ind in island.population:
                all_obj.append(ind.obj)
        if all_obj:
            self.z_ideal = np.min(np.array(all_obj), axis=0)
        else:
            self.z_ideal = np.zeros(self.config.NOBJ)

        # 各岛独立演化
        for island in self.islands:
            for _ in range(gens):
                island.evolve_generation(
                    eval_fn=self._evaluate_single,
                    z_ideal=self.z_ideal,
                    phase=phase
                )

    # ===================== 阶段④：极值重组 =====================
    def extreme_recombination(self):
        """极值重组：每岛多精英跨岛交叉，产生混血子代

        每岛产出 ELITES_PER_ISLAND 个精英，任意两精英进行 SBX 交叉 + 变异，
        混血池大小 ≈ C(K × ELITES_PER_ISLAND, 2) × 2。
        多精英策略防止单一代表导致搜索方向过早收敛。
        """
        # 每岛多精英收集
        all_elites = []
        for island in self.islands:
            elites = island.select_elites(n=self.elites_per_island)
            all_elites.extend(elites)

        K = len(all_elites)
        nvars = self.config.NVARS
        mut_rate = 1.0 / nvars

        offspring = []
        for i in range(K):
            for j in range(i + 1, K):
                if random.random() < self.config.RECOMBINE_PXOVER:
                    g1, g2 = sbx_crossover(
                        all_elites[i].gene, all_elites[j].gene,
                        all_elites[i].lower, all_elites[i].upper,
                        self.config.ETA_C, nvars
                    )
                else:
                    g1, g2 = all_elites[i].gene.copy(), all_elites[j].gene.copy()

                for g in [g1, g2]:
                    g = polynomial_mutation(
                        g, all_elites[0].lower, all_elites[0].upper,
                        mut_rate, self.config.ETA_M, nvars
                    )
                    child = Individual(self.config)
                    child.lower = all_elites[0].lower.copy()
                    child.upper = all_elites[0].upper.copy()
                    child.gene = g
                    self._evaluate_single(child)
                    offspring.append(child)

        return offspring

    # ===================== 阶段⑤：外层 NSGA-III 筛选 =====================
    def outer_selection(self, offspring):
        """NSGA-III 环境选择：从（起源种群 + 混血子代）选出下一轮起源者"""
        merge_pop = self.origin_population + offspring

        selected, self.ref_points = nsga3_selection(
            merge_pop=merge_pop,
            n_select=self.n_origin,
            nobj=self.config.NOBJ,
            ref_div=self.config.REF_DIV,
            ref_points=self.ref_points
        )

        # 更新个体的 rank 和 crowd_dist（为 report 提供）
        fast_non_dominated_sort(selected, self.config.NOBJ)
        # 计算拥挤度
        self._calc_crowding(selected)

        return selected

    def _calc_crowding(self, pop):
        """计算拥挤度距离（外层选择后用于日志）"""
        for ind in pop:
            ind.crowd_dist = 0.0
        if len(pop) <= 2:
            return
        for m in range(self.config.NOBJ):
            sorted_pop = sorted(pop, key=lambda x: x.obj[m])
            sorted_pop[0].crowd_dist = float("inf")
            sorted_pop[-1].crowd_dist = float("inf")
            obj_min = sorted_pop[0].obj[m]
            obj_max = sorted_pop[-1].obj[m]
            if abs(obj_max - obj_min) < 1e-10:
                continue
            for i in range(1, len(pop) - 1):
                dist = (sorted_pop[i + 1].obj[m] -
                        sorted_pop[i - 1].obj[m]) / (obj_max - obj_min)
                sorted_pop[i].crowd_dist += dist

    # ===================== 阶段⑥：收敛判定 =====================
    def check_convergence(self):
        """判断是否收敛

        条件（满足任一即停止）：
          1. 外循环代数达到上限
          2. IGD 连续多代无明显改善
        """
        if self.outer_gen >= self.max_outer_gens:
            return True

        # IGD 停滞检测
        if len(self.igd_record) >= 2:
            current_igd = self.igd_record[-1]
            if self.last_igd is None:
                self.last_igd = current_igd
                self.no_improve_cnt = 0
            else:
                if abs(current_igd - self.last_igd) < self.config.HV_TOL:
                    self.no_improve_cnt += 1
                else:
                    self.no_improve_cnt = 0
                    self.last_igd = current_igd
                if self.no_improve_cnt >= self.config.EARLY_STOP_PATIENCE:
                    return True
        return False

    # ===================== PF 扩展 =====================
    def _pf_expansion(self):
        """收敛后 PF 扩展：从起源种群扩展至 POPSIZE 规模

        策略：
          1. 对起源种群中每个个体进行多次变异，产生 PF_EXPAND_RATIO 倍的候选解
          2. 候选解 + 原起源种群合并
          3. NSGA-III 筛选出 POPSIZE 个解作为最终 PF

        这样在保留极值搜索方向的同时，填充了 PF 中间区域。
        """
        M = self.config.NOBJ
        nvars = self.config.NVARS
        mut_rate = 1.0 / nvars
        target_total = int(self.config.POPSIZE * self.pf_expand_ratio)

        # 生成候选解
        candidates = list(self.origin_population)
        n_per_origin = max(1, (target_total - len(candidates)) // max(1, len(self.origin_population)))

        for origin_ind in self.origin_population:
            for _ in range(n_per_origin):
                if len(candidates) >= target_total:
                    break
                child = Individual(self.config)
                child.copy_from(origin_ind)
                # 较大变异率以探索 PF 间隙区域
                child.gene = polynomial_mutation(
                    child.gene, child.lower, child.upper,
                    mut_rate * 2.0,  # 双倍变异率增强探索
                    self.config.ETA_M, nvars
                )
                self._evaluate_single(child)
                candidates.append(child)

        # 合并后 NSGA-III 筛选
        selected, _ = nsga3_selection(
            merge_pop=candidates,
            n_select=self.config.POPSIZE,
            nobj=M,
            ref_div=self.config.REF_DIV,
            ref_points=self.ref_points
        )

        self.origin_population = selected
        fast_non_dominated_sort(self.origin_population, M)
        self._calc_crowding(self.origin_population)
        print(f"  PF 扩展完成: {len(selected)} 个解")

    # ===================== 评估辅助 =====================
    def _evaluate_single(self, ind):
        """评估单个个体（支持 V-C 尺度缩放）"""
        eval_dtlz(ind, dt_id=self.dt_id, M=self.dtlz_M)
        if self.scale_scheme != "A":
            for m in range(self.config.NOBJ):
                ind.obj[m] *= self.config.scale_factors[m]

    def _evaluate_pop(self, pop):
        """评估种群"""
        for ind in pop:
            self._evaluate_single(ind)

    # ===================== 主循环 =====================
    def run(self):
        """IE_MOEC 主入口 — 执行完整 6 阶段循环"""
        # ── 设置输出目录 ──
        project_root = get_project_root()
        base_output_dir = os.path.join(project_root, "output")

        dir_name = f"DTLZ{self.dt_id}_M{self.dtlz_M}_IEMOEC"
        if self.scale_scheme != "A":
            dir_name += f"_Scale{self.scale_scheme}"

        if self.root_output_dir is not None:
            run_output_dir = os.path.join(self.root_output_dir, dir_name)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sub_dir = f"{dir_name}_{timestamp}"
            run_output_dir = os.path.join(base_output_dir, sub_dir)

        os.makedirs(run_output_dir, exist_ok=True)

        # ── 日志文件 ──
        log_path = os.path.join(run_output_dir, "ga_log.txt")
        self.log_file = open(log_path, "w", encoding="utf-8")
        M = self.config.NOBJ
        header_parts = ["Gen | IGD | GD | HV | SP | ONVG"]
        for i in range(1, M + 1):
            header_parts.append(f"MinF{i}")
        self.log_file.write(" ".join(header_parts) + "\n")
        self.log_file.write("-" * 130 + "\n")

        # ── 阶段①：初始化 ──
        self.initialize_origin_population()
        # 初始化 z_ideal
        F_init = np.array([ind.obj for ind in self.origin_population])
        self.z_ideal = np.min(F_init, axis=0)

        t_start = time.time()
        print(f"\n{'='*60}")
        print(f"  IE_MOEC | DTLZ{self.dt_id} | M={self.dtlz_M}")
        if self.scale_scheme != "A":
            print(f"  Scale: {self.scale_scheme}")
        print(f"  起源种群={self.n_origin} | 孤岛数={self.n_islands}")
        print(f"  岛种群={self.island_popsize} | 岛代数={self.island_gens_early}(早期)/{self.island_gens_late}(后期)")
        print(f"  外循环上限={self.max_outer_gens} | 精英/岛={self.elites_per_island}")
        print(f"{'='*60}")

        # ── 外循环 ──
        for self.outer_gen in range(self.max_outer_gens + 1):
            # 报告当前状态
            self._report(run_output_dir)

            if self.outer_gen >= self.max_outer_gens:
                break

            # 阶段②：孤岛分离与扩增
            self.island_separation_and_expansion()

            # 阶段③：孤岛独立演化
            self.island_parallel_evolution()

            # 阶段④：极值重组
            offspring = self.extreme_recombination()

            # 阶段⑤：外层 NSGA-III 筛选
            self.origin_population = self.outer_selection(offspring)

            # 更新 z_ideal
            F_cur = np.array([ind.obj for ind in self.origin_population])
            cur_min = np.min(F_cur, axis=0)
            self.z_ideal = np.minimum(self.z_ideal, cur_min)

            # 阶段⑥：收敛判定
            if self.check_convergence():
                print(f"\n[收敛] 外循环第 {self.outer_gen} 代停止")
                break

        # ── PF 扩展：收敛后扩展/填充 PF 间隙 ──
        print(f"\n[PF扩展] 收敛后填充 PF 间隙 → {self.config.POPSIZE} 个解")
        self._pf_expansion()

        # ── 最终报告 ──
        self._report(run_output_dir)

        # ── 可视化 ──
        self._generate_plots(run_output_dir)

        self.log_file.close()
        elapsed = time.time() - t_start
        elapsed_str = f"{int(elapsed // 60)}m{elapsed % 60:.1f}s" if elapsed >= 60 else f"{elapsed:.1f}s"
        print(f"\n[OK] 结果保存在：{run_output_dir}")
        print(f"[DONE] 完成 {dir_name} | 耗时: {elapsed_str}")
        print(f"最终 IGD  = {self.igd_record[-1]:.6f}" if self.igd_record else "")
        print(f"最终 GD   = {self.gd_record[-1]:.6f}" if self.gd_record else "")
        print(f"最终 HV   = {self.hv_record[-1]:.4f}" if self.hv_record else "")
        print(f"最终 SP   = {self.sp_record[-1]:.4f}" if self.sp_record else "")
        print(f"最终 ONVG = {self.onvg_record[-1]}" if self.onvg_record else "")
        print("=" * 60, "\n")

    # ===================== 日志 & 报告 =====================
    def _report(self, run_output_dir):
        """生成一代报告：计算指标、写入日志、打印终端"""
        compute_hv = (self.outer_gen % max(1, self.config.HV_CALC_INTERVAL) == 0)

        # 对 origin_population 做一次非支配排序，确保 rank 正确
        fast_non_dominated_sort(self.origin_population, self.config.NOBJ)
        self._calc_crowding(self.origin_population)

        igd_val, gd_val, current_hv, current_sp, onvg_val = calc_all_metrics(
            self.origin_population, self.ref_pf, compute_hv=compute_hv
        )

        if not compute_hv and len(self.hv_record) > 0:
            current_hv = self.hv_record[-1]

        self.gen_record.append(self.outer_gen)
        self.igd_record.append(igd_val)
        self.gd_record.append(gd_val)
        self.hv_record.append(current_hv)
        self.sp_record.append(current_sp)
        self.onvg_record.append(onvg_val)

        # 各目标最小值
        front0 = [ind for ind in self.origin_population if ind.rank == 0]
        M = self.config.NOBJ
        min_f_parts = []
        for m in range(M):
            all_vals = [ind.obj[m] for ind in front0
                        if not np.isnan(ind.obj[m])]
            min_val = min(all_vals) if all_vals else np.nan
            min_f_parts.append(f"MinF{m + 1}:{min_val:.4f}")

        igd_str = f"{igd_val:.6f}" if not np.isnan(igd_val) else "  NAN   "
        gd_str = f"{gd_val:.6f}" if not np.isnan(gd_val) else "  NAN   "
        hv_mark = "*" if compute_hv else " "

        line = (f"{self.outer_gen:3d} | IGD:{igd_str} GD:{gd_str} "
                f"HV{hv_mark}:{current_hv:.4f} SP:{current_sp:.4f} "
                f"ONVG:{onvg_val:3d} | " + " ".join(min_f_parts))

        print(line)
        self.log_file.write(line + "\n")

    # ===================== 可视化 =====================
    def _generate_plots(self, run_output_dir):
        """生成全部指标曲线和帕累托前沿图"""
        GAVisualizer.plot_igd_curve(
            self.gen_record, self.igd_record, save_dir=run_output_dir
        )
        GAVisualizer.plot_gd_curve(
            self.gen_record, self.gd_record, save_dir=run_output_dir
        )
        GAVisualizer.plot_hv_curve(
            self.gen_record, self.hv_record, save_dir=run_output_dir
        )
        GAVisualizer.plot_sp_curve(
            self.gen_record, self.sp_record, save_dir=run_output_dir
        )
        GAVisualizer.plot_onvg_curve(
            self.gen_record, self.onvg_record, save_dir=run_output_dir
        )

        M = self.config.NOBJ
        if M >= 3:
            GAVisualizer.plot_pareto_front_parallel(
                self.origin_population, save_dir=run_output_dir
            )
        else:
            GAVisualizer.plot_pareto_front(
                self.origin_population, save_dir=run_output_dir
            )
