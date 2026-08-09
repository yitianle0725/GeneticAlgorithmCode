"""
MOEC多目标遗传算法主类
双算法共存：NSGA-II / NSGA-III
适配 ZDT / DTLZ / MOP7(7目标) / Abilene
"""
import math
import random
import itertools
import numpy as np
import os
import datetime
from visualizer import GAVisualizer
from metric_calc import get_zdt_true_pf, get_dtlz_true_pf, calc_all_metrics
from config import GAConfig
from individual import Individual

from fitness_multi import eval_zdt, eval_dtlz, eval_mop7_all
from path_tool import get_project_root

class MultiObjMOECAbilene:
    def __init__(self, zdt_id=None, dt_id=None, dtlz_M=3, run_mop7=False,
                 algo_type="NSGA3", root_output_dir=None, scale_scheme=None):
        self.zdt_id = zdt_id
        self.dt_id = dt_id
        self.dtlz_M = dtlz_M
        self.run_mop7 = run_mop7
        self.algo_type = algo_type.strip()
        self.root_output_dir = root_output_dir

        # 根据问题类型确定目标维度，传入 GAConfig
        if self.run_mop7:
            nobj = 7
        elif self.dt_id is not None:
            nobj = self.dtlz_M
        elif self.zdt_id is not None:
            nobj = 2
        else:
            nobj = 4  # Abilene 网络 4 目标

        self.config = GAConfig(nobj=nobj, scale_scheme=scale_scheme)
        self.ref_pf = None
        # NSGA3 全局参考点缓存
        self.ref_points = None
        # 问题类型配置优先级 MOP7 > DTLZ > Z > Abilene
        if self.run_mop7:
            self.config.VAR_LB = -10.0
            self.config.VAR_UB = 10.0
            self.config.NVARS = 30
            self.ref_pf = None
        elif self.dt_id is not None:
            self.config.VAR_LB = 0.0
            self.config.VAR_UB = 1.0
            self.config.NVARS = self.dtlz_M + 19
            self.ref_pf = get_dtlz_true_pf(self.dt_id, M=self.dtlz_M, num_samples=1000)
            # V-C 尺度缩放: 参考 PF 也需要同步缩放
            if self.config.SCALE_SCHEME != "A":
                self.ref_pf = self.ref_pf * self.config.scale_factors
        elif self.zdt_id is not None:
            if self.zdt_id == 4:
                self.config.NVARS = 10
                self.config.VAR_LB = -5.0
            else:
                self.config.NVARS = 30
                self.config.VAR_LB = 0.0
            self.config.VAR_UB = 1.0
            self.ref_pf = get_zdt_true_pf(self.zdt_id, num_samples=1000)

        # ────── MOEA/D 专用初始化（在创建种群之前） ──────
        self.weight_vectors = None  # N×M 权重向量
        self.neighborhood = None    # B[i] = list of T neighbor indices
        self.z_star = None          # 理想点 (M,)
        if self.algo_type == "MOEAD":
            self.weight_vectors = self._generate_moead_weights()
            # MOEA/D 种群大小 = 权重向量数
            self.config.POPSIZE = len(self.weight_vectors)
            self.neighborhood = self._build_neighborhood()
            self.z_star = np.full(self.config.NOBJ, np.inf)

        # 通用缓存列表
        self.log_file = None
        self.gen_record = []
        self.igd_record = []
        self.gd_record = []  # 新增GD记录
        self.hv_record = []
        self.sp_record = []
        self.onvg_record = []  # 新增ONVG记录
        self.min_f1_record = []
        self.population = [Individual(self.config) for _ in range(self.config.POPSIZE)]
        self.offspring = [Individual(self.config) for _ in range(self.config.POPSIZE)]
        # 早停缓存
        self.last_hv = None
        self.no_improve_cnt = 0

    @staticmethod
    def _randval(low: float, high: float) -> float:
        return (random.randint(0, 999) / 1000.0) * (high - low) + low

    def initialize(self):
        """初始化种群，随机生成决策变量值"""
        for pop_idx in range(self.config.POPSIZE):
            ind = self.population[pop_idx]
            for var_idx in range(self.config.NVARS):
                low = ind.lower[var_idx]
                high = ind.upper[var_idx]
                ind.gene[var_idx] = self._randval(low, high)

    def evaluate_single_ind(self, ind: Individual):
        if self.run_mop7:
            eval_mop7_all(ind)
        elif self.dt_id is not None:
            eval_dtlz(ind, dt_id=self.dt_id, M=self.dtlz_M)
            # V-C 尺度缩放: 目标向量乘以缩放因子
            if self.config.SCALE_SCHEME != "A":
                for m in range(self.config.NOBJ):
                    ind.obj[m] *= self.config.scale_factors[m]
        elif self.zdt_id is not None:
            eval_zdt(ind, zdt_id=self.zdt_id)

    def evaluate_pop(self, pop):
        for ind in pop:
            self.evaluate_single_ind(ind)

    # ===================== NSGA-II 核心函数 =====================
    def fast_non_dominated_sort(self, pop):
        """向量化快速非支配排序

        使用 numpy 广播一次性计算所有 (N×N) 对的支配关系，
        避免 Python 双层循环，大幅降低 M≥5 时的开销。
        复杂度仍为 O(M·N²)，但常数因子缩小 10-50 倍。
        """
        n = len(pop)
        M = self.config.NOBJ
        # 提取目标值矩阵 (n, M)
        F = np.array([[ind.obj[m] for m in range(M)] for ind in pop])

        # 向量化支配判定: dom[i, j] = True 表示 i 支配 j
        # 条件: ∀m: F[i,m] ≤ F[j,m] 且 ∃m: F[i,m] < F[j,m]
        le_all = np.ones((n, n), dtype=bool)   # i ≤ j on all objectives
        lt_any = np.zeros((n, n), dtype=bool)  # i < j on any objective
        for m in range(M):
            le_all &= (F[:, None, m] <= F[None, :, m])
            lt_any |= (F[:, None, m] < F[None, :, m])
        dom_matrix = le_all & lt_any  # (n, n): dom[i,j] = i dominates j

        # 被支配计数: n_dominate_me[j] = 支配 j 的个体数
        dom_count = np.sum(dom_matrix, axis=0).astype(np.int32)

        # 支配集合映射: dominates_whom[i] = i 支配的所有 j
        dominates_whom = [list(np.where(dom_matrix[i])[0]) for i in range(n)]

        # 非支配排序（front 提取仍用循环，但 n 足够小故 OK）
        fronts = []
        assigned = np.zeros(n, dtype=bool)
        remaining = n
        while remaining > 0:
            current_front = np.where((dom_count == 0) & (~assigned))[0]
            if len(current_front) == 0:
                break
            assigned[current_front] = True
            remaining -= len(current_front)
            # 更新被支配计数
            for i in current_front:
                for j in dominates_whom[i]:
                    if not assigned[j]:
                        dom_count[j] -= 1
            fronts.append([pop[i] for i in current_front])

        # 设置 rank
        for fi, front in enumerate(fronts):
            for ind in front:
                ind.rank = fi
        return fronts

    def calc_crowding_distance(self, front):
        n_ind = len(front)
        if n_ind == 0:
            return
        for ind in front:
            ind.crowd_dist = 0.0
        for m in range(self.config.NOBJ):
            front_sorted = sorted(front, key=lambda x: x.obj[m])
            front_sorted[0].crowd_dist = float("inf")
            front_sorted[-1].crowd_dist = float("inf")
            obj_min = front_sorted[0].obj[m]
            obj_max = front_sorted[-1].obj[m]
            if abs(obj_max - obj_min) < 1e-10:
                continue
            for i in range(1, n_ind - 1):
                dist = (front_sorted[i + 1].obj[m] - front_sorted[i - 1].obj[m]) / (obj_max - obj_min)
                front_sorted[i].crowd_dist += dist

    def select_parent(self):
        """锦标赛选择父代 — NSGA2/3 通用

        标准做法 (Deb & Jain, 2014): 父代选择都用 (rank, -crowd_dist)。
        小生境/参考向量仅在 NSGA3 的**环境选择**中起作用。
        """
        size = self.config.TOURNAMENT_SIZE
        candidates = random.sample(self.population, size)
        return min(candidates, key=lambda x: (x.rank, -x.crowd_dist))

    def _sbx_crossover(self, p1_gene, p2_gene, low, high):
        c1, c2 = [0.0] * self.config.NVARS, [0.0] * self.config.NVARS
        for i in range(self.config.NVARS):
            if random.random() <= 0.5:
                xi1 = p1_gene[i]
                xi2 = p2_gene[i]
                if abs(xi1 - xi2) < 1e-10:
                    c1[i], c2[i] = xi1, xi2
                    continue
                x1 = min(xi1, xi2)
                x2 = max(xi1, xi2)
                r = random.random()
                if r < 0.5:
                    beta = (2 * r) ** (1 / (self.config.ETA_C + 1))
                else:
                    beta = (1 / (2 * (1 - r))) ** (1 / (self.config.ETA_C + 1))
                cc1 = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                cc2 = 0.5 * ((x1 + x2) + beta * (x2 - x1))
                c1[i] = max(min(cc1, high[i]), low[i])
                c2[i] = max(min(cc2, high[i]), low[i])
            else:
                c1[i], c2[i] = p1_gene[i], p2_gene[i]
        return c1, c2

    def generate_offspring(self):
        idx = 0
        while idx < self.config.POPSIZE:
            p1 = self.select_parent()
            p2 = self.select_parent()
            c1 = Individual(self.config)
            c2 = Individual(self.config)
            c1.copy_from(p1)
            c2.copy_from(p2)
            if random.random() < self.config.PXOVER:
                c1.gene, c2.gene = self._sbx_crossover(p1.gene, p2.gene, p1.lower, p1.upper)
            mut_rate = self.config.MUT_START - (self.generation / self.config.MAXGENS) * (
                    self.config.MUT_START - self.config.MUT_END)
            for var_idx in range(self.config.NVARS):
                low_c1 = c1.lower[var_idx]
                high_c1 = c1.upper[var_idx]
                if random.random() < mut_rate:
                    c1.gene[var_idx] = self._randval(low_c1, high_c1)
                low_c2 = c2.lower[var_idx]
                high_c2 = c2.upper[var_idx]
                if random.random() < mut_rate:
                    c2.gene[var_idx] = self._randval(low_c2, high_c2)
            self.offspring[idx].copy_from(c1)
            if idx + 1 < self.config.POPSIZE:
                self.offspring[idx + 1].copy_from(c2)
            idx += 2

    # ===================== NSGA-III 新增核心函数 =====================
    def generate_reference_points(self):
        """生成Das-Dennis结构化参考点

        标准组合数: C(M+p-1, p) 个参考点。
        之前误用 combinations_with_replacement 导致 M=8 时生成 11440 个点（正确 120），
        使小生境关联慢 95 倍。现已修正为 itertools.combinations。
        """
        M = self.config.NOBJ
        p = self.config.REF_DIV
        # 标准 Das-Dennis: 从 M+p-1 个位置中选 M-1 个分隔点
        combinations = list(itertools.combinations(range(M + p - 1), M - 1))
        ref = []
        for comb in combinations:
            temp = np.zeros(M)
            prev = -1
            for i in range(M - 1):
                temp[i] = comb[i] - prev - 1
                prev = comb[i]
            # 最后一个分量 = 总位数 - 最后一个分隔位置
            temp[-1] = M + p - 2 - prev
            ref.append(temp / p)
        self.ref_points = np.array(ref)

    def normalize_population(self, pop):
        """目标空间归一化（NSGA-III 标准 ASF 极值点法）

        参考: Deb & Jain (2014) "An Evolutionary Many-Objective Optimization
        Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I"

        步骤:
        1. 计算理想点 z_min, 平移目标空间
        2. 对每个目标轴 m, 用 ASF 标量化函数找到极值点 z^{m,max}
        3. 通过 M 个极值点构建超平面, 求截距 a_m
        4. 用截距归一化: f^n = (f - z_min) / (a - z_min)
        """
        M = self.config.NOBJ
        F = np.array([ind.obj for ind in pop])
        # 1. 理想点 & 平移
        z_min = np.min(F, axis=0)
        F_shift = F - z_min

        # 2. ASF 极值点搜索: 对每个目标轴找使 ASF 最小的个体
        extreme_points = np.zeros((M, M))
        for m in range(M):
            # 权重向量: 第 m 维 = 1.0, 其余 = 1e-6
            w = np.full(M, 1e-6)
            w[m] = 1.0
            # ASF(x, w) = max_i { f'_i(x) / w_i }
            asf_vals = np.max(F_shift / w, axis=1)
            # 取 ASF 最小的个体作为第 m 个极值点
            extreme_idx = np.argmin(asf_vals)
            extreme_points[m] = F_shift[extreme_idx]

        # 3. 通过 M 个极值点构建超平面, 求截距
        # 超平面方程: a_1*x_1 + ... + a_M*x_M = 1
        # 即 extreme_points @ a = [1, 1, ..., 1]^T
        try:
            a = np.linalg.solve(extreme_points, np.ones(M))
            intercept = 1.0 / a
        except np.linalg.LinAlgError:
            # 极值点退化 (线性相关), 回退到各维度最大值
            intercept = np.max(F_shift, axis=0)

        # 4. 纠错: 截距不能小于任何观测值, 且必须为正
        intercept = np.maximum(intercept, np.max(F_shift, axis=0))
        intercept = np.maximum(intercept, 1e-10)

        # 5. 归一化
        scale = intercept - np.zeros(M)  # ideal_shift = 0 (已平移)
        scale[scale < 1e-10] = 1.0
        F_norm = F_shift / scale
        return F_norm, z_min

    def nsga3_environment_selection(self, merge_pop):
        """NSGA3 专属环境选择（替代拥挤度）

        标准 NSGA-III (Deb & Jain, 2014, Algorithm 1):
        1. 将前 k 个前沿的已接受个体关联到参考点，统计小生境计数 ρ_j
        2. 将最后前沿 F_k 关联到参考点
        3. 从 F_k 补选时，优先选 ρ_j 最小的参考方向（维护均匀性）
        """
        N = self.config.POPSIZE
        fronts = self.fast_non_dominated_sort(merge_pop)
        new_pop = []
        k = 0
        while len(new_pop) + len(fronts[k]) <= N:
            new_pop.extend(fronts[k])
            k += 1
        remain = N - len(new_pop)
        if remain == 0:
            return new_pop
        last_front = fronts[k]

        # 生成参考点（仅首次）
        if self.ref_points is None:
            self.generate_reference_points()
        R = len(self.ref_points)

        # 归一化整个合并种群
        F_norm_all, _ = self.normalize_population(merge_pop)
        merge_idx_map = {id(ind): i for i, ind in enumerate(merge_pop)}

        # ── 辅助: 计算个体到各参考点的垂直距离矩阵 ──
        def _perp_dist(F_sub, ref_pts):
            """返回 (n, R) 的垂直距离矩阵"""
            dots = np.dot(F_sub, ref_pts.T)                     # (n, R)
            proj = dots[:, :, None] * ref_pts[None, :, :]       # (n, R, M)
            diff = F_sub[:, None, :] - proj                      # (n, R, M)
            return np.linalg.norm(diff, axis=2)                  # (n, R)

        # ── 1. 统计已接受个体 (new_pop) 的小生境计数 ρ ──
        rho = np.zeros(R, dtype=int)
        if len(new_pop) > 0:
            new_indices = [merge_idx_map[id(ind)] for ind in new_pop]
            F_new = F_norm_all[new_indices]
            dist_new = _perp_dist(F_new, self.ref_points)
            nearest_new = np.argmin(dist_new, axis=1)
            for r in nearest_new:
                rho[int(r)] += 1

        # ── 2. 关联最后前沿 F_k 到参考点 ──
        last_indices = [merge_idx_map[id(ind)] for ind in last_front]
        F_last = F_norm_all[last_indices]
        dist_last = _perp_dist(F_last, self.ref_points)      # (|last_front|, R)
        nearest_last = np.argmin(dist_last, axis=1)

        # niche_dict: r → list of (local_idx, individual)
        niche_dict = {r: [] for r in range(R)}
        for local_i, ind in enumerate(last_front):
            r = int(nearest_last[local_i])
            niche_dict[r].append((local_i, ind))

        # ── 3. 小生境选择：从 F_k 补选，优先 ρ_j 最小的参考方向 ──
        selected = []
        exhausted = set()  # 已被掏空的参考方向
        while len(selected) < remain:
            # 候选: 未耗尽且有关联个体的参考方向
            candidates = [r for r in range(R)
                          if r not in exhausted and len(niche_dict[r]) > 0]
            min_rho = min(rho[r] for r in candidates)
            ref_candidates = [r for r in candidates if rho[r] == min_rho]
            pick_r = random.choice(ref_candidates)

            ind_list = niche_dict[pick_r]
            # 选垂直距离最近的个体
            best_item = min(ind_list, key=lambda x: dist_last[x[0], pick_r])
            sel_ind = best_item[1]
            selected.append(sel_ind)
            ind_list.remove(best_item)
            rho[pick_r] += 1
            if len(ind_list) == 0:
                exhausted.add(pick_r)

        new_pop.extend(selected)
        return new_pop

    # ===================== 统一环境选择入口（自动切换算法） =====================
    def elite_survival(self):
        merge_pop = self.population + self.offspring
        if self.algo_type == "NSGA2":
            # NSGA-II 拥挤度策略
            fronts = self.fast_non_dominated_sort(merge_pop)
            new_pop = []
            for front in fronts:
                self.calc_crowding_distance(front)
                if len(new_pop) + len(front) <= self.config.POPSIZE:
                    new_pop.extend(front)
                else:
                    front_sorted = sorted(front, key=lambda x: -x.crowd_dist)
                    new_pop.extend(front[:self.config.POPSIZE - len(new_pop)])
                    break
            for i in range(self.config.POPSIZE):
                self.population[i].copy_from(new_pop[i])
        elif self.algo_type == "NSGA3":
            # NSGA-III 参考向量小生境策略
            new_pop = self.nsga3_environment_selection(merge_pop)
            for i in range(self.config.POPSIZE):
                self.population[i].copy_from(new_pop[i])

    def report(self):
        # 判断是否在本代计算 HV（高维 HV 非常昂贵，降低频率）
        compute_hv = (self.generation % self.config.HV_CALC_INTERVAL == 0)
        igd_val, gd_val, current_hv, current_sp, onvg_val = calc_all_metrics(
            self.population, self.ref_pf, compute_hv=compute_hv
        )
        # 非 HV 代沿用上一次 HV 值
        if not compute_hv and len(self.hv_record) > 0:
            current_hv = self.hv_record[-1]

        self.gen_record.append(self.generation)
        self.igd_record.append(igd_val)
        self.gd_record.append(gd_val)
        self.hv_record.append(current_hv)
        self.sp_record.append(current_sp)
        self.onvg_record.append(onvg_val)
        front0 = [ind for ind in self.population if ind.rank == 0]
        M = self.config.NOBJ
        min_f_str_parts = []
        for obj_idx in range(M):
            all_obj = [ind.obj[obj_idx] for ind in front0 if not np.isnan(ind.obj[obj_idx])]
            min_val = min(all_obj) if len(all_obj) else np.nan
            min_f_str_parts.append(f"MinF{obj_idx + 1}:{min_val:.4f}")
        igd_str = f"{igd_val:.6f}" if not np.isnan(igd_val) else "  NAN   "
        gd_str = f"{gd_val:.6f}" if not np.isnan(gd_val) else "  NAN   "
        hv_mark = "*" if compute_hv else " "
        base_part = (
            f"{self.generation:3d} | IGD:{igd_str} GD:{gd_str} HV{hv_mark}:{current_hv:.4f} SP:{current_sp:.4f} ONVG:{onvg_val:3d} | "
        )
        full_log_line = base_part + " ".join(min_f_str_parts) + "\n"
        print(full_log_line.strip())
        self.log_file.write(full_log_line)

    # ===================== MOEA/D-TCH 算法 =====================
    def _generate_moead_weights(self):
        """生成 MOEA/D 权重向量（Das-Dennis 方法）

        MOEA/D 每个权重向量对应一个子问题，种群大小 = 权重向量数。
        与 NSGA-III 参考点生成相同，但用途不同：NSGA-III 用于小生境引导，
        MOEA/D 直接定义分解后的单目标子问题。
        """
        M = self.config.NOBJ
        p = self.config.REF_DIV
        combos = list(itertools.combinations(range(M + p - 1), M - 1))
        weights = np.zeros((len(combos), M))
        for idx, comb in enumerate(combos):
            prev = -1
            for i in range(M - 1):
                weights[idx, i] = comb[i] - prev - 1
                prev = comb[i]
            weights[idx, -1] = M + p - 2 - prev
            weights[idx] /= p
        return weights

    def _build_neighborhood(self):
        """为每个权重向量构建邻域索引列表

        邻域大小 T = max(⌊N × T_RATIO⌋, 20)，通过两两欧氏距离取最近 T 个。
        """
        W = self.weight_vectors
        N = len(W)
        T = max(int(N * self.config.MOEAD_T_RATIO), 20)
        T = min(T, N)  # 不能超过种群大小

        # 向量化距离矩阵: (N, N)
        diff = W[:, None, :] - W[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # 每行取 T 个最近邻（跳过自己）
        neighborhood = []
        for i in range(N):
            sorted_idx = np.argsort(dist[i])
            # 第一个是自身，取之后的 T 个
            neighbors = sorted_idx[1:T + 1].tolist()
            neighborhood.append(neighbors)
        return neighborhood

    @staticmethod
    def _tchebycheff(f, w, z_star):
        """Tchebycheff 标量化函数

        g^{tch}(f | w, z*) = max_i { w_i * |f_i - z*_i| }

        标准 MOEA/D-TCH (Zhang & Li, 2007): 权重越大 = 该目标越受重视。
        值越小越好。
        """
        return np.max(w * np.abs(f - z_star))

    def _moead_iteration(self):
        """执行一代 MOEA/D-TCH 迭代

        对每个子问题 i:
          1. 以概率 delta 从邻域 B(i) 选父母，否则全局选
          2. SBX 交叉 + 多项式变异 → 一个子代 y
          3. 评估 y
          4. 更新理想点 z*
          5. 用 TCH 值尝试替换邻域内最多 nr 个个体
        """
        N = self.config.POPSIZE
        delta = self.config.MOEAD_DELTA
        nr = self.config.MOEAD_NR

        # 打乱子问题顺序（每代随机，避免偏向）
        perm = list(range(N))
        random.shuffle(perm)

        for i in perm:
            # ── 1. 选择父母 ──
            if random.random() < delta:
                pool = self.neighborhood[i]  # 邻域
            else:
                pool = list(range(N))        # 全局

            # 从池中随机选两个不同的父母
            p1_idx, p2_idx = random.sample(pool, 2)
            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            # ── 2. SBX + 变异 ──
            child = Individual(self.config)
            child.copy_from(p1)
            if random.random() < self.config.PXOVER:
                c1_gene, _ = self._sbx_crossover(p1.gene, p2.gene,
                                                  p1.lower, p1.upper)
                child.gene = c1_gene
            # 多项式变异
            mut_rate = self.config.MUT_START - \
                       (self.generation / self.config.MAXGENS) * \
                       (self.config.MUT_START - self.config.MUT_END)
            for var_idx in range(self.config.NVARS):
                if random.random() < mut_rate:
                    child.gene[var_idx] = self._randval(
                        child.lower[var_idx], child.upper[var_idx]
                    )

            # ── 3. 评估 ──
            self.evaluate_single_ind(child)

            # ── 4. 更新理想点 ──
            for m in range(self.config.NOBJ):
                if child.obj[m] < self.z_star[m]:
                    self.z_star[m] = child.obj[m]

            # ── 5. 更新邻域（最多 nr 个） ──
            neighbors = self.neighborhood[i][:]  # copy
            random.shuffle(neighbors)
            replaced = 0
            for j in neighbors:
                old_tch = self._tchebycheff(
                    self.population[j].obj, self.weight_vectors[j], self.z_star
                )
                new_tch = self._tchebycheff(
                    child.obj, self.weight_vectors[j], self.z_star
                )
                if new_tch <= old_tch:
                    self.population[j].copy_from(child)
                    replaced += 1
                    if replaced >= nr:
                        break

        # 每代结束时做一次非支配排序（为 report 提供 rank）
        self.fast_non_dominated_sort(self.population)

    def run(self):
        # 统一通过path_tool获取项目根目录
        project_root = get_project_root()
        base_output_dir = os.path.join(project_root, "output")
        dir_name = ""
        if self.root_output_dir is not None:
            if self.run_mop7:
                dir_name = f"MOP7_F1_F2_F6_F7_F11_F18_{self.algo_type}"
                run_output_dir = os.path.join(self.root_output_dir, dir_name)
            elif self.dt_id is not None:
                dir_name = f"DTLZ{self.dt_id}_M{self.dtlz_M}_{self.algo_type}"
                if self.config.SCALE_SCHEME != "A":
                    dir_name += f"_Scale{self.config.SCALE_SCHEME}"
                run_output_dir = os.path.join(self.root_output_dir, dir_name)
            elif self.zdt_id is not None:
                dir_name = f"ZDT{self.zdt_id}_{self.algo_type}"
                run_output_dir = os.path.join(self.root_output_dir, dir_name)
            else:
                dir_name = f"Abilene_4obj_{self.algo_type}"
                run_output_dir = os.path.join(self.root_output_dir, dir_name)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.run_mop7:
                sub_dir = f"MOP7_F1-F2-F6-F7-F11-F14-F18_{self.algo_type}_{timestamp}"
            elif self.dt_id is not None:
                sub_dir = f"DTLZ{self.dt_id}_M{self.dtlz_M}_{self.algo_type}_{timestamp}"
                if self.config.SCALE_SCHEME != "A":
                    sub_dir = sub_dir.replace(f"_{timestamp}", f"_Scale{self.config.SCALE_SCHEME}_{timestamp}")
            elif self.zdt_id is not None:
                sub_dir = f"ZDT{self.zdt_id}_{self.algo_type}_{timestamp}"
            else:
                sub_dir = f"Abilene_4obj_{self.algo_type}_{timestamp}"
            run_output_dir = os.path.join(base_output_dir, sub_dir)
            dir_name = sub_dir
        os.makedirs(run_output_dir, exist_ok=True)
        log_path = os.path.join(run_output_dir, "ga_log.txt")
        self.log_file = open(log_path, "a", encoding="utf-8")
        if os.path.getsize(log_path) == 0:
            M = self.config.NOBJ
            header_parts = ["Gen | IGD | GD | HV | SP | ONVG"]
            for i in range(1, M + 1):
                header_parts.append(f"MinF{i}")
            header = " ".join(header_parts) + "\n"
            self.log_file.write(header)
            self.log_file.write("-" * 130 + "\n")
        # 初始化种群
        self.initialize()
        self.evaluate_pop(self.population)

        # MOEA/D: 初始化理想点 z*
        if self.algo_type == "MOEAD":
            F_init = np.array([ind.obj for ind in self.population])
            self.z_star = np.min(F_init, axis=0)
            self.fast_non_dominated_sort(self.population)  # 为 gen=0 report 提供 rank
        else:
            fronts = self.fast_non_dominated_sort(self.population)
            for f in fronts:
                self.calc_crowding_distance(f)

        self.generation = 0
        # 主迭代循环
        for self.generation in range(self.config.MAXGENS + 1):
            self.report()
            # 提前终止判断
            current_hv = self.hv_record[-1]
            if self.last_hv is None:
                self.last_hv = current_hv
                self.no_improve_cnt = 0
            else:
                delta_hv = abs(current_hv - self.last_hv)
                if delta_hv < self.config.HV_TOL:
                    self.no_improve_cnt += 1
                else:
                    self.no_improve_cnt = 0
                    self.last_hv = current_hv
                if self.no_improve_cnt >= self.config.EARLY_STOP_PATIENCE:
                    print(
                        f"\n【提前终止】连续{self.config.EARLY_STOP_PATIENCE}代HV变化<{self.config.HV_TOL}，解集无明显更新，停止迭代！")
                    break
            if self.generation >= self.config.MAXGENS:
                break

            # ── 算法分支 ──
            if self.algo_type == "MOEAD":
                self._moead_iteration()
            else:
                self.generate_offspring()
                self.evaluate_pop(self.offspring)
                self.elite_survival()
                new_fronts = self.fast_non_dominated_sort(self.population)
                for f in new_fronts:
                    self.calc_crowding_distance(f)
        # 绘制全部指标曲线
        GAVisualizer.plot_igd_curve(self.gen_record, self.igd_record, save_dir=run_output_dir)
        GAVisualizer.plot_gd_curve(self.gen_record, self.gd_record, save_dir=run_output_dir)
        GAVisualizer.plot_hv_curve(self.gen_record, self.hv_record, save_dir=run_output_dir)
        GAVisualizer.plot_sp_curve(self.gen_record, self.sp_record, save_dir=run_output_dir)
        GAVisualizer.plot_onvg_curve(self.gen_record, self.onvg_record, save_dir=run_output_dir)
        if not self.run_mop7:
            M = self.config.NOBJ
            if M >= 3:
                GAVisualizer.plot_pareto_front_parallel(self.population, save_dir=run_output_dir)
            else:
                GAVisualizer.plot_pareto_front(self.population, save_dir=run_output_dir)
        self.log_file.close()
        # 新增：输出保存路径提示
        print(f"\n📁 本次所有结果保存在：{run_output_dir}")
        print(f"✅ 完成 {dir_name}")
        print(f"最终 IGD  = {self.igd_record[-1]:.6f}")
        print(f"最终 GD   = {self.gd_record[-1]:.6f}")
        print(f"最终 HV   = {self.hv_record[-1]:.4f}")
        print(f"最终 SP   = {self.sp_record[-1]:.4f}")
        print(f"最终 ONVG = {self.onvg_record[-1]}")
        print("=" * 30, "\n")