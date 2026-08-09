class GAConfig:
    def __init__(self, nobj=None, scale_scheme=None):
        # ========= 切换目标维度 3 / 5 / 8 / 10 / 15 =========
        # 可通过 nobj 参数动态设置，不传则默认 8
        if nobj is not None:
            if nobj not in (3, 5, 8, 10, 15):
                raise ValueError(f"仅支持 NOBJ = 3,5,8,10,15，收到 {nobj}")
            self.NOBJ = nobj
        else:
            self.NOBJ = 8

        # ──────── V-C 尺度缩放方案 ────────
        # None / "A": 无缩放（对照）
        # "B": f1=1, 其余=10
        # "C": 奇偶交替 [10,1,10,1,...]
        # "D": 递增幂 [1,1,10,100,1000,...]
        if scale_scheme is not None and scale_scheme not in ("A", "B", "C", "D"):
            raise ValueError(f"scale_scheme 仅支持 A/B/C/D/None，收到 {scale_scheme}")
        self.SCALE_SCHEME = scale_scheme if scale_scheme else "A"
        self.scale_factors = self._compute_scale_factors()

        # 根据 NOBJ 加载论文标准参数
        # 参考: Deb & Jain (2014) "NSGA-III, Part I"
        # N = 最小的 4 的倍数 ≥ H (参考点数)
        if self.NOBJ == 3:
            self.REF_DIV = 12          # 参考点数 H = C(14,12) = 91
            self.POPSIZE = 92          # 4 的倍数 ≥ 91 → 92 ✓
            self.MAXGENS = 400
        elif self.NOBJ == 5:
            self.REF_DIV = 6           # 参考点数 H = C(10,6) = 210
            self.POPSIZE = 212         # 4 的倍数 ≥ 210 → 212
            self.MAXGENS = 600
        elif self.NOBJ == 8:
            self.REF_DIV = 3           # 参考点数 H = C(10,3) = 120
            self.POPSIZE = 120         # 4 的倍数 ≥ 120 → 120
            self.MAXGENS = 750
        elif self.NOBJ == 10:
            self.REF_DIV = 3           # 参考点数 H = C(12,3) = 220
            self.POPSIZE = 220         # 4 的倍数 ≥ 220 → 220
            self.MAXGENS = 1000
        elif self.NOBJ == 15:
            self.REF_DIV = 2           # 参考点数 H = C(16,2) = 120
            self.POPSIZE = 120         # 4 的倍数 ≥ 120 → 120
            self.MAXGENS = 1500
        else:
            raise ValueError("仅支持 NOBJ = 3,5,8,10,15")

        # DTLZ 变量数 n = M + (k=19), 即 M + 19
        self.NVARS = self.NOBJ + 19

        # 进化算子（匹配原文 Deb & Jain 2014）
        self.PXOVER = 1.0
        self.ETA_C = 30               # SBX 分布指数（论文值 30）
        self.ETA_M = 20               # 多项式变异分布指数（论文值 20）

        # 多项式变异概率 p_m = 1/n（论文标准）
        p_m = 1.0 / self.NVARS
        self.MUT_START = p_m
        self.MUT_END = p_m            # 恒定变异率 = 1/n

        self.TOURNAMENT_SIZE = 2      # 二元锦标赛（论文标准）
        self.VAR_LB = 0.0
        self.VAR_UB = 1.0

        # 提前终止
        self.EARLY_STOP_PATIENCE = 30
        self.HV_TOL = 1e-4

        # HV 计算间隔: 每隔 N 代计算一次 HV（高维 HV 很昂贵）
        # M≥8 时 HV 复杂度指数增长，降低频率可大幅提速
        self.HV_CALC_INTERVAL = 1 if self.NOBJ <= 3 else (5 if self.NOBJ <= 5 else 10)

        # ──────────── MOEA/D 专用参数 ────────────
        # 邻域大小: POPSIZE × T_RATIO 与 20 取较大值
        self.MOEAD_T_RATIO = 0.1
        # 邻域交配概率: 以 delta 概率从邻域选父母，否则全局
        self.MOEAD_DELTA = 0.9
        # 最大替换数: 每个子代最多替换 nr 个邻居（防止早熟）
        self.MOEAD_NR = 2

    def _compute_scale_factors(self):
        """返回 M 维缩放因子向量（V-C 尺度缩放实验）

        A: [1, 1, ..., 1]           — 对照组无缩放
        B: [1, 10, 10, ..., 10]     — f1 正常, 其余扩大 10×
        C: [10, 1, 10, 1, ...]      — 奇偶交替
        D: [1, 1, 10, 100, ...]     — 逐目标递增幂次
        """
        import numpy as np
        M = self.NOBJ
        s = self.SCALE_SCHEME
        if s == "A":
            return np.ones(M)
        elif s == "B":
            vec = np.full(M, 10.0)
            vec[0] = 1.0
            return vec
        elif s == "C":
            vec = np.ones(M)
            vec[::2] = 10.0
            return vec
        elif s == "D":
            return np.array([10 ** max(0, i - 1) for i in range(M)], dtype=float)
        return np.ones(M)  # fallback