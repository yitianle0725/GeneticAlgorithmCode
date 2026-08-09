# metric_calc.py
import numpy as np
from pymoo.indicators.hv import HV
def get_zdt_true_pf(zdt_id: int, num_samples=1000):
    x1 = np.linspace(0, 1, num_samples)
    if zdt_id == 1:
        f1 = x1
        f2 = 1 - np.sqrt(f1)
    elif zdt_id == 2:
        f1 = x1
        f2 = 1 - np.square(f1)
    elif zdt_id == 3:
        f1 = []
        f2 = []
        intervals = [(0,0.0830), (0.1822,0.2577), (0.4093,0.4539), (0.6140,0.6564), (0.8233,0.8518)]
        for l, r in intervals:
            t = np.linspace(l, r, 200)
            f1.extend(t)
            f2.extend(1 - np.sqrt(t) - t * np.sin(10 * np.pi * t))
        f1 = np.array(f1)
        f2 = np.array(f2)
    elif zdt_id == 4:
        f1 = x1
        f2 = 1 - np.sqrt(f1)
    elif zdt_id == 6:
        f1 = 1 - np.exp(-4 * x1) * (np.sin(6 * np.pi * x1)) ** 6
        f2 = 1 - np.square(f1)
    else:
        raise ValueError("仅支持 zdt_id:1,2,3,4,6")
    return np.column_stack([f1, f2])

def calc_all_metrics(population, ref_pf=None, compute_hv=True):
    """计算全部指标。compute_hv=False 时跳过昂贵的 HV 计算，返回缓存值。"""
    front0 = [ind for ind in population if ind.rank == 0]
    onvg_val = len(front0)
    if len(front0) == 0:
        return np.nan, np.nan, 0.0, 0.0, onvg_val
    # 全部目标矩阵
    F_all = np.array([ind.obj for ind in front0])
    # 根据参考PF维度自动截取对应目标
    if ref_pf is not None:
        ref_dim = ref_pf.shape[1]
        F = F_all[:, :ref_dim]
    else:
        F = F_all
    N, M = F.shape

    # ========== IGD + GD 计算 ==========
    igd_val = np.nan
    gd_val = np.nan
    if ref_pf is not None and len(ref_pf) > 0:
        ref_F = ref_pf
        # IGD：真实PF点到解集最小距离均值
        total_igd = 0.0
        for r_pt in ref_F:
            diff = F - r_pt
            diff_sq = np.square(diff)
            diff_sq = np.clip(diff_sq, 0.0, None)
            sq_sum = np.sum(diff_sq, axis=1)
            dists = np.sqrt(sq_sum)
            total_igd += np.min(dists)
        igd_val = total_igd / len(ref_F)

        # GD：解集点到真实PF最小距离均值
        total_gd = 0.0
        for f_pt in F:
            diff = ref_F - f_pt
            diff_sq = np.square(diff)
            diff_sq = np.clip(diff_sq, 0.0, None)
            sq_sum = np.sum(diff_sq, axis=1)
            dists = np.sqrt(sq_sum)
            total_gd += np.min(dists)
        gd_val = total_gd / len(F)

    # ========== HV + SP 计算 ==========
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    range_diff = f_max - f_min
    denom = np.where(np.abs(range_diff) < 1e-10, 1.0, range_diff)
    F_norm = (F - f_min) / denom
    if compute_hv:
        ref_point_norm = np.full(M, 1.1)
        hv_val = _monte_carlo_hv(F_norm, ref_point_norm)
    else:
        hv_val = 0.0  # 非 HV 代返回占位值，由调用方用缓存覆盖
    # Spacing 修正
    d_list = []
    for i in range(N):
        diff = np.abs(F_norm - F_norm[i])
        dist_sum = np.sum(diff, axis=1)
        dist_sum[i] = np.inf
        min_d = np.min(dist_sum)
        d_list.append(min_d)
    d_arr = np.array(d_list)
    d_mean = np.mean(d_arr)
    if N <= 1:
        sp_val = 0.0
    else:
        var = np.sum((d_arr - d_mean) ** 2) / (N - 1)
        sp_val = np.sqrt(var)
    return round(igd_val, 6), round(gd_val, 6), round(hv_val, 4), round(sp_val, 4), onvg_val


def _monte_carlo_hv(F, ref_point, n_samples=50000):
    """蒙特卡洛 HV 估算（M >= 10 时替代精确算法）

    原理: 在参考点围成的超矩形内随机采样 n_samples 个点，
    统计被至少一个 Pareto 解支配的比例，乘以超矩形体积。
    复杂度 O(n_samples × N × M)，在高维中远比精确 WFG 算法可行。
    """
    N, M = F.shape
    rng = np.random.RandomState(42)  # 固定种子保证可复现
    samples = rng.random((n_samples, M)) * ref_point  # (S, M)

    # 向量化: sample s 被解 i 支配 ⇔ F[i,m] <= samples[s,m] ∀m
    # dominated[s] = True if ∃i: ∀m F[i,m] <= samples[s,m]
    dominated = np.zeros(n_samples, dtype=bool)
    # 分批处理避免内存爆炸
    batch_size = 5000
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = samples[start:end]  # (B, M)
        # (N, M) ≤ (B, M) → (N, B, M) → all(N, B)
        dom_check = np.all(F[:, None, :] <= batch[None, :, :], axis=2)  # (N, B)
        dominated[start:end] = np.any(dom_check, axis=0)  # (B,)

    fraction = np.sum(dominated) / n_samples
    volume = np.prod(ref_point)
    return fraction * volume


# ========== DTLZ 真实 Pareto 前沿生成 ==========
def get_dtlz_true_pf(dt_id: int, M: int=3, num_samples=1200):
    """生成标准 DTLZ 问题的真实 PF 采样点（均匀分布）

    使用固定种子确保可复现。
    """
    rng = np.random.RandomState(42)

    if dt_id == 1:
        # DTLZ1 PF: sum(f_i) = 0.5 的单纯形 (均匀采样)
        pf = rng.dirichlet(np.ones(M), size=num_samples) * 0.5

    elif dt_id in [2, 3, 4]:
        # DTLZ2/3/4 PF: sum(f_i²) = 1 的单位球面第一象限
        # 高斯采样 → 均匀分布在正象限球面
        z = np.abs(rng.randn(num_samples, M))
        pf = z / np.sqrt(np.sum(z ** 2, axis=1, keepdims=True))

    elif dt_id in [5, 6]:
        # DTLZ5/6 PF: 退化曲线 (球面上的一条弧线)
        # 退化 PF 难以显式均匀采样，仍用球面采样作为上界参考
        z = np.abs(rng.randn(num_samples, M))
        pf = z / np.sqrt(np.sum(z ** 2, axis=1, keepdims=True))

    elif dt_id == 7:
        # DTLZ7 PF: 分段不连续 (M-1 个变量 + 1 个约束)
        # 简单网格采样 f_1..f_{M-2} = 0, f_{M-1} 扫描
        f_last = np.linspace(0, 1, num_samples)
        pf = np.zeros((num_samples, M))
        pf[:, M - 2] = f_last
        for i in range(num_samples):
            h = M - f_last[i] * (1 + np.sin(3 * np.pi * f_last[i]))
            pf[i, M - 1] = max(0, h)
        # 截断负值
        pf = np.maximum(pf, 0)

    elif dt_id == 8:
        # DTLZ8 PF: sum(f_i) = M, 带约束
        pf = rng.dirichlet(np.ones(M), size=num_samples) * M

    elif dt_id == 9:
        # DTLZ9 PF: sum(f_i) = M (g=0 时)
        pf = rng.dirichlet(np.ones(M), size=num_samples) * M

    elif dt_id == 10:
        # C-DTLZ2 PF: Σ(1-f_i)² = 1 的凸曲面
        # 生成 DTLZ2 球面点，变换 f_i = 1 - z_i
        z = np.abs(rng.randn(num_samples, M))
        z = z / np.sqrt(np.sum(z ** 2, axis=1, keepdims=True))
        pf = 1.0 - z

    else:
        raise ValueError("dt_id仅支持1~10")
    return pf