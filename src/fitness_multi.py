# fitness_multi.py
# ZDT1~ZDT6 + DTLZ1/2/4/7 标准多目标测试函数
# 兼容现有Individual、MOEC/NSGA-II框架
import numpy as np
from individual import Individual
from test_dtlz import compute_dtlz

# ====================== 底层ZDT计算函数（原有全部保留） ======================
def compute_zdt1(x: np.ndarray):
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0/(n-1) * np.sum(x[1:])
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return f1, f2

def compute_zdt2(x: np.ndarray):
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0/(n-1) * np.sum(x[1:])
    f2 = g * (1.0 - (f1 / g) ** 2)
    return f1, f2

def compute_zdt3(x: np.ndarray):
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0/(n-1) * np.sum(x[1:])
    f2 = g * (1.0 - np.sqrt(f1/g) - (f1/g)*np.sin(10 * np.pi * f1))
    return f1, f2

def compute_zdt4(x: np.ndarray):
    n = len(x)
    f1 = x[0]
    sum_term = 0.0
    for i in range(1, n):
        xi = x[i]
        sum_term += xi**2 - 10 * np.cos(4 * np.pi * xi)
    g = 1.0 + 10*(n-1) + sum_term
    ratio = np.clip(f1 / g, 0.0, None)
    f2 = g * (1.0 - np.sqrt(ratio))
    return f1, f2

def compute_zdt5(x: np.ndarray):
    n = len(x)
    count = np.sum(np.round(np.clip(x,0,1)))
    f1 = 1 + count
    g = 1 + 9.0/(n-1)*np.sum(np.abs(x[1:]-0.5))
    f2 = g * (1.0 / f1)
    return f1, f2

def compute_zdt6(x: np.ndarray):
    n = len(x)
    f1 = 1.0 - np.exp(-4*x[0]) * (np.sin(6 * np.pi * x[0]))**6
    avg = np.sum(x[1:]) / (n-1)
    g = 1.0 + 9.0 * (avg ** 0.25)
    f2 = g * (1.0 - (f1 / g)**2)
    return f1, f2

# ====================== 统一对外ZDT评估接口（原有保留） ======================
def eval_zdt(ind: Individual, zdt_id: int):
    x = np.array(ind.gene)
    if zdt_id == 1:
        f1, f2 = compute_zdt1(x)
    elif zdt_id == 2:
        f1, f2 = compute_zdt2(x)
    elif zdt_id == 3:
        f1, f2 = compute_zdt3(x)
    elif zdt_id == 4:
        f1, f2 = compute_zdt4(x)
    elif zdt_id == 5:
        f1, f2 = compute_zdt5(x)
    elif zdt_id == 6:
        f1, f2 = compute_zdt6(x)
    else:
        raise ValueError("zdt_id仅支持 1,2,3,4,5,6")
    ind.obj[0] = f1
    ind.obj[1] = f2
    ind.obj[2] = 0.0
    ind.obj[3] = 0.0
    ind.raw_F1 = f1
    ind.raw_F2 = f2
    ind.raw_F3 = 0.0
    ind.raw_F4 = 0.0

# # ====================== 新增 DTLZ 底层计算 ======================
# def compute_dtlz1(x: np.ndarray, M: int):
#     """
#     DTLZ1：线性前沿，大量局部最优
#     M:目标数量(3/5)，变量全部[0,1]
#     g = 100*(k + sum((xi-0.5)^2 - cos(20π(xi-0.5))))
#     k = len(x)-M+1
#     """
#     D = len(x)
#     k = D - M + 1
#     xc = x[:M-1]
#     xm = x[M-1:]
#     g = 100 * (k + np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5))))
#     f = []
#     prod = 1.0
#     for i in range(M-1):
#         prod *= xc[i]
#         fi = 0.5 * (1 + g) * prod
#         f.append(fi)
#     fm = 0.5 * (1 + g) * (1 - xc[-1])
#     f.append(fm)
#     return np.array(f)
#
# def compute_dtlz2(x: np.ndarray, M: int):
#     """DTLZ2 球形凸前沿，无局部最优"""
#     D = len(x)
#     k = D - M + 1
#     xc = x[:M-1]
#     xm = x[M-1:]
#     g = np.sum((xm - 0.5)**2)
#     f = []
#     pi2 = np.pi / 2
#     prod = 1.0
#     for i in range(M-1):
#         prod *= np.cos(xc[i] * pi2)
#         fi = (1 + g) * prod
#         f.append(fi)
#     fm = (1 + g) * prod * np.sin(xc[-1] * pi2)
#     f.append(fm)
#     return np.array(f)
#
# def compute_dtlz4(x: np.ndarray, M: int, alpha=100):
#     """DTLZ4 前沿分布偏移，alpha=100"""
#     D = len(x)
#     k = D - M + 1
#     xc = x[:M-1]
#     xm = x[M-1:]
#     g = np.sum((xm - 0.5)**2)
#     f = []
#     pi2 = np.pi / 2
#     prod = 1.0
#     for i in range(M-1):
#         xi_alpha = xc[i] ** alpha
#         prod *= np.cos(xi_alpha * pi2)
#         fi = (1 + g) * prod
#         f.append(fi)
#     xlast_alpha = xc[-1] ** alpha
#     fm = (1 + g) * prod * np.sin(xlast_alpha * pi2)
#     f.append(fm)
#     return np.array(f)
#
# def compute_dtlz7(x: np.ndarray, M: int):
#     """DTLZ7 分段不连续前沿"""
#     D = len(x)
#     k = D - M + 1
#     f = x[:M-1].copy()
#     xm = x[M-1:]
#     g = np.sum(xm)
#     h = M
#     for fi in f:
#         h -= fi / (1 + g) * (1 + np.sin(3 * np.pi * fi))
#     fm = (1 + g) * h
#     f = np.concatenate([f, [fm]])
#     return f


def eval_dtlz(ind: Individual, dt_id: int, M: int=3):
    x = np.array(ind.gene)
    f_vec = compute_dtlz(dt_id, x, M)
    # 修复：用config.NOBJ而不是固定7
    max_obj = ind.config.NOBJ
    for m in range(max_obj):
        if m < len(f_vec):
            val = f_vec[m]
        else:
            val = 0.0
        ind.obj[m] = val
    # raw缓存
    ind.raw_F1 = f_vec[0] if M>=1 else 0
    ind.raw_F2 = f_vec[1] if M>=2 else 0
    ind.raw_F3 = f_vec[2] if M>=3 else 0
    ind.raw_F4 = f_vec[3] if M>=4 else 0
    ind.raw_F5 = f_vec[4] if M>=5 else 0
    ind.raw_F6 = f_vec[5] if M>=6 else 0
    ind.raw_F7 = f_vec[6] if M>=7 else 0

# ===================== 7维多目标 MOP7 评估 =====================
from fitness_function import compute_f1,compute_f2,compute_f6,compute_f7,compute_f11,compute_f14,compute_f18

# 固定7目标顺序 F1,F2,F6,F7,F11,F14,F18
F7_ORDER = [1,2,6,7,11,14,18]
FUNC_DICT = {
    1: compute_f1,
    2: compute_f2,
    6: compute_f6,
    7: compute_f7,
    11: compute_f11,
    14: compute_f14,
    18: compute_f18
}

def eval_mop7_all(ind: Individual):
    x = np.array(ind.gene)
    f_list = []
    for fid in F7_ORDER:
        val = FUNC_DICT[fid](x)
        f_list.append(val)
    # 填充7维obj
    for idx in range(7):
        ind.obj[idx] = f_list[idx]
    # 原始值缓存
    ind.raw_F1 = f_list[0]
    ind.raw_F2 = f_list[1]
    ind.raw_F3 = f_list[2]
    ind.raw_F4 = f_list[3]
    ind.raw_F5 = f_list[4]
    ind.raw_F6 = f_list[5]
    ind.raw_F7 = f_list[6]