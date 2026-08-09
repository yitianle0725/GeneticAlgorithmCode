# test_dtlz.py
# DTLZ1 ~ DTLZ9 标准多目标测试函数完整实现
import numpy as np

def calc_g_dtlz1(X_M):
    k = len(X_M)
    g = 100 * (k + np.sum((X_M - 0.5) ** 2 - np.cos(20 * np.pi * (X_M - 0.5))))
    return g

def dtlz1(x: np.ndarray, M: int):
    """标准 DTLZ1: PF 为 sum(f_i) = 0.5 的单纯形

    f_1 = 0.5 * x_1 * ... * x_{M-1} * (1+g)
    f_i = 0.5 * (Π_{j=1}^{M-i} x_j) * (1 - x_{M-i+1}) * (1+g)   i=2..M-1
    f_M = 0.5 * (1 - x_1) * (1+g)
    """
    n = len(x)
    k = n - M + 1
    X_M = x[M - 1:]  # k 个距离变量
    g = calc_g_dtlz1(X_M)

    f = np.zeros(M)
    for i in range(M):
        base = 0.5 * (1 + g)
        # 乘入 x_1 * ... * x_{M-i-1}
        for j in range(M - i - 1):
            base *= x[j]
        # f_2..f_M 需要 (1 - x_{M-i-1}) 因子
        if i > 0:
            base *= (1 - x[M - i - 1])
        f[i] = base
    return f

def calc_g_dtlz23456(X_M):
    return np.sum((X_M - 0.5) ** 2)

def dtlz2(x: np.ndarray, M: int):
    """标准 DTLZ2: PF 为单位球面 Σf_i² = 1

    f_i = (1+g) * cos(x_1*π/2) * ... * cos(x_{M-i}*π/2) * sin(x_{M-i+1}*π/2)
    f_M = (1+g) * sin(x_1*π/2)
    """
    n = len(x)
    k = n - M + 1
    X_M = x[M - 1:]
    g = calc_g_dtlz23456(X_M)
    pi2 = np.pi / 2

    f = np.zeros(M)
    for i in range(M - 1):  # f_1 .. f_{M-1}
        base = (1 + g)
        # cos(x_1) * ... * cos(x_{M-1-i})
        for j in range(M - 1 - i):
            base *= np.cos(x[j] * pi2)
        # sin(x_{M-i}) for i > 0
        if i > 0:
            base *= np.sin(x[M - 1 - i] * pi2)
        f[i] = base
    # f_M
    f[M - 1] = (1 + g) * np.sin(x[0] * pi2)
    return f

def dtlz3(x: np.ndarray, M: int):
    """标准 DTLZ3: DTLZ2 结构 + DTLZ1 的 g 函数（多模态）"""
    n = len(x)
    k = n - M + 1
    X_M = x[M - 1:]
    g = calc_g_dtlz1(X_M)
    pi2 = np.pi / 2

    f = np.zeros(M)
    for i in range(M - 1):
        base = (1 + g)
        for j in range(M - 1 - i):
            base *= np.cos(x[j] * pi2)
        if i > 0:
            base *= np.sin(x[M - 1 - i] * pi2)
        f[i] = base
    f[M - 1] = (1 + g) * np.sin(x[0] * pi2)
    return f

def dtlz4(x: np.ndarray, M: int, alpha=100):
    """标准 DTLZ4: DTLZ2 结构 + x^alpha 映射（测试收敛均匀性）"""
    n = len(x)
    k = n - M + 1
    X_M = x[M - 1:]
    g = calc_g_dtlz23456(X_M)
    pi2 = np.pi / 2

    f = np.zeros(M)
    for i in range(M - 1):
        base = (1 + g)
        for j in range(M - 1 - i):
            base *= np.cos(x[j] ** alpha * pi2)
        if i > 0:
            base *= np.sin(x[M - 1 - i] ** alpha * pi2)
        f[i] = base
    f[M - 1] = (1 + g) * np.sin(x[0] ** alpha * pi2)
    return f

def dtlz5(x: np.ndarray, M: int):
    """标准 DTLZ5: DTLZ2 结构 + θ 参数化（退化 PF 曲线）"""
    n = len(x)
    k = n - M + 1
    X_M = x[M - 1:]
    g = calc_g_dtlz23456(X_M)
    pi2 = np.pi / 2

    theta = np.zeros(M - 1)
    theta[0] = x[0] * pi2
    for j in range(1, M - 1):
        theta[j] = pi2 / (4 * (1 + g)) * (1 + 2 * g * x[j])

    f = np.zeros(M)
    for i in range(M - 1):
        base = (1 + g)
        for j in range(M - 1 - i):
            base *= np.cos(theta[j])
        if i > 0:
            base *= np.sin(theta[M - 1 - i])
        f[i] = base
    f[M - 1] = (1 + g) * np.sin(theta[0])
    return f

def dtlz6(x: np.ndarray, M: int):
    """标准 DTLZ6: DTLZ5 结构 + 不同的 g 函数"""
    n = len(x)
    k = n - M + 1
    X_M = x[M - 1:]
    g = np.sum(X_M ** 0.1)
    pi2 = np.pi / 2

    theta = np.zeros(M - 1)
    theta[0] = x[0] * pi2
    for j in range(1, M - 1):
        theta[j] = pi2 / (4 * (1 + g)) * (1 + 2 * g * x[j])

    f = np.zeros(M)
    for i in range(M - 1):
        base = (1 + g)
        for j in range(M - 1 - i):
            base *= np.cos(theta[j])
        if i > 0:
            base *= np.sin(theta[M - 1 - i])
        f[i] = base
    f[M - 1] = (1 + g) * np.sin(theta[0])
    return f

def dtlz7(x: np.ndarray, M: int):
    n = len(x)
    k = n - M + 1
    f = x[:M-1].copy()
    Xm = x[M-1:]
    g = np.sum(Xm)
    h = M
    for fi in f:
        h -= fi / (1 + g) * (1 + np.sin(3 * np.pi * fi))
    fm = (1 + g) * h
    f = np.concatenate([f, [fm]])
    return f

def dtlz8(x: np.ndarray, M: int):
    n = len(x)
    k = n - M + 1
    Xc = x[:M-1]
    Xm = x[M-1:]
    g = np.sum(Xm)
    f = []
    for i in range(M):
        if i < M-1:
            fi = Xc[i]
        else:
            sum_f = np.sum(f)
            fi = (M - sum_f) / (1 + g)
        f.append(fi)
    return np.array(f)

def dtlz9(x: np.ndarray, M: int):
    n = len(x)
    k = n - M + 1
    Xc = x[:M-1]
    Xm = x[M-1:]
    g = np.sum(Xm ** 0.1)
    f = []
    for i in range(M):
        if i < M-1:
            fi = Xc[i]
        else:
            sum_f = np.sum(f)
            fi = (M - sum_f) * (1 + g)
        f.append(fi)
    return np.array(f)

def cdtlz2(x: np.ndarray, M: int):
    """C-DTLZ2 = 凸帕累托前沿（论文 V-D 实验）

    f_i = (1+g) × (1 - t_i)，其中 t_i 为 DTLZ2 球面映射函数。
    g=0 时 PF 满足 Σ(1-f_i)² = 1，为凸曲面（DTLZ2 的 Σf²=1 是凹的）。

    等价于: f = (1+g) - f_DTLZ2
    """
    f_dtlz2_vals = dtlz2(x, M)
    one_plus_g = np.sqrt(np.sum(f_dtlz2_vals ** 2))  # DTLZ2: Σf² = (1+g)²
    return one_plus_g - f_dtlz2_vals


# 统一调度入口
DTLZ_FUNC_MAP = {
    1: dtlz1,
    2: dtlz2,
    3: dtlz3,
    4: dtlz4,
    5: dtlz5,
    6: dtlz6,
    7: dtlz7,
    8: dtlz8,
    9: dtlz9,
    10: cdtlz2,  # V-D 凸帕累托前沿
}

def compute_dtlz(dtid: int, x: np.ndarray, M: int):
    """统一调用DTLZ1~9"""
    if dtid not in DTLZ_FUNC_MAP:
        raise ValueError(f"dt_id仅支持1~10，输入{dtid}非法")
    func = DTLZ_FUNC_MAP[dtid]
    return func(x, M)