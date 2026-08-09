# fitness_function.py
import numpy as np
from individual import Individual

# ===================== 底层目标计算函数 F1~F18 =====================
# F1 DeJong1 球函数
def compute_f1(x: np.ndarray) -> float:
    return np.sum(np.square(x))

# F2 Rosenbrock香蕉函数
def compute_f2(x: np.ndarray) -> float:
    total = 0.0
    dim = len(x)
    for i in range(dim - 1):
        xi = x[i]
        xi1 = x[i+1]
        term1 = 100 * (xi1 - xi ** 2) ** 2
        term2 = (1 - xi) ** 2
        total += term1 + term2
    return total

# F3 DeJong3 阶跃函数
def compute_f3(x: np.ndarray) -> float:
    return np.sum(np.floor(x))

# F4 四次多项式（噪声默认关闭）
def compute_f4(x: np.ndarray, add_gauss_noise: bool = False) -> float:
    dim = len(x)
    total = 0.0
    for i in range(dim):
        weight = i + 1
        total += weight * (x[i] ** 4)
    if add_gauss_noise:
        total += np.random.normal(loc=0, scale=1)
    return total

# F5 DeJong5 2维多峰
def compute_f5(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    a = np.array([
        [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
        [-32, -32, -32, -32, -32, -16, -16, 0, 32, 32]
    ])
    sum_f = 0.0
    for j in range(10):
        aj1, aj2 = a[0, j], a[1, j]
        fj = (x1 - aj1)**6 + (x2 - aj2)**6
        cj = j + 1
        sum_f += 1 / (cj + fj)
    res = 1 / (0.002 + sum_f)
    return res

# F6 Schaffer F6
def compute_f6(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    r_sq = x1**2 + x2**2
    numerator = np.sin(np.sqrt(r_sq)) ** 2 - 0.5
    denominator = (1 + 0.001 * r_sq) ** 2
    return 0.5 + numerator / denominator

# F7 Schaffer F7
def compute_f7(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    r_sq = x1**2 + x2**2
    term1 = r_sq ** 0.25
    term2 = np.sin(50 * (r_sq ** 0.1)) ** 2 + 1.0
    return term1 * term2

# F8 Goldstein-Price
def compute_f8(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    part1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    part2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return part1 * part2

# F9 Branin RCOS
def compute_f9(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    d = 6
    e = 10
    f = 1 / (8 * np.pi)
    term1 = a * (x2 - b * x1**2 + c * x1 - d) ** 2
    term2 = e * (1 - f) * np.cos(x1)
    return term1 + term2 + e

# F11 六峰骆驼函数 six-hump camel back
def compute_f11(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    t1 = (4 - 2.1 * x1**2 + (1/3) * x1**4) * x1**2
    t2 = x1 * x2
    t3 = (-4 + 4 * x2**2) * x2**2
    return t1 + t2 + t3

# F12 Shubert 函数
def compute_f12(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    sum1 = 0.0
    sum2 = 0.0
    for i in range(1, 6):
        sum1 += i * np.cos((i + 1) * x1 + i)
        sum2 += i * np.cos((i + 1) * x2 + i)
    return sum1 * sum2

# ========== 新增 F14 Easom 函数 ==========
def compute_f14(x: np.ndarray) -> float:
    """
    F14 Easom函数，2维
    定义域：-100 ≤ x1,x2 ≤ 100
    全局最小值：f(π, π) = -1
    特点：全局最优点区域极窄，极易早熟
    """
    x1, x2 = x[0], x[1]
    cos_part = np.cos(x1) * np.cos(x2)
    exp_part = np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))
    return -cos_part * exp_part

# ========== 新增 F18 Colville 函数 ==========
def compute_f18(x: np.ndarray) -> float:
    """
    F18 Colville函数，4维
    定义域：-10 ≤ xi ≤ 10
    全局最小值：f(1,1,1,1) = 0
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    t1 = 100 * (x2 - x1**2) ** 2
    t2 = (1 - x1) ** 2
    t3 = 90 * (x4 - x3**2) ** 2
    t4 = (1 - x3) ** 2
    t5 = 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
    t6 = 19.8 * (x2 - 1) * (x4 - 1)
    return t1 + t2 + t3 + t4 + t5 + t6

# ===================== 统一总入口：eval_func 支持1/2/3/4/5/6/7/8/9/11/12/14/18 =====================
def eval_func(ind: Individual, func_id: int):
    """
    统一评估接口，一个函数切换所有测试函数
    :param ind: 种群个体
    :param func_id: 1=F1,2=F2,3=F3,4=F4,5=F5,6=F6,7=F7,8=F8,9=F9,11=F11,12=F12,14=F14,18=F18
    """
    x = np.array(ind.gene)
    if func_id == 1:
        f_val = compute_f1(x)
    elif func_id == 2:
        f_val = compute_f2(x)
    elif func_id == 3:
        f_val = compute_f3(x)
    elif func_id == 4:
        f_val = compute_f4(x, add_gauss_noise=False)
    elif func_id == 5:
        f_val = compute_f5(x)
    elif func_id == 6:
        f_val = compute_f6(x)
    elif func_id == 7:
        f_val = compute_f7(x)
    elif func_id == 8:
        f_val = compute_f8(x)
    elif func_id == 9:
        f_val = compute_f9(x)
    elif func_id == 11:
        f_val = compute_f11(x)
    elif func_id == 12:
        f_val = compute_f12(x)
    elif func_id == 14:
        f_val = compute_f14(x)
    elif func_id == 18:
        f_val = compute_f18(x)
    else:
        raise ValueError("func_id仅支持输入1/2/3/4/5/6/7/8/9/11/12/14/18")

    # 兼容原有4目标框架，仅obj[0]存目标值，其余置0
    ind.obj[0] = f_val
    for i in range(1, ind.config.NOBJ):
        ind.obj[i] = 0.0
    # 缓存原始值用于日志、收敛曲线绘图
    ind.raw_F1 = f_val
    ind.raw_F2 = 0.0
    ind.raw_F3 = 0.0
    ind.raw_F4 = 0.0