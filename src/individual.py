"""
多目标优化个体类

每个个体包含：
- 基因（决策变量值）
- 目标函数值
- 支配关系相关属性（rank、crowd_dist）
- 原始目标值缓存（用于日志输出）

支持方法：
- copy_from: 从另一个个体复制属性
- dominates: 判断是否支配另一个个体
"""


class Individual:
    """
    多目标遗传算法个体类
    
    属性说明：
    - gene: 决策变量向量，长度为NVARS
    - lower: 决策变量下界向量
    - upper: 决策变量上界向量
    - obj: 目标函数值向量，长度为NOBJ（已归一化）
    - rank: 非支配排序等级（0表示Pareto最优前沿）
    - crowd_dist: 拥挤度距离（用于精英保留选择）
    - raw_F1~F4: 原始目标值缓存（未归一化，用于日志查看）
    """

    def __init__(self, config):
        """
        初始化个体
        
        参数：
        - config: GAConfig配置对象，提供NVARS、NOBJ、VAR_LB、VAR_UB等参数
        """
        self.config = config

        # 决策变量相关
        self.gene = [0.0] * self.config.NVARS
        self.lower = [self.config.VAR_LB] * self.config.NVARS
        self.upper = [self.config.VAR_UB] * self.config.NVARS

        # 目标函数值（归一化后）
        self.obj = [0.0] * self.config.NOBJ

        # 非支配排序属性
        self.rank = 0
        self.crowd_dist = 0.0

        # 原始目标值缓存（未归一化，用于日志输出）
        # 7目标原始缓存
        self.raw_F1 = 0.0
        self.raw_F2 = 0.0
        self.raw_F3 = 0.0
        self.raw_F4 = 0.0
        self.raw_F5 = 0.0
        self.raw_F6 = 0.0
        self.raw_F7 = 0.0

    def copy_from(self, other):
        """
        从另一个个体复制所有属性
        
        参数：
        - other: 源个体对象
        """
        self.gene = other.gene.copy()
        self.lower = other.lower.copy()
        self.upper = other.upper.copy()
        self.obj = other.obj.copy()
        self.rank = other.rank
        self.crowd_dist = other.crowd_dist

        self.raw_F1 = other.raw_F1
        self.raw_F2 = other.raw_F2
        self.raw_F3 = other.raw_F3
        self.raw_F4 = other.raw_F4
        self.raw_F5 = other.raw_F5
        self.raw_F6 = other.raw_F6
        self.raw_F7 = other.raw_F7

    def dominates(self, other):
        """
        判断当前个体是否支配另一个个体
        
        支配定义：
        - 当前个体的所有目标函数值都不大于另一个体
        - 当前个体至少有一个目标函数值严格小于另一个体
        
        参数：
        - other: 被比较的个体对象
        
        返回：
        - bool: True表示当前个体支配另一个体，False表示不支配
        """
        all_less_eq = True
        any_less = False
        for m in range(self.config.NOBJ):
            if self.obj[m] > other.obj[m]:
                all_less_eq = False
            if self.obj[m] < other.obj[m]:
                any_less = True
        return all_less_eq and any_less
