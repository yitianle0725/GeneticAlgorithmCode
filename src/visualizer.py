# visualizer.py
import os
import matplotlib.pyplot as plt
import numpy as np
class GAVisualizer:
    @staticmethod
    def plot_igd_curve(gen_list, igd_list, save_dir=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 4))
        plt.plot(gen_list, igd_list, "g-", linewidth=2, label="IGD指标")
        plt.xlabel("迭代代数 Generation")
        plt.ylabel("IGD值（越小收敛越好）")
        plt.title("IGD收敛曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            igd_png_path = os.path.join(save_dir, "igd_convergence.png")
            plt.savefig(igd_png_path, dpi=300, bbox_inches="tight")
        # plt.show()

    # 新增 GD 收敛曲线
    @staticmethod
    def plot_gd_curve(gen_list, gd_list, save_dir=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 4))
        plt.plot(gen_list, gd_list, "orange", linewidth=2, label="GD指标（越小收敛越好）")
        plt.xlabel("迭代代数 Generation")
        plt.ylabel("GD 世代距离")
        plt.title("GD收敛曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            gd_png_path = os.path.join(save_dir, "gd_convergence.png")
            plt.savefig(gd_png_path, dpi=300, bbox_inches="tight")
        # plt.show()

    @staticmethod
    def plot_pareto_front(population, save_dir=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        front0 = [ind for ind in population if ind.rank == 0]
        f1 = [ind.raw_F1 for ind in front0]
        f2 = [ind.raw_F2 for ind in front0]
        plt.figure(figsize=(8, 6))
        plt.scatter(f1, f2, c="red", s=18, label="Pareto最优解集")
        plt.xlabel("目标1 f1")
        plt.ylabel("目标2 f2")
        plt.title("Pareto前沿分布")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            pareto_png_path = os.path.join(save_dir, "pareto_front.png")
            plt.savefig(pareto_png_path, dpi=300, bbox_inches="tight")
        # plt.show()

    # ===================== HV 收敛曲线 =====================
    @staticmethod
    def plot_hv_curve(gen_list, hv_list, save_dir=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 4))
        plt.plot(gen_list, hv_list, "darkblue", linewidth=2, label="HV指标（越高越好）")
        plt.xlabel("迭代代数 Generation")
        plt.ylabel("HV 超体积")
        plt.title("HV超体积收敛曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            hv_png_path = os.path.join(save_dir, "hv_convergence.png")
            plt.savefig(hv_png_path, dpi=300, bbox_inches="tight")
        # plt.show()

    # ===================== SP(Spacing) 收敛曲线 =====================
    @staticmethod
    def plot_sp_curve(gen_list, sp_list, save_dir=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 4))
        plt.plot(gen_list, sp_list, "crimson", linewidth=2, label="SP间距指标（越小分布越均匀）")
        plt.xlabel("迭代代数 Generation")
        plt.ylabel("SP Spacing")
        plt.title("SP解集均匀性收敛曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            sp_png_path = os.path.join(save_dir, "sp_convergence.png")
            plt.savefig(sp_png_path, dpi=300, bbox_inches="tight")
        # plt.show()

    # 新增 ONVG 非支配解数量曲线
    @staticmethod
    def plot_onvg_curve(gen_list, onvg_list, save_dir=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 4))
        plt.plot(gen_list, onvg_list, "purple", linewidth=2, label="ONVG 非支配解数量")
        plt.xlabel("迭代代数 Generation")
        plt.ylabel("第一层非支配个体数")
        plt.title("ONVG非支配解数量变化曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            onvg_png_path = os.path.join(save_dir, "onvg_convergence.png")
            plt.savefig(onvg_png_path, dpi=300, bbox_inches="tight")
        # plt.show()

    # ===================== 高维 Pareto 前沿 — 平行坐标图 =====================
    @staticmethod
    def plot_pareto_front_parallel(population, save_dir=None):
        """当目标维度 M >= 3 时, 使用平行坐标图可视化 Pareto 前沿"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        front0 = [ind for ind in population if ind.rank == 0]
        if len(front0) == 0:
            print("⚠️ 没有 rank=0 的个体, 跳过平行坐标图")
            return
        M = front0[0].config.NOBJ
        # 提取所有 Pareto 解的目标值矩阵: N x M
        F = np.array([[ind.obj[m] for m in range(M)] for ind in front0])

        # 对每个目标维度做 min-max 归一化, 使平行坐标可读
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        denom = f_max - f_min
        denom[denom < 1e-10] = 1.0
        F_norm = (F - f_min) / denom

        fig, ax = plt.subplots(figsize=(12, 5))
        x_ticks = list(range(1, M + 1))
        # 每条折线是一个 Pareto 最优解
        for i in range(len(F_norm)):
            alpha_val = min(0.6, max(0.05, 20.0 / len(F_norm)))
            ax.plot(x_ticks, F_norm[i], color="steelblue", alpha=alpha_val,
                    linewidth=0.8, marker='o', markersize=2)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"f{m + 1}" for m in range(M)])
        ax.set_xlabel("目标维度")
        ax.set_ylabel("归一化目标值 (0–1)")
        ax.set_title(f"Pareto前沿平行坐标图 (M={M}, {len(F_norm)} 个非支配解)")
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if save_dir is not None:
            para_path = os.path.join(save_dir, "pareto_parallel.png")
            plt.savefig(para_path, dpi=300, bbox_inches="tight")
            print(f"[OK] 平行坐标图已保存: {para_path}")
        # plt.show()