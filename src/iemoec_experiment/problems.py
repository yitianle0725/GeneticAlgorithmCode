from __future__ import annotations

import re

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems import get_problem


_STANDARD = re.compile(r"^(dtlz[1-7]|wfg[1-9]|zdt[1-6])$")


class ConvexDTLZ2(Problem):
    """保留旧实验的 C-DTLZ2 目标变换，基础 DTLZ2 由 pymoo 计算。"""

    def __init__(self, n_var: int, n_obj: int):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.base = get_problem("dtlz2", n_var=n_var, n_obj=n_obj)

    def _evaluate(self, x, out, *args, **kwargs):
        base_f = self.base.evaluate(x, return_values_of=["F"])
        radius = np.linalg.norm(base_f, axis=1, keepdims=True)
        out["F"] = radius - base_f

    def _calc_pareto_front(self, ref_dirs=None, *args, **kwargs):
        if ref_dirs is None:
            return None
        base_pf = self.base.pareto_front(ref_dirs=ref_dirs)
        return 1.0 - base_pf

    def _calc_pareto_set(self, *args, **kwargs):
        return None

    def ideal_point(self):
        return np.zeros(self.n_obj)

    def nadir_point(self):
        return np.ones(self.n_obj)


def make_problem(name: str, n_obj: int, n_var: int | None = None) -> Problem:
    """只对旧 C-DTLZ2 保留薄包装，其余问题直接交给 pymoo。"""
    normalized = name.lower().replace("-", "")
    if normalized in ("cdtlz2", "convexdtlz2"):
        return ConvexDTLZ2(n_var=n_var or n_obj + 9, n_obj=n_obj)
    if not _STANDARD.match(normalized):
        raise ValueError("问题名须为 DTLZ1-7、WFG1-9、ZDT1-6 或 C-DTLZ2")
    if normalized.startswith("zdt"):
        if n_obj != 2:
            raise ValueError("ZDT 仅支持 M=2")
        kwargs = {"n_var": n_var} if n_var is not None else {}
        return get_problem(normalized, **kwargs)
    if normalized.startswith("dtlz"):
        if n_var is None:
            if normalized == "dtlz1":
                n_var = n_obj + 4   # 标准 k=5
            elif normalized == "dtlz7":
                n_var = n_obj + 19  # 标准 k=20
            else:
                n_var = n_obj + 9   # 标准 k=10
        return get_problem(normalized, n_var=n_var, n_obj=n_obj)

    # WFG 要求 k 可被 M-1 整除；2(M-1) 是 pymoo 示例中的稳定选择。
    k = 2 * (n_obj - 1)
    resolved_n_var = n_var or (k + 20)
    if resolved_n_var <= k:
        raise ValueError(f"WFG 的 n_var 必须大于 k={k}")
    return get_problem(normalized, n_var=resolved_n_var, n_obj=n_obj, k=k)
