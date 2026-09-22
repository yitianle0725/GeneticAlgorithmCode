from __future__ import annotations

import re

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems import get_problem


_STANDARD = re.compile(r"^(dtlz[1-7]|wfg[1-9]|zdt[1-6])$")
_CONSTRAINED_DTLZ = re.compile(r"^(c1dtlz1|c1dtlz3|c2dtlz2|c3dtlz4)$")
_DASCMOP = re.compile(r"^dascmop([1-9])_d(\d{1,2})$")


def is_constrained_problem_name(name: str) -> bool:
    """判断规范化问题名是否属于当前受支持的约束测试集。"""
    normalized = name.lower().replace("-", "")
    return bool(_CONSTRAINED_DTLZ.match(normalized) or _DASCMOP.match(normalized))


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


def standard_problem_dimensions(name: str, n_obj: int) -> tuple[int, int, int | None]:
    """返回标准 n_var、k，以及仅 WFG 使用的 l。"""
    normalized = name.lower().replace("-", "")
    if normalized == "dtlz1":
        k = 5
        return n_obj + k - 1, k, None
    if normalized == "dtlz7":
        k = 20
        return n_obj + k - 1, k, None
    if normalized.startswith("dtlz"):
        k = 10
        return n_obj + k - 1, k, None
    if normalized.startswith("wfg"):
        k = 2 * (n_obj - 1)
        l = 20
        return k + l, k, l
    if normalized == "c1dtlz1":
        k = 5
        return n_obj + k - 1, k, None
    if normalized in ("c1dtlz3", "c2dtlz2"):
        k = 10
        return n_obj + k - 1, k, None
    if normalized == "c3dtlz4":
        k = 5
        return n_obj + k - 1, k, None
    raise ValueError(f"{name} 不是 DTLZ/WFG 问题")


def make_problem(name: str, n_obj: int, n_var: int | None = None) -> Problem:
    """构造统一命名的问题实例。"""
    normalized = name.lower().replace("-", "")
    if normalized == "cdtlz2":
        raise ValueError(
            "cdtlz2 名称有歧义：旧凸前沿问题请使用 convexdtlz2，"
            "约束问题请使用 c2dtlz2"
        )
    if normalized == "convexdtlz2":
        return ConvexDTLZ2(n_var=n_var or n_obj + 9, n_obj=n_obj)
    if _CONSTRAINED_DTLZ.match(normalized):
        resolved_n_var, _, _ = standard_problem_dimensions(normalized, n_obj)
        problem = get_problem(
            normalized,
            n_var=n_var or resolved_n_var,
            n_obj=n_obj,
        )
        problem._iemoec_problem_id = normalized
        return problem
    dascmop_match = _DASCMOP.match(normalized)
    if dascmop_match:
        problem_number = int(dascmop_match.group(1))
        difficulty = int(dascmop_match.group(2))
        expected_objectives = 2 if problem_number <= 6 else 3
        if n_obj != expected_objectives:
            raise ValueError(
                f"DASCMOP{problem_number} 仅支持 M={expected_objectives}"
            )
        if not 1 <= difficulty <= 16:
            raise ValueError("DAS-CMOP difficulty 必须在 1 到 16 之间")
        if n_var is not None and n_var != 30:
            raise ValueError("DAS-CMOP 固定使用 n_var=30")
        problem = get_problem(f"dascmop{problem_number}", difficulty)
        problem._iemoec_problem_id = normalized
        return problem
    if not _STANDARD.match(normalized):
        raise ValueError(
            "问题名须为 DTLZ1-7、WFG1-9、ZDT1-6、约束 C-DTLZ、"
            "DASCMOP<n>_d<difficulty> 或旧 C-DTLZ2"
        )
    if normalized.startswith("zdt"):
        if n_obj != 2:
            raise ValueError("ZDT 仅支持 M=2")
        kwargs = {"n_var": n_var} if n_var is not None else {}
        return get_problem(normalized, **kwargs)
    if normalized.startswith("dtlz"):
        if n_var is None:
            n_var, _, _ = standard_problem_dimensions(normalized, n_obj)
        return get_problem(normalized, n_var=n_var, n_obj=n_obj)

    # WFG 要求 k 可被 M-1 整除；2(M-1) 是 pymoo 示例中的稳定选择。
    standard_n_var, k, _ = standard_problem_dimensions(normalized, n_obj)
    resolved_n_var = n_var or standard_n_var
    if resolved_n_var <= k:
        raise ValueError(f"WFG 的 n_var 必须大于 k={k}")
    return get_problem(normalized, n_var=resolved_n_var, n_obj=n_obj, k=k)
