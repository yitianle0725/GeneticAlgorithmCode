from __future__ import annotations

import math
from collections.abc import Mapping


SOURCES = ("isolated", "shared", "recombination")
ADAPTIVE_BOUNDS = {
    "isolated": (0.30, 0.70),
    "shared": (0.10, 0.40),
    "recombination": (0.10, 0.30),
}


def _bounded_proportions(
    scores: Mapping[str, float],
    enabled_sources: tuple[str, ...],
) -> dict[str, float]:
    """把非负贡献分数投影到预设比例边界内。"""
    proportions = {source: 0.0 for source in SOURCES}
    active = list(enabled_sources)
    remaining = 1.0
    minimum = sum(ADAPTIVE_BOUNDS[source][0] for source in active)
    maximum = sum(ADAPTIVE_BOUNDS[source][1] for source in active)
    if minimum > 1.0 + 1e-12 or maximum < 1.0 - 1e-12:
        raise ValueError("启用来源的比例边界无法组成完整预算")

    while active and remaining > 1e-12:
        score_sum = sum(max(0.0, scores[source]) for source in active)
        if score_sum <= 1e-12:
            weights = {source: 1.0 / len(active) for source in active}
        else:
            weights = {
                source: max(0.0, scores[source]) / score_sum
                for source in active
            }

        proposed = {
            source: remaining * weights[source]
            for source in active
        }
        above_maximum = [
            source
            for source in active
            if proposed[source] > ADAPTIVE_BOUNDS[source][1] + 1e-12
        ]
        if above_maximum:
            for source in above_maximum:
                proportions[source] = ADAPTIVE_BOUNDS[source][1]
                remaining -= proportions[source]
            active = [source for source in active if source not in above_maximum]
            continue

        below_minimum = [
            source
            for source in active
            if proposed[source] < ADAPTIVE_BOUNDS[source][0] - 1e-12
        ]
        if below_minimum:
            for source in below_minimum:
                proportions[source] = ADAPTIVE_BOUNDS[source][0]
                remaining -= proportions[source]
            active = [source for source in active if source not in below_minimum]
            continue

        for source in active:
            proportions[source] = proposed[source]
        remaining = 0.0

    if remaining > 1e-9:
        raise ValueError("来源预算边界没有分配完全部比例")

    return proportions


def _integer_budget(total: int, proportions: Mapping[str, float]) -> dict[str, int]:
    """按最大余数法分配整数预算，并保证总数严格守恒。"""
    raw = {source: total * proportions[source] for source in SOURCES}
    budget = {source: math.floor(raw[source]) for source in SOURCES}
    unassigned = total - sum(budget.values())
    order = sorted(
        SOURCES,
        key=lambda source: (raw[source] - budget[source], -SOURCES.index(source)),
        reverse=True,
    )
    for source in order[:unassigned]:
        budget[source] += 1
    return budget


class SourceBudgetController:
    """管理 S3 三类后代的固定或贡献驱动预算。"""

    def __init__(
        self,
        ratios: Mapping[str, float],
        adaptive: bool,
        smoothing: float = 0.8,
    ) -> None:
        self.ratios = {source: float(ratios[source]) for source in SOURCES}
        if any(ratio < 0 for ratio in self.ratios.values()):
            raise ValueError("来源预算比例不能为负数")
        if not math.isclose(sum(self.ratios.values()), 1.0, abs_tol=1e-12):
            raise ValueError("来源预算比例之和必须等于 1")
        if not 0 <= smoothing < 1:
            raise ValueError("预算平滑系数必须在 [0, 1) 内")
        self.adaptive = bool(adaptive)
        self.smoothing = float(smoothing)
        self.scores = dict(self.ratios)
        self.enabled_sources = tuple(
            source for source in SOURCES if self.ratios[source] > 0
        )

    def allocate(self, total: int) -> dict[str, int]:
        if total < 0:
            raise ValueError("总预算不能为负数")
        proportions = (
            _bounded_proportions(self.scores, self.enabled_sources)
            if self.adaptive
            else self.ratios
        )
        return _integer_budget(total, proportions)

    def update(self, contributions: Mapping[str, float]) -> None:
        if not self.adaptive:
            return
        for source in SOURCES:
            contribution = max(0.0, float(contributions[source]))
            self.scores[source] = (
                self.smoothing * self.scores[source]
                + (1.0 - self.smoothing) * contribution
            )
