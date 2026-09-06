from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class ObjectiveNormalization:
    """一轮外循环共享的目标归一化上下文。"""

    ideal: np.ndarray
    nadir: np.ndarray

    @classmethod
    def from_objectives(cls, F: np.ndarray) -> ObjectiveNormalization:
        values = np.asarray(F, dtype=float)
        return cls(ideal=np.min(values, axis=0), nadir=np.max(values, axis=0))

    def apply(self, F: np.ndarray) -> np.ndarray:
        values = np.asarray(F, dtype=float)
        return (values - self.ideal) / np.maximum(self.nadir - self.ideal, EPSILON)

    def tchebycheff(self, F: np.ndarray, weight: np.ndarray) -> np.ndarray:
        normalized = self.apply(F)
        return np.max(np.asarray(weight, dtype=float) * np.abs(normalized), axis=1)
