from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ZScoreParams:
    mean: float
    std: float


def zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(np.mean(x))
    sd = float(np.std(x) + eps)
    return ((x - mu) / sd).astype(np.float32)


def zscore_fit_1d(x: np.ndarray, eps: float = 1e-8) -> ZScoreParams:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(np.mean(x))
    sd = float(np.std(x) + eps)
    return ZScoreParams(mean=mu, std=sd)


def zscore_apply_1d(x: np.ndarray, params: ZScoreParams) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - params.mean) / (params.std + 1e-8)).astype(np.float32)
