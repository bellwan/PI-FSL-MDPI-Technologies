from __future__ import annotations
import math
import random
from typing import Iterable, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def mean_ci95(values: Iterable[float]) -> Tuple[float, Optional[float]]:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return float("nan"), None
    m = float(vals.mean())
    if vals.size < 2:
        return m, None
    s = float(vals.std(ddof=1))
    ci = 1.96 * s / math.sqrt(vals.size)
    return m, float(ci)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
