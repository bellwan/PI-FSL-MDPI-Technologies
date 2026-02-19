from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np
import torch
import os

def set_seed(seed: int) -> None:
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


def ensure_dir(path: str) -> None:

    os.makedirs(path, exist_ok=True)


def compact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != ""} 
