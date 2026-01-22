from __future__ import annotations
import random
from typing import Dict, List, Tuple
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stratified_split_indices(y: np.ndarray, *, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0,1)")

    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)

    by_class: Dict[int, List[int]] = {}
    for i, c in enumerate(y.tolist()):
        by_class.setdefault(int(c), []).append(i)

    train_idx: List[int] = []
    test_idx: List[int] = []

    for c, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_ratio)))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
