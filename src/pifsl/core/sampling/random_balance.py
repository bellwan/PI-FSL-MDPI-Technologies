import numpy as np
from typing import List, Tuple

class RandomClassBalancedSampler:
    def __init__(self, target_imbalance_ratio: float = 0.3, seed: int = 42):
        self.target_ratio = float(target_imbalance_ratio)
        self.rng = np.random.default_rng(int(seed))

    def balance_dataset(self, X: List[np.ndarray], y: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        y_np = np.asarray(y, dtype=int)
        idx0 = np.where(y_np == 0)[0]
        idx1 = np.where(y_np == 1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return X, y

        # Determine majority/minority
        if idx0.size >= idx1.size:
            maj_idx, maj_label = idx0, 0
            min_idx, min_label = idx1, 1
        else:
            maj_idx, maj_label = idx1, 1
            min_idx, min_label = idx0, 0

        min_ct = int(min_idx.size)
        if self.target_ratio <= 0:
            target_maj_ct = min_ct
        else:
            target_maj_ct = int(np.floor(min_ct / self.target_ratio))
            target_maj_ct = max(min_ct, target_maj_ct)
        target_maj_ct = min(int(maj_idx.size), target_maj_ct)

        keep_maj = self.rng.choice(maj_idx, size=target_maj_ct, replace=False)
        keep_idx = np.concatenate([min_idx, keep_maj])
        self.rng.shuffle(keep_idx)

        X_out = [X[int(i)] for i in keep_idx]
        y_out = [int(y_np[int(i)]) for i in keep_idx]
        return X_out, y_out
