from __future__ import annotations
import numpy as np


def sliding_windows_1d(
    x: np.ndarray,
    window_size: int,
    stride: int,
    *,
    drop_last: bool = True
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.shape[0]

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    last_start = n - window_size
    starts = np.arange(0, last_start + 1, stride, dtype=np.int64)

    if not drop_last:
        # include a final window ending at n if stride doesn't land exactly
        if starts.size == 0 or starts[-1] != last_start:
            starts = np.append(starts, last_start)

    windows = np.stack([x[s:s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32, copy=False)
