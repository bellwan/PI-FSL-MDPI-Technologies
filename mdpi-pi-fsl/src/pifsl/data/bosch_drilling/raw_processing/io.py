import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

import h5py
from scipy.signal import welch, resample

from pifsl.data.bosch_drilling import config
DATASET_KEYS_TRY = ["vibration_data", "AI0", "vibration", "acceleration"]

# ------------------------------------------------------------------
# H5 readers 
# ------------------------------------------------------------------

def read_h5_tri(fp: str) -> np.ndarray:
    fp_str = str(fp)

    if fp_str.lower().endswith(".npz"):
        # NPZ path: acc_xyz stored by h5_to_npz.py
        with np.load(fp_str, allow_pickle=False) as z:
            if "acc_xyz" not in z:
                raise RuntimeError(f"{os.path.basename(fp_str)}: 'acc_xyz' missing in npz")
            A = np.asarray(z["acc_xyz"])
    else:
        # Original H5 path
        with h5py.File(fp_str, "r") as f:
            if config.ACC_KEY not in f:
                raise RuntimeError(f"{os.path.basename(fp_str)}: '{config.ACC_KEY}' missing")
            A = np.asarray(f[config.ACC_KEY][()])

    A = np.array(A, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim == 2 and A.shape[0] == 3 and A.shape[1] != 3:
        A = A.T
    if A.shape[1] == 1:
        A = np.repeat(A, 3, axis=1)
    if A.ndim != 2 or A.shape[1] != 3:
        raise RuntimeError(f"{os.path.basename(fp_str)}: shape {A.shape} != (N,3)")
    return A

# Robust reader used by diagnostics
def read_any_h5(fp: Path | str) -> np.ndarray:
    fp = Path(fp)

    if fp.suffix.lower() == ".npz":
        with np.load(fp, allow_pickle=False) as z:
            arr = None
            if "acc_xyz" in z:
                arr = np.asarray(z["acc_xyz"])
            else:
                files = list(z.files)
                if not files:
                    raise RuntimeError(f"No arrays in {fp.name}")
                arr = np.asarray(z[files[0]])
    else:
        with h5py.File(fp, "r") as f:
            arr = None
            if config.ACC_KEY in f:
                arr = np.asarray(f[config.ACC_KEY][()])
            else:
                for k in DATASET_KEYS_TRY:
                    if k in f:
                        arr = np.asarray(f[k][()])
                        break
                if arr is None:
                    keys = list(f.keys())
                    if not keys:
                        raise RuntimeError(f"No datasets in {fp.name}")
                    arr = np.asarray(f[keys[0]][()])

    a = np.array(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] == 1:
        return a
    if a.shape[1] >= 3:
        return a[:, :3]
    return a.reshape(-1, 1)

# ------------------------------------------------------------------
# DSP helpers
# ------------------------------------------------------------------
def compute_psd(x: np.ndarray, fs: float, max_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    fs = float(fs)
    max_hz = float(max_hz)

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 8:
        return np.array([]), np.array([])

    f, Pxx = welch(
        x,
        fs=fs,
        nperseg=min(2048, max(256, len(x) // 2))
    )

    hi_eff = min(max_hz, 0.49 * fs)
    m = (f > 0) & (f <= hi_eff)
    if not np.any(m):
        return np.array([]), np.array([])

    return f[m], np.clip(Pxx[m], 1e-20, None)

