
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, welch, resample
import pywt
from typing import List, Tuple
from pifsl.data.bosch_drilling import config
from pifsl.data.bosch_drilling.raw_processing.io import read_h5_tri
import numpy as np

def robust_norm(x: np.ndarray) -> np.ndarray:
    q25, q75 = np.percentile(x,25), np.percentile(x,75)
    return (x - np.median(x)) / ((q75-q25)+1e-12)

def envelope_track(xyz: np.ndarray, fs: float, lowpass_hz: float = 10.0) -> np.ndarray:
    mag = np.linalg.norm(xyz, axis=1)
    env = np.abs(hilbert(mag))
    wlen = max(3, int(fs / max(lowpass_hz, 1.0)))
    ker = np.ones(wlen, dtype=np.float64) / wlen
    env_lp = np.convolve(env, ker, mode="same")
    return env_lp

def bandpower_block_axis(seg: np.ndarray,
                         fs: float,
                         lo: float,
                         hi: float) -> float:
    # Ensure numeric
    fs = float(fs)
    lo = float(lo)
    hi = float(hi)

    if seg.ndim != 1:
        seg = np.asarray(seg).ravel()

    # Effective upper bound below Nyquist
    hi_eff = min(float(hi), 0.49 * fs)
    if hi_eff <= lo:
        return 0.0

    f, Pxx = welch(
        seg,
        fs=fs,
        nperseg=min(256, len(seg)),
        noverlap=None
    )

    m = (f >= lo) & (f < hi_eff)
    if not np.any(m):
        return 0.0

    return float(np.trapz(Pxx[m], f[m]))

def hf_lf_ratio(xyz: np.ndarray, fs: float,
                hf=(75.0,1000.0), lf=(0.0,75.0)) -> np.ndarray:
    N = len(xyz); blk = int(fs)
    if N < blk:
        hfE = sum(bandpower_block_axis(xyz[:,c], fs, *hf) for c in range(3))
        lfE = sum(bandpower_block_axis(xyz[:,c], fs, *lf) for c in range(3))
        return np.full(N, hfE/(lfE+1e-12), dtype=np.float64)
    vals=[]
    for i in range(0, N, blk):
        seg = xyz[i:i+blk]
        hfE = sum(bandpower_block_axis(seg[:,c], fs, *hf) for c in range(3))
        lfE = sum(bandpower_block_axis(seg[:,c], fs, *lf) for c in range(3))
        vals.append(hfE/(lfE+1e-12))
    return np.repeat(np.array(vals, dtype=np.float64), blk)[:N]

def tri_state(env_lp: np.ndarray, ratio: np.ndarray,
              thrE_lo_q: float, thrR_q: float) -> np.ndarray:
    E = robust_norm(env_lp); R = robust_norm(ratio)
    thrE_lo = np.quantile(E, thrE_lo_q); thrR = np.quantile(R, thrR_q)
    s = np.zeros_like(E, dtype=np.int32)
    s[(R >= thrR) & (E >= thrE_lo)] = 2
    s[(s==0) & (E >= thrE_lo) & (R < thrR)] = 1
    return s

def fixed_windows(N:int, win:int, hop:int):
    for st in range(0, max(0, N-win+1), hop):
        en = st + win; ce = st + win//2
        yield st, en, ce

def order_normalize(sig: np.ndarray, op: str) -> np.ndarray:
    fac = config.ORDER_NORM.get(op, 1.0)
    if abs(fac - 1.0) < 1e-6:
        return sig
    new_len = max(8, int(round(len(sig) * fac)))
    return resample(sig, new_len)

def scalogram_64x64(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    x = (signal_1d - np.mean(signal_1d)) / (np.std(signal_1d) + 1e-12)
    coef, _ = pywt.cwt(x, scales=config.CWT_SCALES, wavelet=config.CWT_WAVELET, sampling_period=1.0/fs)
    S = np.abs(coef)
    S_r = resample(S, config.IMG_H, axis=0); S_r = resample(S_r, config.IMG_W, axis=1)
    S_r = S_r.astype(np.float32)
    S_r = np.log1p(S_r - np.min(S_r) + 1e-6)
    S_r = (S_r - np.mean(S_r)) / (np.std(S_r) + 1e-12)
    return S_r

def quick_psd_embed(seg: np.ndarray, fs: float, bins: int = 64) -> np.ndarray:
    fs = float(fs)

    if seg.ndim != 1:
        seg = np.asarray(seg).ravel()

    f, Pxx = welch(
        seg,
        fs=fs,
        nperseg=min(256, len(seg)),
        noverlap=None
    )

    # Limit to a safe band below Nyquist
    max_band = min(1000.0, 0.49 * fs)
    m = (f >= 1.0) & (f <= max_band)
    if not np.any(m):
        # degenerate but safe fallback
        return np.zeros(bins, dtype=np.float32)

    f = f[m]
    Pxx = Pxx[m]

    # Geometric frequency bins
    f_min = max(1.0, float(f.min()))
    edges = np.geomspace(f_min, float(max_band), bins + 1)

    v = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        mm = (f >= edges[i]) & (f < edges[i + 1])
        if np.any(mm):
            v[i] = np.mean(np.log(Pxx[mm] + 1e-20))
        else:
            v[i] = np.log(1e-20)

    v = (v - np.mean(v)) / (np.std(v) + 1e-12)
    return v.astype(np.float32)

def build_windows_for_file(fp: str, op_hint:str):
    A = read_h5_tri(fp)
    env_lp = envelope_track(A, config.FS, config.ENV_LOWPASS_HZ)
    ratio  = hf_lf_ratio(A, config.FS)
    s      = tri_state(env_lp, ratio, config.TRI_thrE_lo_q, config.TRI_thrR_q)
    Xs=[]
    for (WIN, HOP) in config.WINDOW_SPECS:
        for st,en,ce in fixed_windows(len(A), WIN, HOP):
            if (en-st)!=WIN or s[ce]!=2:
                continue
            seg = A[st:en]
            mag = np.linalg.norm(seg, axis=1).astype(np.float32)
            mag = order_normalize(mag, op_hint)
            Xs.append(mag)
    return Xs

def collect_windows(inv):
    X=[]; y=[]
    for r in inv.itertuples(index=False):
        try:
            wins = build_windows_for_file(r.file_path, op_hint=r.ProcessName)
        except Exception:
            continue
        for w in wins:
            X.append(w); y.append(int(r.Label))
    return X, y

def make_embed_matrix_for_cnn(X_windows, fs: float):
    return np.vstack([quick_psd_embed(w, fs, config.PSD_BINS) for w in X_windows]).astype(np.float32)
