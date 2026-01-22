from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re
import numpy as np
import pandas as pd
from pifsl.core.base import DatasetBundle
import re

_DOM_RE = re.compile(r"(\d+Hz|0-40-0Hz)$", re.IGNORECASE)

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(x.mean())
    sd = float(x.std() + 1e-8)
    return ((x - mu) / sd).astype(np.float32)

def _parse_domain(stem: str) -> str:
    s = stem.replace(" ", "")
    m = _DOM_RE.search(s)
    if m:
        return m.group(1)
    toks = re.split(r"[_\-]+", s)
    return toks[-1]

def _parse_class_9(stem: str) -> int:
    s = stem.upper()
    dom = _parse_domain(stem).upper()
    s = s.replace(dom, "").strip("_-")
    sev = "0.5X" if "0.5X" in s else ("1X" if "1X" in s else "1X")
    code = None
    for c in ["H","I","O","B","C"]:
        if re.search(rf"(^|[_\-]){c}($|[_\-])", s) or s == c:
            code = c
            break
    if code is None:
        toks = re.split(r"[_\-]+", s)
        for t in toks:
            if t in ("H","I","O","B","C"):
                code = t
                break
    if code is None:
        raise ValueError(f"Cannot parse HUST-CN class from: {stem}")
    if code == "H":
        return 0
    mapping = {
        ("0.5X","I"): 1, ("1X","I"): 2,
        ("0.5X","O"): 3, ("1X","O"): 4,
        ("0.5X","B"): 5, ("1X","B"): 6,
        ("0.5X","C"): 7, ("1X","C"): 8,
    }
    return mapping[(sev, code)]

def _read_vibration_excel(p: Path) -> np.ndarray:
    suffix = p.suffix.lower()

    # -----------------------------
    # 1) Try reading as Excel
    # -----------------------------
    df = None
    if suffix == ".xlsx":
        try:
            df = pd.read_excel(p, header=None, engine="openpyxl")
        except Exception:
            df = None
    elif suffix == ".xls":
        try:
            df = pd.read_excel(p, header=None, engine="xlrd")
        except Exception:
            df = None
    else:
        try:
            df = pd.read_excel(p, header=None)
        except Exception:
            df = None

    if df is not None:
        arr = df.to_numpy()
        cols = []
        for c in range(arr.shape[1]):
            col = pd.to_numeric(arr[:, c], errors="coerce")
            if np.isfinite(col).sum() > 0:
                cols.append(np.asarray(col, dtype=np.float64))
        if cols:
            best = max(cols, key=lambda v: float(np.nanstd(v)))
            best = best[np.isfinite(best)]
            return best.reshape(-1)

    # -----------------------------
    # 2) Fallback: parse as text and extract floats by regex
    # -----------------------------

    float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

    try:
        raw = p.read_bytes()
        # try a few common encodings
        for enc in ("utf-8", "utf-16", "gbk", "latin1"):
            try:
                text = raw.decode(enc, errors="strict")
                break
            except Exception:
                text = None
        if text is None:
            text = raw.decode("latin1", errors="ignore")

        lines = text.splitlines()
        sequences = []
        for ln in lines:
            nums = float_re.findall(ln)
            if len(nums) >= 1:
                sequences.append([float(x) for x in nums])

        if not sequences:
            raise RuntimeError("no numeric tokens found")

        # Choose the longest numeric sequence as signal candidate
        best_seq = max(sequences, key=len)
        sig = np.asarray(best_seq, dtype=np.float64)
        sig = sig[np.isfinite(sig)]
        if sig.size < 64:
            # If the longest line is short, concatenate all sequences
            sig2 = np.asarray([v for seq in sequences for v in seq], dtype=np.float64)
            sig2 = sig2[np.isfinite(sig2)]
            if sig2.size >= sig.size:
                sig = sig2

        if sig.size == 0:
            raise RuntimeError("numeric tokens parsed but empty after filtering")

        return sig.reshape(-1)

    except Exception as e:
        raise ValueError(f"Cannot parse HUST-CN file: {p} ({e})")


def _window(x: np.ndarray, win: int, stride: int) -> List[np.ndarray]:
    if x.size < win:
        return []
    out = []
    for i in range(0, x.size - win + 1, stride):
        out.append(x[i:i+win].copy())
    return out

class HUSTCNAdapter:
    name = "hust_cn"

    def load_windows(
        self,
        data_root: str,
        source_domain: str,
        target_domain: str,
        window_seconds: float = 0.08,
        overlap_ratio: float = 0.5,
        normalization: str = "per_window",
        fs_override: float = 25600.0,
        label_mode: str = "binary",  
    ) -> Tuple[DatasetBundle, DatasetBundle]:
        root = Path(data_root)
        files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".xlsx",".xls")])
        if not files:
            raise RuntimeError(f"No Excel files under {root}")
        fs = float(fs_override)
        win = max(256, int(round(window_seconds * fs)))
        stride = max(1, int(round(win * (1.0 - overlap_ratio))))

        Xs: List[np.ndarray] = []; ys: List[int] = []; fids: List[str] = []
        Xt: List[np.ndarray] = []; yt: List[int] = []; fidt: List[str] = []

        for p in files:
            dom = _parse_domain(p.stem)
            if dom not in (source_domain, target_domain):
                continue
            cls9 = _parse_class_9(p.stem)
            y = 0 if cls9 == 0 else 1 if label_mode == "binary" else cls9
            try:
                sig = _read_vibration_excel(p)
            except Exception as e:
                print(f"[HUST-CN][SKIP] {p.name}: {e}")
                continue
            wins = _window(sig, win, stride)
            if normalization == "per_window":
                wins = [_zscore(w) for w in wins]
            if dom == source_domain:
                Xs.extend([w.astype(np.float32) for w in wins]); ys.extend([y]*len(wins)); fids.extend([p.name]*len(wins))
            else:
                Xt.extend([w.astype(np.float32) for w in wins]); yt.extend([y]*len(wins)); fidt.extend([p.name]*len(wins))

        if not Xs or not Xt:
            raise RuntimeError("Empty windows for source/target; check domains and filenames.")
        src = DatasetBundle(X=Xs, y=ys, domain=[source_domain]*len(ys), file_id=fids, fs=fs)
        tgt = DatasetBundle(X=Xt, y=yt, domain=[target_domain]*len(yt), file_id=fidt, fs=fs)
        return src, tgt
