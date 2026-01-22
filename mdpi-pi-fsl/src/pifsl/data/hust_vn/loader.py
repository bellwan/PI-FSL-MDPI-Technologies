from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import re
from scipy.io import loadmat


@dataclass
class Bundle:
    X: List[np.ndarray]
    y: List[int]
    domain: List[str]
    file_id: List[str]
    fs: float


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


_STEM_RE = re.compile(r"^([A-Z]{1,3})(\d+)$")


def _parse_name(stem: str) -> Tuple[str, int, int]:
    s = stem.strip().upper().replace(" ", "")
    m = _STEM_RE.match(s)
    if not m:
        raise ValueError(f"Bad HUST-VN filename stem: {stem} (expect e.g., B500 or I402)")
    fault = m.group(1)
    digits = m.group(2)

    bearing_id = 0
    load_w: int

    # Case: 2 digits like "42" -> bearing=4, load_code=2
    if len(digits) == 2 and digits[1] in "0123":
        bearing_digit = int(digits[0])
        load_digit = int(digits[1])
        bearing_id = 6200 + bearing_digit
        load_w = {0: 0, 1: 0, 2: 200, 3: 400}.get(load_digit, load_digit)

    # Case: 3 digits like "402" with middle '0' -> bearing=4, load_code=2
    elif len(digits) == 3 and digits[1] == "0" and digits[2] in "0123":
        bearing_digit = int(digits[0])
        load_digit = int(digits[2])
        bearing_id = 6200 + bearing_digit
        load_w = {0: 0, 1: 0, 2: 200, 3: 400}.get(load_digit, load_digit)

    # Otherwise interpret digits directly as load value
    else:
        load_w = int(digits)

    return fault, int(bearing_id), int(load_w)


def _map_class(fault: str, label_mode: str) -> int:
    fc = str(fault).upper()

    # Treat N as healthy (same as H)
    if fc == "N":
        fc = "H"

    if label_mode == "multiclass":
        label_mode = "full"

    if label_mode == "binary":
        return 0 if fc == "H" else 1

    if label_mode == "simple":
        if fc == "H":
            return 0
        if fc.startswith("I"):
            return 1
        if fc.startswith("O"):
            return 2
        if fc.startswith("B"):
            return 3
        return 3

    # full (default)
    mapping = {"H": 0, "I": 1, "O": 2, "B": 3, "IO": 4, "IB": 5, "OB": 6}
    return mapping.get(fc, 3)


def load_hust_vn_windows(
    data_root: str,
    source_domain: str,
    target_domain: str,
    domain_axis: str = "load",   # load or bearing
    window_seconds: float = 0.08,
    overlap_ratio: float = 0.5,
    normalization: str = "per_window",
    fs_default: float = 51200.0,
    seed: int | None = None,
    label_mode: str = "binary",
) -> Tuple[Bundle, Bundle]:
    root = Path(data_root)
    mats = sorted(root.rglob("*.mat"))
    if not mats:
        raise RuntimeError(f"No .mat files found under {root}")

    # determine fs from first file if available
    fs = float(fs_default)
    try:
        md0 = loadmat(mats[0])
        if "fs" in md0:
            fs = float(np.asarray(md0["fs"]).reshape(-1)[0])
    except Exception:
        pass

    win = int(round(window_seconds * fs))
    win = max(win, 1024)
    stride = max(1, int(round(win * (1.0 - float(overlap_ratio)))))

    def window_1d(x: np.ndarray) -> List[np.ndarray]:
        if x.size < win:
            return []
        return [x[i:i+win].copy() for i in range(0, x.size - win + 1, stride)]

    Xs: List[np.ndarray] = []; ys: List[int] = []; ds: List[str] = []; fsid: List[str] = []
    Xt: List[np.ndarray] = []; yt: List[int] = []; dt: List[str] = []; ftid: List[str] = []

    # Normalize domains to string to avoid "200" vs 200 mismatches.
    src_dom = str(source_domain)
    tgt_dom = str(target_domain)

    for p in mats:
        fault, bearing_id, load_w = _parse_name(p.stem)
        dom = str(load_w) if domain_axis == "load" else str(bearing_id)
        if dom not in (src_dom, tgt_dom):
            continue
        md = loadmat(p)
        if "data" not in md:
            continue
        sig = np.asarray(md["data"]).reshape(-1).astype(np.float64)

        wins = window_1d(sig)
        if not wins:
            continue

        y = _map_class(fault, label_mode)

        if normalization == "per_window":
            wins = [_zscore(w) for w in wins]
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        if dom == src_dom:
            Xs.extend([w.astype(np.float32) for w in wins])
            ys.extend([y]*len(wins))
            ds.extend([dom]*len(wins))
            fsid.extend([p.name]*len(wins))
        else:
            Xt.extend([w.astype(np.float32) for w in wins])
            yt.extend([y]*len(wins))
            dt.extend([dom]*len(wins))
            ftid.extend([p.name]*len(wins))

    if len(ys) == 0 or len(yt) == 0:
        raise RuntimeError("HUST-VN empty windows for src/target. Check domain_axis and domains.")
    return Bundle(X=Xs, y=ys, domain=ds, file_id=fsid, fs=fs), Bundle(X=Xt, y=yt, domain=dt, file_id=ftid, fs=fs)


def summarize_hust_vn(data_root: str) -> Dict[str, Any]:
    root = Path(data_root)
    mats = sorted(root.rglob("*.mat"))
    loads = {}
    bearings = {}
    faults = {}
    for p in mats:
        try:
            f, b, l = _parse_name(p.stem)
        except Exception:
            continue
        loads[str(l)] = loads.get(str(l), 0) + 1
        bearings[str(b)] = bearings.get(str(b), 0) + 1
        faults[f] = faults.get(f, 0) + 1
    return {"n_mat_files": len(mats), "loads": loads, "bearings": bearings, "faults": faults, "fs_typical": 51200.0}
