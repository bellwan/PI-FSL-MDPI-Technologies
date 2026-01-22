from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import re
import numpy as np
from scipy.io import loadmat
from pifsl.core.base import DatasetBundle

_STEM_RE = re.compile(r"^([A-Z]{1,3})(\d+)$")

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(x.mean())
    sd = float(x.std() + 1e-8)
    return ((x - mu) / sd).astype(np.float32)

def _parse(stem: str) -> Tuple[str, int, int]:
    s = stem.strip().upper().replace(" ", "")
    m = _STEM_RE.match(s)
    if not m:
        raise ValueError(f"Bad HUST-VN filename stem: {stem} (expect e.g., B500 or I402)")
    fault = m.group(1)
    digits = m.group(2)

    # Try legacy bearing+load-code interpretation first
    bearing_id = 0
    load_w = None

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
    fc = fault.upper()
    # run_one exposes label_mode {binary,multiclass,simple,full}
    # For HUST-VN, treat "multiclass" as "full"
    if label_mode == "multiclass":
        label_mode = "full"

    if label_mode == "binary":
        return 0 if fc == "H" else 1
    if label_mode == "simple":  # H/I/O/B
        if fc == "H": return 0
        if fc.startswith("I"): return 1
        if fc.startswith("O"): return 2
        if fc.startswith("B"): return 3
        return 3
    mapping = {"H":0,"I":1,"O":2,"B":3,"IO":4,"IB":5,"OB":6}
    return mapping.get(fc, 3)


def _window(x: np.ndarray, win: int, stride: int) -> List[np.ndarray]:
    if x.size < win: return []
    return [x[i:i+win].copy() for i in range(0, x.size-win+1, stride)]

class HUSTVNAdapter:
    name = "hust_vn"

    def load_windows(
        self,
        data_root: str,
        source_domain: str,
        target_domain: str,
        window_seconds: float = 0.08,
        overlap_ratio: float = 0.5,
        normalization: str = "per_window",
        domain_axis: str = "load",      # load or bearing
        label_mode: str = "binary",     # binary / simple / full (multiclass treated as full)
        fs_override: float = 51200.0,

    ) -> Tuple[DatasetBundle, DatasetBundle]:
        root = Path(data_root)
        mats = sorted(root.rglob("*.mat"))
        if not mats:
            raise RuntimeError(f"No .mat files under {root}")
        fs = float(fs_override)
        win = max(256, int(round(window_seconds * fs)))
        stride = max(1, int(round(win * (1.0 - overlap_ratio))))

        Xs: List[np.ndarray] = []; ys: List[int] = []; fids: List[str] = []
        Xt: List[np.ndarray] = []; yt: List[int] = []; fidt: List[str] = []
        fs_final = fs
        seen_domains = {}
        matched_src = 0
        matched_tgt = 0


        for p in mats:
            try:
                fault, bearing_id, load_w = _parse(p.stem)
            except Exception as e:
                print(f"[HUST-VN][SKIP] {p.name}: {e}")
                continue

            # Normalize everything to string to avoid "400" vs 400 mismatches
            src_dom = str(source_domain)
            tgt_dom = str(target_domain)

            dom = str(load_w) if domain_axis == "load" else str(bearing_id)
            seen_domains[dom] = seen_domains.get(dom, 0) + 1

            if dom not in (src_dom, tgt_dom):
                continue

            md = loadmat(p)
            if "data" not in md:
                continue
            sig = np.asarray(md["data"]).reshape(-1).astype(np.float64)
            if "fs" in md:
                try:
                    fs_val = float(np.asarray(md["fs"]).reshape(-1)[0])
                    if fs_val > 1: fs_final = fs_val
                except Exception:
                    pass
            wins = _window(sig, win, stride)
            if normalization == "per_window":
                wins = [_zscore(w) for w in wins]
            y = _map_class(fault, label_mode)

            if dom == src_dom:
                matched_src += 1
                Xs.extend([w.astype(np.float32) for w in wins]); ys.extend([y]*len(wins)); fids.extend([p.name]*len(wins))
            else:
                matched_tgt += 1
                Xt.extend([w.astype(np.float32) for w in wins]); yt.extend([y]*len(wins)); fidt.extend([p.name]*len(wins))


        if not Xs or not Xt:
            top = sorted(seen_domains.items(), key=lambda kv: kv[1], reverse=True)[:20]
            print("[HUST-VN] domain_axis =", domain_axis)
            print("[HUST-VN] requested src =", src_dom, "tgt =", tgt_dom)
            print("[HUST-VN] seen domains top20 =", top)
            print("[HUST-VN] matched files src =", matched_src, "tgt =", matched_tgt)
            raise RuntimeError("Empty windows for source/target; check domains and parsing.")

        src = DatasetBundle(X=Xs, y=ys, domain=[source_domain]*len(ys), file_id=fids, fs=float(fs_final))
        tgt = DatasetBundle(X=Xt, y=yt, domain=[target_domain]*len(yt), file_id=fidt, fs=float(fs_final))
        return src, tgt
