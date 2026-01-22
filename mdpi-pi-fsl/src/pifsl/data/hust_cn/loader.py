from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import re

def _read_hust_cn_xls(path):
    p = Path(path)

    with p.open("rb") as f:
        head = f.read(64)

    # OLE2 .xls signature
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        # needs xlrd installed for true .xls
        return pd.read_excel(p, header=None, engine="xlrd")

    # ZIP-based .xlsx signature
    if head[:2] == b"PK":
        return pd.read_excel(p, header=None, engine="openpyxl")

    # HTML disguised as .xls
    if head.lstrip().startswith(b"<"):

        return pd.read_html(p)[0]

    # --- Text / delimited "xls" ---
    # Fast path: look for the "Data" marker in the first few hundred lines.
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            pre = [next(f) for _ in range(400)]
    except StopIteration:
        pre = []

    data_row = None
    for i, line in enumerate(pre):
        if line.strip() == "Data":
            data_row = i
            break

    if data_row is not None:
        # The numeric part is a clean tab-delimited table.
        return pd.read_csv(
            p,
            header=None,
            sep="\t",
            skiprows=data_row + 1,
            engine="c",
        )

    # Fallback: try common delimiters, but prefer tab first (comma will
    # "succeed" even when the file is actually tab-delimited).
    for sep in ["\t", ",", ";", "|"]:
        try:
            return pd.read_csv(p, header=None, sep=sep, engine="python")
        except Exception:
            pass

    # Final fallback
    return pd.read_excel(p, header=None)


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


_DOM_RE = re.compile(r"(\d+Hz|0[-_]?40[-_]?0Hz)$", re.IGNORECASE)


def _parse_domain(stem: str) -> str:
    s = stem.replace(" ", "")
    m = _DOM_RE.search(s)
    if m:
        return m.group(1)
    toks = re.split(r"[_\-]+", s)
    return toks[-1]


def _parse_class_binary(stem: str) -> int:
    s = stem.upper()
    # if contains "_H_" or startswith "H"
    if re.search(r"(^|[_\-])H($|[_\-])", s):
        return 0
    if s.startswith("H"):
        return 0
    return 1



def _parse_class_key_multiclass(stem: str) -> str:
    s = stem.strip().replace(" ", "_")
    m = _DOM_RE.search(s)
    if m and m.group(1):
        s = re.sub(r"[_\-]*" + re.escape(m.group(1)) + r"$", "", s, flags=re.IGNORECASE)

    toks = [t for t in re.split(r"[_\-]+", s) if t]
    toks = [t for t in toks if not re.fullmatch(r"\d+", t)]
    if not toks:
        return "UNKNOWN"
    return "_".join([t.upper() for t in toks])

def load_hust_cn_windows(
    data_root: str,
    source_domain: str,
    target_domain: str,
    window_seconds: float = 0.08,
    overlap_ratio: float = 0.5,
    normalization: str = "per_window",
    fs: float = 25600.0,
    label_mode: str = "binary", 
    seed: int = 0
) -> Tuple[Bundle, Bundle]:
    root = Path(data_root)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".xlsx", ".xls")])
    if not files:
        raise RuntimeError(f"No Excel files found under {root}")

    win = int(round(float(window_seconds) * float(fs)))
    win = max(win, 1024)
    stride = max(1, int(round(win * (1.0 - float(overlap_ratio)))))

    def read_sig(p: Path) -> np.ndarray:
        df = _read_hust_cn_xls(p)
        # df should be a DataFrame; if any loader path returns list, normalize it
        if isinstance(df, list):
            df = df[0]

        arr = df.to_numpy()

        # collect numeric columns
        cols: List[np.ndarray] = []
        for c in range(arr.shape[1]):
            col = np.asarray(pd.to_numeric(arr[:, c], errors="coerce"), dtype=np.float64)
            col = col[np.isfinite(col)]
            if col.size >= 200:
                cols.append(col)

        if not cols:
            raise RuntimeError(f"No usable numeric columns in {p}")

        # HUST format: [time, speed, X, Y, Z, ...]
        # Prefer X/Y/Z explicitly if present
        if len(cols) >= 5:
            candidates = cols[2:5]   # X/Y/Z
        elif len(cols) >= 3:
            candidates = cols[-3:]   # best guess: last three are vibration
        else:
            candidates = cols

        # filter out monotonic/time-like columns (high corr with index)
        cleaned: List[np.ndarray] = []
        for v in candidates:
            if v.size < 200:
                continue
            t = np.arange(v.size, dtype=np.float64)
            if np.std(v) < 1e-12:
                continue
            corr = np.corrcoef(t, v)[0, 1]
            if np.isfinite(corr) and abs(corr) > 0.995:
                continue
            cleaned.append(v)

        if cleaned:
            candidates = cleaned

        # pick the most energetic vibration axis
        best = max(candidates, key=lambda v: float(np.nanstd(v)))
        best = best[np.isfinite(best)]
        return best.reshape(-1)


    def window_1d(x: np.ndarray) -> List[np.ndarray]:
        if x.size < win:
            return []
        return [x[i:i+win].copy() for i in range(0, x.size - win + 1, stride)]

    Xs: List[np.ndarray] = []; ds: List[str] = []; fsid: List[str] = []
    Xt: List[np.ndarray] = []; dt: List[str] = []; ftid: List[str] = []

    ys_any: List[Any] = []
    yt_any: List[Any] = []

    for p in files:
        dom = _parse_domain(p.stem)
        if dom not in (source_domain, target_domain):
            continue

        sig = read_sig(p)
        wins = window_1d(sig)
        if not wins:
            continue

        if label_mode == "binary":
            y = _parse_class_binary(p.stem)
        elif label_mode in ("multiclass", "full", "9class"):
            y = _parse_class_key_multiclass(p.stem)
        else:
            raise ValueError(f"Unsupported label_mode for HUST-CN: {label_mode}")

        if normalization == "per_window":
            wins = [_zscore(w) for w in wins]
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        if dom == source_domain:
            Xs.extend([w.astype(np.float32) for w in wins])
            ys_any.extend([y] * len(wins))
            ds.extend([dom] * len(wins))
            fsid.extend([p.name] * len(wins))
        else:
            Xt.extend([w.astype(np.float32) for w in wins])
            yt_any.extend([y] * len(wins))
            dt.extend([dom] * len(wins))
            ftid.extend([p.name] * len(wins))

    if len(ys_any) == 0 or len(yt_any) == 0:
        raise RuntimeError(f"HUST-CN empty windows for src/target. Check domain strings and data_root.")

    # Finalize labels as ints
    if label_mode in ("multiclass", "full", "9class"):
        uniq = sorted(set(list(ys_any) + list(yt_any)))
        class_key_to_id = {k: ii for ii, k in enumerate(uniq)}
        ys = [int(class_key_to_id[k]) for k in ys_any]
        yt = [int(class_key_to_id[k]) for k in yt_any]
    else:
        ys = [int(v) for v in ys_any]
        yt = [int(v) for v in yt_any]

    return Bundle(X=Xs, y=ys, domain=ds, file_id=fsid, fs=float(fs)), Bundle(X=Xt, y=yt, domain=dt, file_id=ftid, fs=float(fs))


def summarize_hust_cn(data_root: str) -> Dict[str, Any]:
    root = Path(data_root)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".xlsx", ".xls")])
    doms = {}
    for p in files:
        d = _parse_domain(p.stem)
        doms[d] = doms.get(d, 0) + 1
    return {"n_excel_files": len(files), "domains_filecount": doms, "fs_typical": 25600.0}
