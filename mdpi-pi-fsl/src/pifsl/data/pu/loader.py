from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

from pifsl.data.bosch_drilling.loader import Bundle

def _parse_code_list(s: str):
    # Accept "K001" or "K001,K002,K003"
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return {_norm_code(x) for x in s.split(",") if str(x).strip()}


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-8)
    return ((x - mu) / sd).astype(np.float32)


def _norm_code(code: str) -> str:
    s = str(code or "").strip().upper()
    # keep alnum only (drop dashes/spaces)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _norm_setting(setting: str) -> str:
    s = str(setting or "").strip().upper()
    s = s.replace("-", "_").replace(" ", "")
    # canonicalize Nxx_Mxx_Fxx if possible
    m = re.search(r"N(\d+).*?M(\d+).*?F(\d+)", s)
    if m:
        n, mo, f = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"N{n:02d}_M{mo:02d}_F{f:02d}"
    # otherwise collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s


def _parse_bearing_code(stem: str) -> Optional[str]:
    s = str(stem or "").upper()
    m = re.search(r"(K[A-Z]{0,2}\d{2,3})", s)
    if not m:
        return None
    return _norm_code(m.group(1))


def _parse_setting(stem: str) -> Optional[str]:
    s = str(stem or "")
    m = re.search(r"N(\d+)[_\-]?M(\d+)[_\-]?F(\d+)", s, flags=re.IGNORECASE)
    if m:
        n, mo, f = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"N{n:02d}_M{mo:02d}_F{f:02d}"

    # fallback: old underscore split
    parts = s.split("_")
    if len(parts) >= 3 and parts[0].upper().startswith("N") and parts[1].upper().startswith("M") and parts[2].upper().startswith("F"):
        try:
            n = int(re.sub(r"\D", "", parts[0]))
            mo = int(re.sub(r"\D", "", parts[1]))
            f = int(re.sub(r"\D", "", parts[2]))
            return f"N{n:02d}_M{mo:02d}_F{f:02d}"
        except Exception:
            return _norm_setting(f"{parts[0]}_{parts[1]}_{parts[2]}")
    return None


def _default_mapping() -> Dict[int, List[str]]:
    return {
        0: ["K001", "K002"],      # healthy
        1: ["KI14", "KA04", "KA01"],  # localized faults (inner/outer)
        2: ["KB23"],              # distributed/combined faults
    }


def _label_from_code(code: str, mapping: Dict[int, List[str]]) -> Optional[int]:
    c = str(code).upper()

    # explicit mapping wins
    for lab, codes in (mapping or {}).items():
        if c in {str(x).upper() for x in (codes or [])}:
            return int(lab)

    # broad defaults
    if re.match(r"^K0\d{2}$", c):
        return 0
    if c.startswith(("KA", "KI")):
        return 1
    if c.startswith("KB"):
        return 2
    return None


def _iter_named_numeric_vectors(obj: Any, prefix: str = "", max_depth: int = 7) -> Iterable[Tuple[str, np.ndarray]]:
    if max_depth <= 0 or obj is None:
        return

    if hasattr(obj, "_fieldnames") and isinstance(getattr(obj, "_fieldnames"), (list, tuple)):
        for fname in obj._fieldnames:
            try:
                v = getattr(obj, fname)
            except Exception:
                continue
            name = f"{prefix}.{fname}" if prefix else str(fname)
            yield from _iter_named_numeric_vectors(v, prefix=name, max_depth=max_depth - 1)
        return

    # list/tuple container
    if isinstance(obj, (list, tuple)):
        for i, el in enumerate(obj):
            name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _iter_named_numeric_vectors(el, prefix=name, max_depth=max_depth - 1)
        return

    # dict-like
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_named_numeric_vectors(v, prefix=name, max_depth=max_depth - 1)
        return

    # numpy array
    if isinstance(obj, np.ndarray):
        # MATLAB struct represented as ndarray with fields
        if obj.dtype.names is not None:
            for name in obj.dtype.names:
                try:
                    yield from _iter_named_numeric_vectors(
                        obj[name],
                        prefix=f"{prefix}.{name}" if prefix else name,
                        max_depth=max_depth - 1,
                    )
                except Exception:
                    continue
            return

        # cell array / object array
        if obj.dtype == object:
            for idx, el in enumerate(obj.flat):
                yield from _iter_named_numeric_vectors(
                    el,
                    prefix=f"{prefix}[{idx}]" if prefix else f"[{idx}]",
                    max_depth=max_depth - 1,
                )
            return

        # numeric array
        if np.issubdtype(obj.dtype, np.number):
            a = np.asarray(obj)
            if a.size == 0:
                return

            # vector-ish
            if a.ndim == 1:
                yield prefix, a.astype(np.float32, copy=False)
                return
            if a.ndim == 2 and 1 in a.shape:
                yield prefix, a.reshape(-1).astype(np.float32, copy=False)
                return

            # 2D matrix: treat the longer dimension as time, yield per-channel vectors
            if a.ndim == 2:
                n0, n1 = int(a.shape[0]), int(a.shape[1])
                if n0 >= n1:
                    for ch in range(n1):
                        v = a[:, ch].reshape(-1).astype(np.float32, copy=False)
                        yield (f"{prefix}[:,{ch}]" if prefix else f"[:,{ch}]"), v
                else:
                    for ch in range(n0):
                        v = a[ch, :].reshape(-1).astype(np.float32, copy=False)
                        yield (f"{prefix}[{ch},:]" if prefix else f"[{ch},:]"), v
                return

            # higher-d arrays: flatten (last resort)
            yield prefix, a.reshape(-1).astype(np.float32, copy=False)
            return

    # numpy scalar with fields
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        for name in obj.dtype.names:
            try:
                yield from _iter_named_numeric_vectors(
                    obj[name],
                    prefix=f"{prefix}.{name}" if prefix else name,
                    max_depth=max_depth - 1,
                )
            except Exception:
                continue
        return

    # fallback: try array conversion
    try:
        a = np.asarray(obj)
        if np.issubdtype(a.dtype, np.number) and a.size > 0:
            if a.ndim == 1:
                yield prefix, a.astype(np.float32, copy=False)
            elif a.ndim == 2 and 1 in a.shape:
                yield prefix, a.reshape(-1).astype(np.float32, copy=False)
            elif a.ndim == 2:
                n0, n1 = int(a.shape[0]), int(a.shape[1])
                if n0 >= n1:
                    for ch in range(n1):
                        yield (f"{prefix}[:,{ch}]" if prefix else f"[:,{ch}]"), a[:, ch].reshape(-1).astype(np.float32, copy=False)
                else:
                    for ch in range(n0):
                        yield (f"{prefix}[{ch},:]" if prefix else f"[{ch},:]"), a[ch, :].reshape(-1).astype(np.float32, copy=False)
    except Exception:
        return


def _find_signal(arrs: Dict[str, Any], keywords: List[str], min_len: int = 128) -> Optional[np.ndarray]:
    keywords_l = [str(kw).lower() for kw in (keywords or [])]

    # infer desired kind from keywords
    want_vibration = any(kw in ("vib", "vibration", "acc", "acceleration", "sensor", "a_") for kw in keywords_l)
    want_current = any(kw in ("current", "motor_current", "motor", "cur", "i_", "current1", "current2") for kw in keywords_l)

    def _safe_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _get_field(obj: Any, field: str) -> Any:
        # MATLAB struct in scipy is often mat_struct (duck-typed with _fieldnames)
        if hasattr(obj, field):
            return getattr(obj, field)
        # numpy void / structured
        try:
            if isinstance(obj, np.void) and obj.dtype.names and field in obj.dtype.names:
                return obj[field]
        except Exception:
            pass
        # dict-like
        try:
            if isinstance(obj, dict) and field in obj:
                return obj[field]
        except Exception:
            pass
        return None

    def _as_1d_numeric(x: Any) -> Optional[np.ndarray]:
        try:
            a = np.asarray(x)
        except Exception:
            return None
        if a.size < int(min_len):
            return None
        if not np.issubdtype(a.dtype, np.number):
            return None
        if a.ndim == 1:
            return a.astype(np.float32, copy=False)
        if a.ndim == 2 and 1 in a.shape:
            return a.reshape(-1).astype(np.float32, copy=False)
        # if matrix, flatten (last resort)
        return a.reshape(-1).astype(np.float32, copy=False)

    def _hf_ratio(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = int(min(x.size, 65536))
        if n < 1024:
            return 0.0
        xx = x[:n] - float(np.mean(x[:n]))
        # real FFT
        spec = np.fft.rfft(xx)
        p = (spec.real * spec.real + spec.imag * spec.imag)
        if p.size < 8:
            return 0.0
        # split at ~20% of Nyquist
        cut = int(max(1, round(0.20 * p.size)))
        lo = float(np.sum(p[:cut]) + 1e-12)
        hi = float(np.sum(p[cut:]) + 1e-12)
        return hi / lo

    def _impulsiveness(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = int(min(x.size, 65536))
        if n < 1024:
            return 0.0
        xx = x[:n] - float(np.mean(x[:n]))
        v2 = float(np.mean(xx * xx) + 1e-12)
        v4 = float(np.mean((xx * xx) * (xx * xx)) + 1e-12)
        return v4 / (v2 * v2)


    Y = None
    for k in ("Y", "y"):
        if k in arrs:
            Y = arrs[k]
            break

    y_channels: List[Tuple[str, np.ndarray]] = []
    if Y is not None:
        # normalize to iterable over channel entries
        try:
            if isinstance(Y, np.ndarray) and Y.dtype == object:
                items = list(Y.flat)
            elif isinstance(Y, np.ndarray) and Y.dtype.names is not None:
                items = [Y]  # structured array case
            elif isinstance(Y, (list, tuple)):
                items = list(Y)
            else:
                # single struct
                items = [Y]
        except Exception:
            items = [Y]

        for i, ch in enumerate(items):
            name = _get_field(ch, "Name")
            data = _get_field(ch, "Data")
            data_v = _as_1d_numeric(data)
            if data_v is None:
                continue
            nm = _safe_str(name).strip()
            if not nm or nm.lower() == "none":
                nm = f"Y[{i}]"
            y_channels.append((nm, data_v))

        # If Y-path produced candidates, select from them
        if y_channels:
            # 2) keyword match on channel Name
            def hit_score(nm: str) -> int:
                nml = nm.lower()
                return 1 if any(kw in nml for kw in keywords_l) else 0

            hits = [(nm, x) for (nm, x) in y_channels if hit_score(nm) > 0]
            if hits:
                # among hits, prefer longest
                hits.sort(key=lambda t: int(t[1].size), reverse=True)
                return hits[0][1]

            # 3) heuristic if no name hit
            # vibration: high HF ratio + impulsiveness
            # current: low HF ratio + smoother (lower impulsiveness)
            if want_vibration and not want_current:
                scored = []
                for nm, x in y_channels:
                    scored.append((_hf_ratio(x) + 0.10 * _impulsiveness(x), nm, x))
                scored.sort(key=lambda t: t[0], reverse=True)
                return scored[0][2]
            if want_current and not want_vibration:
                scored = []
                for nm, x in y_channels:
                    scored.append((-( _hf_ratio(x) + 0.10 * _impulsiveness(x) ), nm, x))
                scored.sort(key=lambda t: t[0], reverse=True)
                return scored[0][2]

            # unknown: pick by length (but at least it’s within Y channels, not arbitrary)
            y_channels.sort(key=lambda t: int(t[1].size), reverse=True)
            return y_channels[0][1]


    vecs: List[Tuple[str, np.ndarray]] = []
    for k, v in arrs.items():
        vecs.extend(list(_iter_named_numeric_vectors(v, prefix=str(k), max_depth=7)))

    vecs = [(n, x) for (n, x) in vecs if x is not None and x.size >= int(min_len)]
    if not vecs:
        return None

    def score(name: str, x: np.ndarray) -> Tuple[int, int]:
        n = str(name).lower()
        hit = 1 if any(kw in n for kw in keywords_l) else 0
        return (hit, int(x.size))

    vecs.sort(key=lambda t: score(t[0], t[1]), reverse=True)
    return vecs[0][1]


def load_pu_source(
    data_root: str,
    normalization: str = "per_window",
    fs_target: float = 64000.0,
    src_fs: float = 64000.0,
    window: int = 1000,
    hop: int = 500,
    mapping: Optional[Dict[int, List[str]]] = None,
    max_files: Optional[int] = None,
    current_key: str = "motor_current",
    include_codes: Optional[List[str]] = None,
    include_settings: Optional[List[str]] = None,
    max_windows_per_file: Optional[int] = None,
    healthy_code: Optional[str] = None,
    pos_code: Optional[str] = None,
) -> Bundle:
    root = Path(data_root)
    mats = sorted(root.rglob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files found under {root}")

    mapping = mapping or _default_mapping()

    healthy_set = _parse_code_list(healthy_code) if healthy_code is not None else None
    pos_set = _parse_code_list(pos_code) if pos_code is not None else None

    use_binary = (healthy_set is not None) and (pos_set is not None)

    codes_keep = {_norm_code(c) for c in (include_codes or [])} if include_codes else None
    settings_keep = {_norm_setting(s) for s in (include_settings or [])} if include_settings else None


    stats: Dict[str, int] = {
        "total_mat_files": int(len(mats)),
        "scanned": 0,
        "skip_no_code": 0,
        "skip_code_filter": 0,
        "skip_setting_filter": 0,
        "skip_no_label": 0,
        "skip_loadmat": 0,
        "skip_no_vib": 0,
        "skip_too_short": 0,
        "windows": 0,
    }
    found_codes: Dict[str, int] = {}
    found_settings: Dict[str, int] = {}

    X: List[Dict[str, np.ndarray]] = []
    y: List[int] = []
    dom = "Paderborn_Bearing"
    file_id: List[str] = []

    file_count = 0
    for fp in mats:
        if max_files is not None and file_count >= int(max_files):
            break
        stats["scanned"] += 1

        code = _parse_bearing_code(fp.stem)
        if code is None:
            parent_code = _norm_code(fp.parent.name)
            if parent_code.startswith("K") and len(parent_code) >= 4:
                code = parent_code
        if code is None:
            stats["skip_no_code"] += 1
            continue

        found_codes[code] = found_codes.get(code, 0) + 1
        if codes_keep is not None and _norm_code(code) not in codes_keep:
            stats["skip_code_filter"] += 1
            continue

        setting_raw = _parse_setting(fp.stem)
        setting = _norm_setting(setting_raw) if setting_raw else "UNKNOWN"

        if setting == "UNKNOWN" and settings_keep:
            stem_norm = _norm_setting(fp.stem)
            hits = [s for s in settings_keep if s in stem_norm]
            if len(hits) == 1:
                setting = hits[0]

        if setting != "UNKNOWN":
            found_settings[setting] = found_settings.get(setting, 0) + 1

        if settings_keep is not None and setting not in settings_keep:
            stats["skip_setting_filter"] += 1
            continue

        if use_binary:
            c = _norm_code(code)
            if c in healthy_set:
                lab = 0
            elif c in pos_set:
                lab = 1
            else:
                stats["skip_no_label"] += 1
                continue
        else:
            lab = _label_from_code(code, mapping)
            if lab is None:
                stats["skip_no_label"] += 1
                continue

        try:
            mat = loadmat(fp, squeeze_me=True, struct_as_record=False)
        except Exception:
            stats["skip_loadmat"] += 1
            continue

        arrs = {k: v for k, v in mat.items() if not str(k).startswith("__")}

        vib = _find_signal(arrs, ["vib", "acc", "a_", "acceleration", "vibration", "sensor"])
        cur = _find_signal(arrs, [str(current_key), "motor_current", "motor", "cur", "current", "i_", "current1", "current2"])

        if vib is None:
            stats["skip_no_vib"] += 1
            continue

        # resample vibration
        vib_r = vib
        if src_fs != fs_target and vib.size > 0:
            n = int(round(vib.size * (fs_target / src_fs)))
            n = max(n, window + 1)
            vib_r = resample(vib, n).astype(np.float32)

        # resample current if present
        cur_r = None
        if cur is not None and cur.size > 0:
            if src_fs != fs_target:
                n = int(round(cur.size * (fs_target / src_fs)))
                n = max(n, window + 1)
                cur_r = resample(cur, n).astype(np.float32)
            else:
                cur_r = cur.astype(np.float32, copy=False)

        L = int(vib_r.size)
        if L < window:
            stats["skip_too_short"] += 1
            continue

        # sliding windows
        # build all possible window start indices first
        starts = list(range(0, max(0, L - window + 1), hop))

        if max_windows_per_file is not None and len(starts) > int(max_windows_per_file):
            m = int(max_windows_per_file)
            idx = np.linspace(0, len(starts) - 1, m, dtype=int)
            starts = [starts[i] for i in idx]

        win_count = 0
        for st in starts:
            en = st + window
            if en > L:
                continue

            item: Dict[str, np.ndarray] = {"vibration": vib_r[st:en].astype(np.float32, copy=False)}
            if cur_r is not None and int(cur_r.size) >= en:
                item["motor_current"] = cur_r[st:en].astype(np.float32, copy=False)

            if normalization == "per_window":
                item["vibration"] = _zscore(item["vibration"])
                if "motor_current" in item:
                    item["motor_current"] = _zscore(item["motor_current"])
            elif normalization == "none":
                pass
            else:
                raise ValueError(f"Unsupported normalization: {normalization}")

            X.append(item)
            y.append(int(lab))
            file_id.append(str(fp.name))
            stats["windows"] += 1
            win_count += 1


        file_count += 1

    if len(y) == 0:
        def _top(d: Dict[str, int], k: int = 12) -> List[str]:
            return [f"{kk}({vv})" for kk, vv in sorted(d.items(), key=lambda t: (-t[1], t[0]))[:k]]

        raise RuntimeError(
            f"data_root={data_root}\n"
            f"include_codes(raw)={include_codes}\n"
            f"include_settings(raw)={include_settings}\n"
            f"include_codes(normalized)={sorted(codes_keep) if codes_keep else None}\n"
            f"include_settings(normalized)={sorted(settings_keep) if settings_keep else None}\n\n"
            f"stats={stats}\n"
            f"found_codes(top)={_top(found_codes)}\n"
            f"found_settings(top)={_top(found_settings)}\n"
        )

    return Bundle(X=X, y=y, domain=[dom] * len(y), file_id=file_id, fs=float(fs_target))


def load_pu_windows(
    data_root: str,
    source_domain: str,
    target_domain: str,
    normalization: str = "per_window",
    **kwargs,
) -> Tuple[Bundle, Bundle]:
    src = load_pu_source(
        data_root=data_root,
        normalization=normalization,
        include_settings=[source_domain],
        **kwargs,
    )

    if str(target_domain).upper() == "NONE":
        empty = Bundle(X=[], y=[], domain=[], file_id=[], fs=src.fs)
        return src, empty

    tgt = load_pu_source(
        data_root=data_root,
        normalization=normalization,
        include_settings=[target_domain],
        **kwargs,
    )
    return src, tgt


def summarize_pu(data_root: str) -> Dict[str, Any]:
    return {"dataset": "Paderborn_Bearing", "data_root": str(data_root)}
