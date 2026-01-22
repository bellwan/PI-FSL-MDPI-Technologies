from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pifsl.data.bosch_drilling.loader import Bundle


# -----------------------------
# filename helpers
# -----------------------------
def _parse_tool_and_cycle(stem: str) -> Tuple[str, Optional[int]]:
    parts = stem.split("_")
    tool_id = parts[0] if len(parts) >= 1 else stem

    cycle = None
    for p in parts:
        if p.startswith("C") and p[1:].isdigit():
            cycle = int(p[1:])
            break
    return tool_id, cycle


# -----------------------------
# raw csv -> compact feature vector
# -----------------------------
def _numeric_sensor_columns(df: pd.DataFrame) -> List[str]:
    drop_names = {
        "time", "timestamp", "t",
        "CycleToFailure", "CycleToFailureNormalized",
        "cycle_to_failure", "cycle_to_failure_normalized",
        "tool_wear", "ToolWear", "wear",
        "label", "y", "target",
    }

    cols: List[str] = []
    for c in df.columns:
        if c in drop_names:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _stats_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((7,), dtype=np.float32)

    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    mn = float(np.min(x))
    mx = float(np.max(x))

    eps = 1e-12
    z = (x - mean) / (std + eps)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)

    return np.array([mean, std, rms, mn, mx, skew, kurt], dtype=np.float32)


def _row_features_from_cycle_csv(csv_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    df = pd.read_csv(csv_path)

    # light column normalization
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    meta: Dict[str, float] = {}

    for key in ["CycleToFailureNormalized", "cycle_to_failure_normalized"]:
        if key in df.columns and pd.api.types.is_numeric_dtype(df[key]):
            v = df[key].dropna()
            if len(v) > 0:
                meta["CycleToFailureNormalized"] = float(v.iloc[0])
            break

    for key in ["CycleToFailure", "cycle_to_failure"]:
        if key in df.columns and pd.api.types.is_numeric_dtype(df[key]):
            v = df[key].dropna()
            if len(v) > 0:
                meta["CycleToFailure"] = float(v.iloc[0])
            break

    sensor_cols = _numeric_sensor_columns(df)
    if not sensor_cols:
        raise ValueError(
            f"[Bosch milling] No numeric sensor columns found in {csv_path}. "
            f"Columns={list(df.columns)}"
        )

    feats = []
    for c in sensor_cols:
        feats.append(_stats_features(df[c].to_numpy()))
    feat_vec = np.concatenate(feats, axis=0).astype(np.float32)

    return feat_vec, meta


def _zscore_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd, mu, sd


# -----------------------------
# processed feature-table loader
# -----------------------------
def _discover_feature_table(root: Path) -> Optional[Path]:
    if root.is_file():
        return root

    candidates: List[Path] = []
    patterns = [
        "*feature*.csv",
        "*features*.csv",
        "*processed*.csv",
        "*feature*.parquet",
        "*features*.parquet",
        "*processed*.parquet",
    ]
    for pat in patterns:
        candidates.extend(root.glob(pat))

    # prefer the smallest matching table (feature tables are typically << raw)
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    # filter out extremely large tables to avoid accidentally reading raw exports
    candidates2 = []
    for p in candidates:
        try:
            if p.stat().st_size <= 2_000_000_000:  # 2GB safety gate
                candidates2.append(p)
        except Exception:
            continue
    candidates = candidates2 or candidates

    candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 1e18)
    return candidates[0] if candidates else None


def _read_feature_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    # 1) Try normal CSV first
    try:
        df = pd.read_csv(path)
        # If it parsed into many columns, we’re good.
        if df.shape[1] >= 5:
            return df
    except Exception:
        pass

    # 2) Try semicolon + header on 2nd line (matches FeatureAndMetadata_Milling.csv)
    df = pd.read_csv(path, sep=";", header=1, engine="python")
    return df

def _load_ctf_norm_lookup(root: Path) -> Dict[str, float]:
    search_roots = [root]
    if root.parent != root:
        search_roots.append(root.parent)

    # preferred order
    candidate_files: List[Path] = []
    for r in search_roots:
        candidate_files += [
            r / "metadata.xlsx",
            r / "FeatureAndMetadata_Milling.csv",
        ]

    df: Optional[pd.DataFrame] = None
    for p in candidate_files:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                # must match the official FeatureAndMetadata_Milling.csv formatting
                df = pd.read_csv(p, sep=";", header=1, engine="python")
            df = df.rename(columns={c: str(c).strip() for c in df.columns})
            break
        except Exception:
            df = None

    if df is None or len(df) == 0:
        return {}

    exp_col = None
    for c in df.columns:
        if str(c).strip().lower() == "experimentindex":
            exp_col = c
            break

    ctf_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("cycletofailurenormalized", "cycle_to_failure_normalized"):
            ctf_col = c
            break

    if exp_col is None or ctf_col is None:
        return {}

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        k = str(row[exp_col]).strip()
        try:
            v = float(row[ctf_col])
        except Exception:
            continue
        if k and np.isfinite(v):
            out[k] = v
    return out


def _label_from_ctf_norm(ctf_norm: float, label_mode: str) -> int:
    m = (label_mode or "3class").lower()
    if m in ("binary", "2class", "fault"):
        return 0 if float(ctf_norm) >= 0.5 else 1
    # 3-class: early / mid / late
    if float(ctf_norm) >= (2.0 / 3.0):
        return 0
    if float(ctf_norm) >= (1.0 / 3.0):
        return 1
    return 2


def _split_feature_cols(df: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    # common targets
    target_candidates = [
        "CycleToFailureNormalized",
        "cycle_to_failure_normalized",
        "CycleToFailure",
        "cycle_to_failure",
        "tool_wear",
        "ToolWear",
        "wear",
        "y",
        "label",
        "target",
    ]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    # metadata columns that should not be treated as features
    meta_like = {
        # file identity
        "File", "file",
        "Filename", "filename",
        "FileName", "filename",  
        "name",
        # indexing / grouping
        "SampleIndex", "sample_index",
        "NumberOfCycle", "numberofcycle", 
        "Part", "part",
        "Layer", "layer",
        "Cycle", "cycle",
        # tool identifiers (both spellings)
        "ToolIndex", "tool_index",
        "TollIndex", "tollindex",  
        "ToolID", "tool_id",
        "Tool", "tool",
    }
    if target_col:
        meta_like.add(target_col)

    feature_cols: List[str] = []
    for c in df.columns:
        if c in meta_like:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    return feature_cols, target_col

# -----------------------------
# public API
# -----------------------------
def load_bosch_mi_source(
    data_root: str,
    normalization: str = "zscore",
    seed: int = 0,
    max_files: Optional[int] = None,
    label_mode: str = "3class",
    prefer_feature_table: bool = True,
    include_domains: Optional[List[str]] = None,
) -> Bundle:
    rng = np.random.default_rng(int(seed))
    root = Path(data_root)

    if not root.exists():
        raise FileNotFoundError(f"[Bosch milling] data_root does not exist: {data_root}")

    # ---- processed feature table path ----
    if prefer_feature_table:
        ft = _discover_feature_table(root)
        if ft is not None and ft.is_file() and ft.suffix.lower() in (".csv", ".parquet"):
            df = _read_feature_table(ft)
            df = df.rename(columns={c: str(c).strip() for c in df.columns})

            feat_cols, target_col = _split_feature_cols(df)
            if not feat_cols:
                raise ValueError(
                    f"[Bosch milling] No numeric feature columns found in feature table: {ft}. "
                    f"Columns={list(df.columns)}"
                )
            if target_col is None:
                raise ValueError(
                    f"[Bosch milling] No target column found in feature table: {ft}. "
                    f"Expected one of CycleToFailureNormalized/CycleToFailure/tool_wear/label."
                )


            if "FileName" in df.columns:
                dom_series = df["FileName"].astype(str).map(lambda s: str(s).split("_")[0])
            elif "Filename" in df.columns:
                dom_series = df["Filename"].astype(str).map(lambda s: str(s).split("_")[0])
            elif "filename" in df.columns:
                dom_series = df["filename"].astype(str).map(lambda s: str(s).split("_")[0])

            elif "TollIndex" in df.columns:
                dom_series = df["TollIndex"].astype(str).map(lambda s: f"Tool{s}")
            elif "ToolIndex" in df.columns:
                dom_series = df["ToolIndex"].astype(str).map(lambda s: f"Tool{s}")
            elif "tool_index" in df.columns:
                dom_series = df["tool_index"].astype(str).map(lambda s: f"Tool{s}")
            elif "ToolID" in df.columns:
                dom_series = df["ToolID"].astype(str)
            elif "tool_id" in df.columns:
                dom_series = df["tool_id"].astype(str)
            else:
                # fallback: single domain
                dom_series = pd.Series(["bosch_milling"] * len(df), index=df.index)


            if include_domains:
                keep = dom_series.isin(set(include_domains))
                df = df.loc[keep].reset_index(drop=True)
                dom_series = dom_series.loc[keep].reset_index(drop=True)

            # max_files (rows) sampling
            if max_files is not None and max_files > 0 and len(df) > max_files:
                idx = rng.choice(len(df), size=int(max_files), replace=False)
                idx = np.asarray(sorted(idx))
                df = df.iloc[idx].reset_index(drop=True)
                dom_series = dom_series.iloc[idx].reset_index(drop=True)

            X = df[feat_cols].to_numpy(dtype=np.float32)
            y_raw = df[target_col].to_numpy()

            # map target -> labels
            y: List[int] = []
            for v in y_raw:
                try:
                    fv = float(v)
                except Exception:
                    fv = float("nan")
                if not np.isfinite(fv):
                    # default to early class if missing
                    y.append(0)
                else:
                    # if target is not normalized, try to normalize loosely
                    if target_col.lower() in ("cycletofailure", "cycle_to_failure"):

                        y.append(0)  # placeholder; overwritten below
                    else:
                        y.append(_label_from_ctf_norm(fv, label_mode))

            if target_col.lower() in ("cycletofailure", "cycle_to_failure"):
                vals = np.asarray([float(v) if np.isfinite(float(v)) else np.nan for v in y_raw], dtype=np.float64)
                # treat higher CTF as earlier; use terciles
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    y = [0] * len(vals)
                else:
                    q1 = float(np.nanquantile(vals, 1.0 / 3.0))
                    q2 = float(np.nanquantile(vals, 2.0 / 3.0))
                    y = []
                    for v in vals:
                        if not np.isfinite(v):
                            y.append(0)
                        elif v >= q2:
                            y.append(0)
                        elif v >= q1:
                            y.append(1)
                        else:
                            y.append(2 if str(label_mode).lower() not in ("binary", "2class", "fault") else 1)

            y_arr = np.asarray(y, dtype=np.int64)
            dom_arr = np.asarray(dom_series.to_list(), dtype=object)
            file_id = np.asarray(
                df["Filename"].astype(str).to_list() if "Filename" in df.columns else
                df["filename"].astype(str).to_list() if "filename" in df.columns else
                [f"row_{i}" for i in range(len(df))],
                dtype=object,
            )

            if str(normalization).lower() == "zscore":
                X, _, _ = _zscore_fit_transform(X)

            return Bundle(X=X, y=y_arr, domain=dom_arr, file_id=file_id, fs=1.0)

    # ---- raw per-cycle CSV mode ----
    if root.is_file():
        raise ValueError(
            f"[Bosch milling] data_root looks like a file but it's not a supported feature-table type: {root}"
        )

    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"[Bosch milling] No .csv files found under {data_root}")

    if max_files is not None and max_files > 0 and len(csvs) > max_files:
        idx = rng.choice(len(csvs), size=int(max_files), replace=False)
        csvs = [csvs[i] for i in sorted(idx)]

    # compute max cycle per domain for fallback labeling
    tool_cycles: Dict[str, List[int]] = {}
    parsed: List[Tuple[Path, str, Optional[int]]] = []
    for p in csvs:
        tool_id, cyc = _parse_tool_and_cycle(p.stem)
        if include_domains and tool_id not in set(include_domains):
            continue
        parsed.append((p, tool_id, cyc))
        if cyc is not None:
            tool_cycles.setdefault(tool_id, []).append(cyc)
    if not parsed:
        raise RuntimeError("[Bosch milling] After filtering, no files remain. Check include_domains/max_files.")

    tool_max_cycle = {k: max(v) for k, v in tool_cycles.items() if v}

    # Keyed by filename stem like "P002_F01_C1"
    ctf_lookup = _load_ctf_norm_lookup(root)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    domain_list: List[str] = []
    file_id_list: List[str] = []

    for p, tool_id, cyc in parsed:
        feat_vec, meta = _row_features_from_cycle_csv(p)

        ctf_norm = None
        if p.stem in ctf_lookup:
            ctf_norm = ctf_lookup[p.stem]
        elif "CycleToFailureNormalized" in meta and np.isfinite(meta["CycleToFailureNormalized"]):
            ctf_norm = float(meta["CycleToFailureNormalized"])

        y: Optional[int] = None
        if ctf_norm is not None and np.isfinite(ctf_norm):
            y = _label_from_ctf_norm(ctf_norm, label_mode)

        if y is None:
            if cyc is None or tool_id not in tool_max_cycle or tool_max_cycle[tool_id] <= 1:
                y = 0
            else:
                maxc = tool_max_cycle[tool_id]
                frac = (cyc - 1) / float(maxc - 1)  # 0..1
                if str(label_mode).lower() in ("binary", "2class", "fault"):
                    y = 1 if frac >= 0.5 else 0
                else:
                    y = 0 if frac < (1.0 / 3.0) else 1 if frac < (2.0 / 3.0) else 2

        X_list.append(feat_vec)
        y_list.append(int(y))
        domain_list.append(tool_id)
        file_id_list.append(p.stem)


    X = np.stack(X_list, axis=0).astype(np.float32)
    y_arr = np.asarray(y_list, dtype=np.int64)
    domain_arr = np.asarray(domain_list, dtype=object)
    file_id = np.asarray(file_id_list, dtype=object)

    if str(normalization).lower() == "zscore":
        X, _, _ = _zscore_fit_transform(X)

    return Bundle(X=X, y=y_arr, domain=domain_arr, file_id=file_id, fs=1.0)


def load_scidata2025_source(
    data_root: str,
    normalization: str = "zscore",
    seed: int = 0,
    max_files: Optional[int] = None,
    label_mode: str = "3class",
    prefer_feature_table: bool = True,
    include_domains: Optional[List[str]] = None,
) -> Bundle:
    return load_bosch_mi_source(
        data_root=data_root,
        normalization=normalization,
        seed=seed,
        max_files=max_files,
        label_mode=label_mode,
        prefer_feature_table=prefer_feature_table,
        include_domains=include_domains,
    )
