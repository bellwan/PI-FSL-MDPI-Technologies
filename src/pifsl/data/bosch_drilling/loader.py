from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from pifsl.data.bosch_drilling import config as cfg
from pifsl.data.common.inventory import build_inventory
from pifsl.data.bosch_drilling.raw_processing.gating import build_windows_for_file

# benchmark bundle
@dataclass
class Bundle:
    X: List[np.ndarray]
    y: List[int]
    domain: List[str]
    file_id: List[str]
    fs: float


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-8)
    return ((x - mu) / sd).astype(np.float32)


def load_bosch_windows(
    data_root: str,
    source_domain: str,
    target_domain: str,
    normalization: str = "per_window",
) -> Tuple[Bundle, Bundle]:
    keep_ops = getattr(cfg, "OPS_IN_SCOPE", ["OP05", "OP07"])
    inv = build_inventory(dataset_dir=str(data_root), keep_ops=keep_ops)

    def _filter(domain: str):
        if "_OP" not in domain.upper():
            raise ValueError("Bosch domain must look like 'M01_OP05'")
        mac, op = domain.upper().split("_", 1)
        op = op.upper()
        mac = mac.upper()
        sub = inv[(inv["Machine"].str.upper() == mac) & (inv["ProcessName"].str.upper() == op)].copy()
        return sub

    inv_s = _filter(source_domain)
    inv_t = _filter(target_domain)

    # --- E0 same-domain split to avoid leakage ---
    if source_domain.upper() == target_domain.upper():
        date_src = {"Aug_2019", "Feb_2020", "Feb_2021"} 
        date_tgt = {"Feb_2019", "Aug_2021"} 

        inv_s = inv_s[inv_s["Date"].isin(date_src)].copy()
        inv_t = inv_t[inv_t["Date"].isin(date_tgt)].copy()


    def _collect(inv_sub, dom: str):
        X, y, fid = [], [], []
        for r in inv_sub.itertuples(index=False):
            try:
                wins = build_windows_for_file(r.file_path, op_hint=r.ProcessName)
            except Exception:
                continue
            for w in wins:
                X.append(np.asarray(w, dtype=np.float32).reshape(-1))
                y.append(int(r.Label))
                fid.append(str(r.file_path))
        if normalization == "per_window":
            X = [_zscore(x) for x in X]
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")
        return Bundle(X=X, y=y, domain=[dom]*len(y), file_id=fid, fs=float(cfg.FS))

    src = _collect(inv_s, source_domain)
    tgt = _collect(inv_t, target_domain)

    if len(src.y) == 0 or len(tgt.y) == 0:
        raise RuntimeError(f"Empty windows: src={len(src.y)} tgt={len(tgt.y)}. Check domain strings and data_root.")
    return src, tgt


def summarize_bosch(data_root: str) -> Dict[str, Any]:
    keep_ops = getattr(cfg, "OPS_IN_SCOPE", ["OP05", "OP07"])
    inv = build_inventory(dataset_dir=str(data_root), keep_ops=keep_ops)
    inv["domain"] = inv["Machine"].astype(str) + "_" + inv["ProcessName"].astype(str)
    return {
        "n_files": int(len(inv)),
        "domains": inv["domain"].value_counts().to_dict(),
        "labels": inv["Label"].value_counts().to_dict(),
        "fs": float(cfg.FS),
    }
