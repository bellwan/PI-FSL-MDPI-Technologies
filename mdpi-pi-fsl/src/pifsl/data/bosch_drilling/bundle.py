from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from pifsl.core.base import DatasetBundle, DatasetAdapter
from pifsl.data.bosch_drilling import config as bosch_config
from pifsl.data.common.inventory import build_inventory
from pifsl.data.bosch_drilling.raw_processing.gating import collect_windows

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(x.mean())
    sd = float(x.std() + 1e-8)
    return ((x - mu) / sd).astype(np.float32)

def _unify_lengths(Xs: List[np.ndarray], Xt: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not Xs or not Xt:
        return Xs, Xt
    L = min([len(np.asarray(x).reshape(-1)) for x in Xs] + [len(np.asarray(x).reshape(-1)) for x in Xt])
    def crop(a):
        a = np.asarray(a).reshape(-1)
        return a[:L]
    return [crop(x) for x in Xs], [crop(x) for x in Xt]

class BoschAdapter:
    name = "bosch"

    def load_windows(
        self,
        data_root: str,
        source_domain: str,
        target_domain: str,
        normalization: str = "per_window",
        keep_ops: Optional[List[str]] = None,
    ) -> Tuple[DatasetBundle, DatasetBundle]:
        if keep_ops is None:
            keep_ops = getattr(bosch_config, "OPS_IN_SCOPE", ["OP05","OP07"])
        fs = float(getattr(bosch_config, "FS", 2000.0))

        inv = build_inventory(dataset_dir=str(data_root), keep_ops=keep_ops)

        def filter_domain(dom: str):
            mac, op = dom.split("_", 1)
            mac, op = mac.upper(), op.upper()
            sub = inv[(inv["Machine"].str.upper() == mac) & (inv["ProcessName"].str.upper() == op)].copy()
            return sub

        inv_s = filter_domain(source_domain)
        inv_t = filter_domain(target_domain)

        Xs, ys = collect_windows(inv_s)
        Xt, yt = collect_windows(inv_t)
        Xs, Xt = _unify_lengths(Xs, Xt)

        if normalization == "per_window":
            Xs = [_zscore(x) for x in Xs]
            Xt = [_zscore(x) for x in Xt]
        elif normalization == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        src = DatasetBundle(
            X=[np.asarray(x, dtype=np.float32).reshape(-1) for x in Xs],
            y=[int(v) for v in ys],
            domain=[source_domain]*len(ys),
            file_id=[source_domain]*len(ys),
            fs=fs,
        )
        tgt = DatasetBundle(
            X=[np.asarray(x, dtype=np.float32).reshape(-1) for x in Xt],
            y=[int(v) for v in yt],
            domain=[target_domain]*len(yt),
            file_id=[target_domain]*len(yt),
            fs=fs,
        )
        return src, tgt
