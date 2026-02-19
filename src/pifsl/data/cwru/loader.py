from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from pifsl.data.cwru.preprocess import load_CWRU_dataset


@dataclass
class Bundle:
    X: List[np.ndarray]
    y: List[int]
    domain: List[str]
    file_id: List[str]
    fs: float


def load_cwru_windows(
    data_root: str,
    source_domain: str,
    target_domain: str,
    time_steps: int = 1024,
    overlap_ratio: float = 0.5,
    normalization: str = "per_window",
    seed: int = 1,
    label_mode: str = "binary",  
    fs: float = 12000.0,
    pos_label: int | None = None,
) -> Tuple[Bundle, Bundle]:
    def _one(dom: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
        ds = load_CWRU_dataset(
            domain=int(dom),
            dir_path=str(data_root),
            time_steps=int(time_steps),
            overlap_ratio=float(overlap_ratio),
            normalization=(normalization != "none"),
            random_seed=int(seed),
            raw=False,
            fft=False,
        )

        X: List[np.ndarray] = []
        y: List[int] = []
        fid: List[str] = []

        rng = np.random.RandomState(int(seed) + int(dom) * 100)

        for label, segs in ds.items():
            lab_i = int(label)

            # Option-Y filter: keep only healthy(0) and selected fault label
            if label_mode == "binary" and pos_label is not None:
                if lab_i not in (0, int(pos_label)):
                    continue

            segs = list(segs)
            n = len(segs)
            if n == 0:
                continue

            # Split windows into 2 pools (pool0 for support, pool1 for query)
            perm = rng.permutation(n)
            cut = max(1, n // 2)
            pool0 = set(perm[:cut].tolist())

            for j, seg in enumerate(segs):
                X.append(np.asarray(seg, dtype=np.float32).reshape(-1))
                if label_mode == "binary":
                    y.append(0 if lab_i == 0 else 1)
                else:
                    y.append(lab_i)

                pool = 0 if j in pool0 else 1
                fid.append(f"{dom}_label{lab_i}_pool{pool}")

        return X, y, fid


    def _expand(dom_spec: str) -> List[str]:
        ds = str(dom_spec).strip()
        if ds.upper() == "ALL":
            return ["0", "1", "2", "3"]
        return [ds]

    def _many(dom_spec: str) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
        X_all: List[np.ndarray] = []
        y_all: List[int] = []
        fid_all: List[str] = []
        dom_all: List[str] = []
        for d in _expand(dom_spec):
            X, y, fid = _one(d)
            X_all.extend(X)
            y_all.extend(y)
            fid_all.extend(fid)
            dom_all.extend([d] * len(y))
        return X_all, y_all, fid_all, dom_all

    Xs, ys, fids, doms = _many(source_domain)
    Xt, yt, fidt, domt = _many(target_domain)

    src = Bundle(X=Xs, y=ys, domain=doms, file_id=fids, fs=float(fs))
    tgt = Bundle(X=Xt, y=yt, domain=domt, file_id=fidt, fs=float(fs))


    if len(src.y) == 0 or len(tgt.y) == 0:
        raise RuntimeError(f"CWRU empty windows: src={len(src.y)} tgt={len(tgt.y)}. Check data_root/domains.")

    # In Option-Y, enforce both classes exist in both splits
    if label_mode == "binary" and pos_label is not None:
        for name, b in ("src", src), ("tgt", tgt):
            u = set(int(v) for v in b.y)
            if u != {0, 1}:
                raise RuntimeError(
                    f"CWRU Option-Y split has missing class(es): {name} classes={sorted(u)}. "
                    f"Try a different pos_label or check domain filtering."
                )

    return src, tgt


def summarize_cwru() -> Dict[str, Any]:
    return {"domains_expected": ["0", "1", "2", "3"], "fs_typical": 12000.0}
