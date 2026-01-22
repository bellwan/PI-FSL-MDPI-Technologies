from __future__ import annotations
from typing import Tuple
import numpy as np
from pifsl.core.base import DatasetBundle
from pathlib import Path
from pifsl.data.cwru.preprocess import load_CWRU_dataset

class CWRUAdapter:
    name = "cwru"

    def load_windows(
        self,
        data_root: str,
        source_domain: str,
        target_domain: str,
        window_seconds: float = 0.0853,
        overlap_ratio: float = 0.5,
        normalization: str = "per_window",
        seed: int = 1,
        label_mode: str = "binary",
        fs_override: float = 12000.0,
    ) -> Tuple[DatasetBundle, DatasetBundle]:
        # preprocess_cwru expects dir_path to be the parent folder containing "CWRU_12k/"
        root = Path(data_root)

        if (root / "CWRU_12k").exists():
            dir_path = str(root)  
        elif (root / "Drive_end_0").exists():
            dir_path = str(root.parent) 
        else:
            # fallback: keep as-is, but error message will show attempted path
            dir_path = str(root)


        fs = float(fs_override)
        time_steps = int(round(window_seconds * fs))
        time_steps = max(256, time_steps)

        def to_xy(dom: str):
            ds = load_CWRU_dataset(
                domain=int(dom),
                dir_path=dir_path,
                time_steps=time_steps,
                overlap_ratio=float(overlap_ratio),
                normalization=(normalization != "none"),
                random_seed=int(seed),
                raw=False,
                fft=False,
            )
            X, y = [], []
            for label, segs in ds.items():
                for seg in segs:
                    X.append(np.asarray(seg, dtype=np.float32).reshape(-1))
                    if label_mode == "binary":
                        y.append(0 if int(label) == 0 else 1)
                    else:
                        y.append(int(label))
            return X, y

        Xs, ys = to_xy(source_domain)
        Xt, yt = to_xy(target_domain)

        src = DatasetBundle(X=Xs, y=ys, domain=[source_domain]*len(ys), file_id=[source_domain]*len(ys), fs=fs)
        tgt = DatasetBundle(X=Xt, y=yt, domain=[target_domain]*len(yt), file_id=[target_domain]*len(yt), fs=fs)
        return src, tgt
