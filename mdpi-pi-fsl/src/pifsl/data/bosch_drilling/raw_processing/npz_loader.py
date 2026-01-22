from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NpzItem:
    acc_xyz: np.ndarray 
    fs: float 
    used_key: str
    row: pd.Series


class NpzDataset:
    def __init__(
        self,
        root: Path | str,
        machines: Optional[Sequence[str]] = None,
        ops: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[int]] = None,   # 0=good, 1=bad
        dates: Optional[Sequence[str]] = None,    # e.g. "Aug_2019"
        require_existing_arrays: bool = True,
    ):
        self.root = Path(root)
        self.reports = self.root / "reports"
        self._meta = self._load_metadata(self.reports)

        # Optional row-level filters without touching files
        m = pd.Series(True, index=self._meta.index)
        if machines:
            machines = [s.upper() for s in machines]
            m &= self._meta["Machine"].isin(machines)
        if ops:
            ops = [s.upper() for s in ops]
            m &= self._meta["ProcessName"].isin(ops)
        if labels is not None:
            m &= self._meta["Label"].isin(list(labels))
        if dates:
            m &= self._meta["Date"].isin(list(dates))

        self._meta = self._meta[m].reset_index(drop=True)

        if require_existing_arrays:
            # keep only rows whose array_path actually exists
            apaths = self._meta["array_path"].apply(Path)
            exists_mask = apaths.apply(lambda p: p.exists())
            self._meta = self._meta[exists_mask].reset_index(drop=True)

    # ------------------------- standard dataset protocol -------------------------

    def __len__(self) -> int:
        return int(len(self._meta))

    def __getitem__(self, idx: int) -> NpzItem:
        row = self._meta.iloc[int(idx)]
        p = Path(row["array_path"])
        if not p.exists():
            raise FileNotFoundError(p)
        with np.load(p, mmap_mode="r", allow_pickle=False) as z:
            acc = z["acc_xyz"] 
            fs = float(z["fs"])
            used_key = str(z["used_key"])
        return NpzItem(acc_xyz=acc, fs=fs, used_key=used_key, row=row)

    # Convenience iterator (never loads all at once)
    def iter(self, shuffle: bool = False, limit: Optional[int] = None) -> Iterable[NpzItem]:
        idxs = np.arange(len(self))
        if shuffle:
            rng = np.random.default_rng(42)
            rng.shuffle(idxs)
        if limit is not None:
            idxs = idxs[:int(limit)]
        for i in idxs:
            yield self[int(i)]

    # Access to the filtered metadata (read-only copy)
    @property
    def metadata(self) -> pd.DataFrame:
        return self._meta.copy()

    # ------------------------- internal -------------------------

    @staticmethod
    def _load_metadata(reports_dir: Path) -> pd.DataFrame:
        parq = reports_dir / "metadata.parquet"
        csv  = reports_dir / "metadata.csv"
        if parq.exists():
            return pd.read_parquet(parq)
        if csv.exists():
            return pd.read_csv(csv)
        raise FileNotFoundError(f"metadata.parquet/csv not found under {reports_dir}")

