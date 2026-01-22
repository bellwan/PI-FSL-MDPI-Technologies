from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

TOOL_TABLE = pd.DataFrame(
    [
        ("OP00","Step Drill",        250,100,132),
        ("OP01","Step Drill",        250,100, 29),
        ("OP02","Drill",             200, 50, 42),
        ("OP03","Step Drill",        250,330, 77),
        ("OP04","Step Drill",        250,100, 64),
        ("OP05","Step Drill",        200, 50, 18),
        ("OP06","Step Drill",        250, 50, 91),
        ("OP07","Step Drill",        200, 50, 24),
        ("OP08","Step Drill",        250, 50, 37),
        ("OP09","Straight Flute",    250, 50,102),
        ("OP10","Step Drill",        250, 50, 45),
        ("OP11","Step Drill",        250, 50, 59),
        ("OP12","Step Drill",        250, 50, 46),
        ("OP13","T-Slot Cutter",      75, 25, 32),
        ("OP14","Step Drill",        250,100, 34),
    ],
    columns=["Tool_Operation","Description","Speed_Hz","Feed_mm_s","Duration_s"]
)

def load_metadata(out_root: Path) -> pd.DataFrame:
    p_parq = out_root / "reports" / "metadata.parquet"
    p_csv  = out_root / "reports" / "metadata.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"No metadata found under {out_root}/reports")

def load_npz(npz_path: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as z:
        return dict(acc_xyz=z["acc_xyz"], fs=float(z["fs"]), used_key=str(z["used_key"]))


