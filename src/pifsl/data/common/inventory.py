import os
import re
from typing import Any, Dict, List
import pandas as pd

_LBL_PAT = re.compile(r"[\\/](good|bad|healthy|worn)[\\/]", re.IGNORECASE)
_M_PAT   = re.compile(r"(M\d{2})", re.IGNORECASE)
_OP_PAT  = re.compile(r"(OP\d{2})", re.IGNORECASE)
_DATE_PAT = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_-](\d{4})", re.IGNORECASE)

def parse_meta_from_path(fp: str) -> Dict[str, Any]:
    s = fp.replace("\\","/")
    name = os.path.basename(fp)

    m_lbl = _LBL_PAT.search(s)
    label = None
    if m_lbl:
        raw = m_lbl.group(1).lower()
        label = {"good":"healthy","bad":"worn","healthy":"healthy","worn":"worn"}[raw]

    m_m = _M_PAT.search(s)
    machine = m_m.group(1).upper() if m_m else None

    m_op = _OP_PAT.search(s)
    op = m_op.group(1).upper() if m_op else None

    m_dt = _DATE_PAT.search(s)
    date_tag = f"{m_dt.group(1).title()}_{m_dt.group(2)}" if m_dt else None

    return dict(File=name, Machine=machine, ProcessName=op, Date=date_tag, Label=label)

def build_inventory(dataset_dir: str, keep_ops: List[str]) -> pd.DataFrame:
    # fixed schema to avoid KeyError anywhere downstream
    cols = ["file_path","File","Machine","ProcessName","Date","LabelStr","Label","N","duration_s"]
    rows: List[Dict[str, Any]] = []

    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            ext = fn.lower()
            if not (ext.endswith(".h5") or ext.endswith(".npz")):
                continue
            fp = os.path.join(root, fn)
            meta = parse_meta_from_path(fp)

            op = meta.get("ProcessName")
            lab = meta.get("Label")
            mac = meta.get("Machine")

            if op not in keep_ops:
                continue
            if lab not in ("healthy","worn"):
                continue
            if mac is None:
                continue

            rows.append({
                "file_path": fp,
                "File": meta.get("File"),
                "Machine": mac,
                "ProcessName": op,
                "Date": meta.get("Date"),
                "LabelStr": lab,
                "Label": 0 if lab == "healthy" else 1,
                "N": None,
                "duration_s": None,
            })

    if not rows:
        return pd.DataFrame(columns=cols)

    inv = pd.DataFrame(rows, columns=cols)

    # ultra-safe sort: only sort by columns that truly exist in this frame
    want = ["Machine","ProcessName","LabelStr","file_path"]
    have = [k for k in want if k in inv.columns]
    if have:
        inv = inv.sort_values(have, kind="mergesort").reset_index(drop=True)

    return inv
