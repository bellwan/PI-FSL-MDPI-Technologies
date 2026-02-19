from __future__ import annotations

from pathlib import Path
from typing import Union, Dict, Any, List

import json
import math

import pandas as pd


def _load_metadata(metadata_path: Union[str, Path]) -> pd.DataFrame:
    p = Path(metadata_path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {p}")

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _build_dataset_summary(meta_df: pd.DataFrame) -> Dict[str, Any]:
    meta = meta_df.copy()

    summary: Dict[str, Any] = {}

    n_files = int(len(meta))
    total_samples = int(meta["n_samples"].sum()) if "n_samples" in meta.columns else 0
    total_duration_s = float(meta["duration_s"].sum()) if "duration_s" in meta.columns else 0.0
    if "fs_hz" in meta.columns:
        fs_vals: List[float] = sorted(
            meta["fs_hz"].dropna().astype(float).unique().tolist()
        )
    else:
        fs_vals = []

    summary["global"] = {
        "n_files": n_files,
        "total_samples": total_samples,
        "total_duration_s": total_duration_s,
        "total_hours_at_fs": total_duration_s / 3600.0 if total_duration_s else 0.0,
        "fs_hz_unique": fs_vals,
    }

    by_machine = []
    if "Machine" in meta.columns:
        for m, grp in meta.groupby("Machine", dropna=False):
            m_val = m
            if isinstance(m_val, float) and math.isnan(m_val):
                m_val = "UNKNOWN"
            counts = grp["LabelStr"].value_counts() if "LabelStr" in grp.columns else {}
            by_machine.append(
                {
                    "Machine": m_val,
                    "n_files": int(len(grp)),
                    "healthy": int(counts.get("good", 0)),
                    "worn": int(counts.get("bad", 0)),
                    "total_samples": int(
                        grp["n_samples"].sum()
                    ) if "n_samples" in grp.columns else 0,
                    "total_duration_s": float(
                        grp["duration_s"].sum()
                    ) if "duration_s" in grp.columns else 0.0,
                }
            )
    summary["by_machine"] = by_machine

    by_operation = []
    if "ProcessName" in meta.columns:
        for op, grp in meta.groupby("ProcessName", dropna=False):
            op_val = op
            if isinstance(op_val, float) and math.isnan(op_val):
                op_val = "UNKNOWN"
            counts = grp["LabelStr"].value_counts() if "LabelStr" in grp.columns else {}
            by_operation.append(
                {
                    "ProcessName": op_val,
                    "n_files": int(len(grp)),
                    "healthy": int(counts.get("good", 0)),
                    "worn": int(counts.get("bad", 0)),
                    "total_samples": int(
                        grp["n_samples"].sum()
                    ) if "n_samples" in grp.columns else 0,
                    "total_duration_s": float(
                        grp["duration_s"].sum()
                    ) if "duration_s" in grp.columns else 0.0,
                }
            )
    summary["by_operation"] = by_operation

    by_mod = []
    cols_triplet = [c for c in ["Machine", "ProcessName", "Date"] if c in meta.columns]
    if len(cols_triplet) == 3:
        for (m, op, date), grp in meta.groupby(cols_triplet, dropna=False):
            m_val = m
            op_val = op
            date_val = date
            if isinstance(m_val, float) and math.isnan(m_val):
                m_val = "UNKNOWN"
            if isinstance(op_val, float) and math.isnan(op_val):
                op_val = "UNKNOWN"
            if isinstance(date_val, float) and math.isnan(date_val):
                date_val = "UNKNOWN"
            counts = grp["LabelStr"].value_counts() if "LabelStr" in grp.columns else {}
            by_mod.append(
                {
                    "Machine": m_val,
                    "ProcessName": op_val,
                    "Date": date_val,
                    "n_files": int(len(grp)),
                    "healthy": int(counts.get("good", 0)),
                    "worn": int(counts.get("bad", 0)),
                    "total_samples": int(
                        grp["n_samples"].sum()
                    ) if "n_samples" in grp.columns else 0,
                    "total_duration_s": float(
                        grp["duration_s"].sum()
                    ) if "duration_s" in grp.columns else 0.0,
                }
            )
    summary["by_machine_operation_date"] = by_mod

    return summary


def export_dataset_info(
    metadata_path: Union[str, Path],
    json_output_path: Union[str, Path],
) -> Dict[str, Any]:
    meta_df = _load_metadata(metadata_path)
    summary = _build_dataset_summary(meta_df)

    print("\n=== DATASET GLOBAL SUMMARY ===")
    g = summary["global"]
    print(f"Files          : {g['n_files']}")
    print(f"Total samples  : {g['total_samples']:,}")
    print(f"Total duration : {g['total_duration_s']:.2f} s")
    print(f"Total hours    : {g['total_hours_at_fs']:.2f} h")
    print(f"fs_hz_unique   : {g['fs_hz_unique']}")

    print("\n=== BY MACHINE ===")
    for row in summary["by_machine"]:
        print(
            f"{row['Machine']}: files={row['n_files']}, "
            f"healthy={row['healthy']}, worn={row['worn']}, "
            f"hours={row['total_duration_s']/3600.0:.2f}"
        )

    print("\n=== BY OPERATION ===")
    for row in summary["by_operation"]:
        print(
            f"{row['ProcessName']}: files={row['n_files']}, "
            f"healthy={row['healthy']}, worn={row['worn']}, "
            f"hours={row['total_duration_s']/3600.0:.2f}"
        )

    json_output_path = Path(json_output_path)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with json_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved dataset summary JSON to: {json_output_path}")
    return summary
