from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

DATASET_KEYS_TRY = ["vibration_data", "AI0", "vibration", "acceleration"]
FS_DEFAULT = 2000.0  # Hz


def _norm_slashes(s: str) -> str:
    return s.replace("\\", "/")


def parse_meta_from_path(fp: Path) -> Dict[str, Any]:
    s = _norm_slashes(str(fp)).lower()
    M = (re.search(r"(m\d{2})", s) or [None, None])[1]
    OP = (re.search(r"(op\d{2})", s) or [None, None])[1]
    M = M.upper() if M else None
    OP = OP.upper() if OP else None
    label = "good" if "/good/" in s else ("bad" if "/bad/" in s else None)

    date_tag = None
    m1 = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_](\d{4})", s)
    m2 = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", s)
    m3 = re.search(r"(\d{8})", s)
    if m1:
        date_tag = f"{m1.group(1).title()}_{m1.group(2)}"
    elif m2:
        y, mo = int(m2.group(1)), int(m2.group(2))
        mon = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][mo - 1]
        date_tag = f"{mon}_{y}"
    elif m3:
        ymd = m3.group(1)
        y, mo = int(ymd[:4]), int(ymd[4:6])
        mon = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][mo - 1]
        date_tag = f"{mon}_{y}"

    return dict(
        Machine=M,
        ProcessName=OP,
        LabelStr=label,
        Date=date_tag,
        File=fp.name,
    )


def read_any_h5(fp: Path) -> Tuple[np.ndarray, str]:
    with h5py.File(fp, "r") as f:
        used = None
        arr = None
        for k in DATASET_KEYS_TRY:
            if k in f:
                used, arr = k, np.asarray(f[k][()])
                break
        if used is None:
            keys = list(f.keys())
            if not keys:
                raise RuntimeError(f"{fp.name}: no datasets")
            used, arr = keys[0], np.asarray(f[keys[0]][()])
    a = np.array(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] > 3:
        a = a[:, :3]
    return a, used


def coerce_to_tri(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        x = np.squeeze(x)
        x = x.reshape(-1, 1)
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        out = np.full((x.shape[0], 3), np.nan, dtype=np.float64)
        out[:, 0] = x[:, 0]
        return out
    if x.shape[1] < 3:
        pad = np.full((x.shape[0], 3 - x.shape[1]), np.nan)
        return np.hstack([x, pad])
    return x[:, :3]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def convert_h5_to_npz(
    dataset_root: Path,
    out_root: Path,
    fs_hz: float = FS_DEFAULT,
    n_workers: int = 0,
) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x  

    files = sorted(dataset_root.rglob("*.h5"))
    arrays_dir = out_root / "arrays"
    reports_dir = out_root / "reports"
    _safe_mkdir(arrays_dir)
    _safe_mkdir(reports_dir)

    rows: List[Dict[str, Any]] = []

    def _process(fp: Path):
        try:
            meta = parse_meta_from_path(fp)
            if not (meta["Machine"] and meta["ProcessName"] and meta["LabelStr"]):
                return
            a, used_key = read_any_h5(fp)
            tri = coerce_to_tri(a)
            N = int(tri.shape[0])
            dur = float(N / fs_hz) if fs_hz > 0 else np.nan

            rel = Path(meta["Machine"]) / meta["ProcessName"] / meta["LabelStr"]
            out_dir = arrays_dir / rel
            _safe_mkdir(out_dir)
            out_fp = out_dir / (fp.stem + ".npz")
            np.savez_compressed(out_fp, acc_xyz=tri, fs=fs_hz, used_key=used_key)

            rows.append(
                dict(
                    file_path=str(fp),
                    array_path=str(out_fp),
                    Machine=meta["Machine"],
                    ProcessName=meta["ProcessName"],
                    LabelStr=meta["LabelStr"],
                    Label=int(meta["LabelStr"] == "bad"),
                    Date=meta["Date"],
                    File=meta["File"],
                    n_samples=N,
                    duration_s=dur,
                    fs_hz=fs_hz,
                    used_key=used_key,
                    has_nan_axes=int(np.isnan(tri[:, 1:]).any()),
                    error=None,
                )
            )
        except Exception as e:
            rows.append(dict(file_path=str(fp), error=str(e)))

    if n_workers and n_workers > 0:

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_process, fp) for fp in files]
            for _ in tqdm(as_completed(futs), total=len(futs), desc="Convert"):
                pass
    else:
        for fp in tqdm(files, desc="Convert"):
            _process(fp)

    meta = pd.DataFrame(rows)

    ok = meta["error"].isna() if "error" in meta.columns else pd.Series(True, index=meta.index)
    meta_ok = meta[ok].copy()

    meta_ok.to_parquet(reports_dir / "metadata.parquet", index=False)
    meta_ok.to_csv(reports_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    # stash a small run config
    (reports_dir / "run_config.json").write_text(
        json.dumps(
            dict(
                dataset=str(dataset_root),
                out=str(out_root),
                fs_hz=fs_hz,
                n_files=int(ok.sum()),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return meta_ok


# Backwards-compat alias 
extract_dataset = convert_h5_to_npz


def build_argparser():
    ap = argparse.ArgumentParser("H5 -> NPZ converter")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fs", type=float, default=FS_DEFAULT)
    ap.add_argument("--workers", type=int, default=0)
    return ap


def main(args=None):
    parser = build_argparser()
    opts = parser.parse_args(args=args)

    meta = convert_h5_to_npz(
        dataset_root=Path(opts.dataset_dir),
        out_root=Path(opts.out_dir),
        fs_hz=opts.fs,
        n_workers=opts.workers,
    )
    print(
        f"Done. Converted files: {len(meta)}\n"
        f"Reports: {Path(opts.out_dir) / 'reports'}"
    )


if __name__ == "__main__":
    main()
