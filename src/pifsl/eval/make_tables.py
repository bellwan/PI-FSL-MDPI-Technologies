from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


def load_jsonl(path: Path):
    rows = []
    dec = json.JSONDecoder()

    bad_lines = 0
    multi_objects = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            pos = 0
            parsed_any = False
            try:
                while pos < len(s):
                    # skip whitespace
                    while pos < len(s) and s[pos].isspace():
                        pos += 1
                    if pos >= len(s):
                        break

                    obj, end = dec.raw_decode(s, pos)
                    rows.append(obj)
                    parsed_any = True
                    if end < len(s):
                        multi_objects += 1
                    pos = end

                continue
            except json.JSONDecodeError:
                # fallback: try from first '{' or '[' if line has leading junk
                start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
                if start_candidates:
                    start = min(start_candidates)
                    try:
                        obj, end = dec.raw_decode(s, start)
                        rows.append(obj)
                        parsed_any = True
                    except Exception:
                        pass

            if not parsed_any:
                bad_lines += 1

    if bad_lines or multi_objects:
        print(f"[WARN] load_jsonl: skipped {bad_lines} bad line(s); recovered {multi_objects} extra object(s) from concatenated lines.")

    return rows


def main():
    p = argparse.ArgumentParser("Aggregate results.jsonl into tables (CSV + LaTeX)")
    p.add_argument("--results_jsonl", type=str, default="artifacts/results/results.jsonl")
    p.add_argument("--out_dir", type=str, default="tables")
    args = p.parse_args()

    rows = load_jsonl(Path(args.results_jsonl))
    if not rows:
        raise RuntimeError("No results found.")

    df = pd.DataFrame(rows)

    # Common key for grouping (same protocol, multiple seeds)
    group_cols = [
        "dataset","scenario","method","variant","result_source",
        "source_domain","target_domain","n_way","k_shot","q_query",
        "train_episodes","eval_episodes","window_samples","normalization","input_representation"
    ]

    agg = df.groupby(group_cols).agg(
        acc_mean=("acc_mean","mean"),
        bacc_mean=("bacc_mean","mean"),
        macro_f1_mean=("macro_f1_mean","mean"),
        n_runs=("seed","count"),
    ).reset_index()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One "All results" table
    agg.sort_values(["dataset","scenario","method","variant","source_domain","target_domain","k_shot"], inplace=True)
    csv_path = out_dir / "Table_AllResults.csv"
    agg.to_csv(csv_path, index=False)

    tex_path = out_dir / "Table_AllResults.tex"
    agg.to_latex(tex_path, index=False, float_format="%.4f")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {tex_path}")


if __name__ == "__main__":
    main()
