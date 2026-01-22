from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--module", default="pifsl.eval.make_tables",
                    help="Module that converts JSONL results into tables.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", args.module,
        "--results_jsonl", args.results_jsonl,
        "--out_dir", args.out_dir,
    ]
    run(cmd)


if __name__ == "__main__":
    main()
