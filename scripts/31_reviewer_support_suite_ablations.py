from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

import yaml
import json
import os, sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]  # Repository root (script is under scripts/).

def _env_with_src(repo_root: Path) -> dict:
    env = os.environ.copy()
    src = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = src + ((";" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    return env

def run(cmd: list[str]) -> None:
    repo_root = _repo_root()

    # Use current interpreter (conda/venv) instead of `python` on PATH
    if cmd and str(cmd[0]).lower() == "python":
        cmd = [sys.executable] + cmd[1:]

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=_env_with_src(repo_root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--runner_module", default="pifsl.runner.bench_run_one")
    ap.add_argument("--datasets", default="cwru,hust_cn,hust_vn,pu,bosch_mi")
    ap.add_argument("--methods", default="pi_fsl")
    ap.add_argument("--variants", default="full,wo_physics,energy_only,spectral_only,envelope_only")
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt","stft"])
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--seed_count", type=int, default=3)
    ap.add_argument("--eval_episodes", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    results_jsonl = Path("artifacts/results/support_suite_ablations.jsonl")

    datasets = [s.strip().lower() for s in args.datasets.split(",") if s.strip()]
    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    variants = [s.strip().lower() for s in args.variants.split(",") if s.strip()]

    # Dataset roots are expected in configs/paths.yaml; missing entries are treated as configuration errors.
    roots = {
        "cwru": cfg.get("cwru", {}).get("data_root", None),
        "hust_cn": cfg.get("hust_cn", {}).get("data_root", None),
        "hust_vn": cfg.get("hust_vn", {}).get("data_root", None),
        "pu": cfg.get("pu", {}).get("data_root", None),
        "bosch_mi": cfg.get("bosch_mi", {}).get("data_root", cfg.get("bosch_milling", {}).get("data_root", None)),
    }

    for ds in datasets:
        if ds not in roots or roots[ds] is None:
            raise ValueError(f"Missing data_root for dataset '{ds}' in configs/paths.yaml")
        data_root = roots[ds]

        for method in methods:
            for variant in variants:
                for seed in range(args.seed_start, args.seed_start + args.seed_count):
                    cmd = [
                        "python", "-m", args.runner_module,
                        "--dataset", ds,
                        "--data_root", str(data_root),
                        "--method", method,
                        "--variant", variant,
                        "--seed", str(seed),
                        "--n_way", "2",
                        "--k_shot", "5",
                        "--q_query", "5",
                        "--train_episodes", "200",
                        "--eval_episodes", str(args.eval_episodes),
                        "--results_jsonl", str(results_jsonl),
                        "--tf_rep", args.tf_rep,
                        "--comprehensive_metrics",
                        "--save_eval_artifacts_dir", "outputs/eval_artifacts",
                    ]
                    run(cmd)


if __name__ == "__main__":
    main()
