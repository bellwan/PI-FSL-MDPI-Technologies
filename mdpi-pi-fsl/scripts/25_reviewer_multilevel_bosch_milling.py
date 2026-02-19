from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
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

    if cmd and str(cmd[0]).lower() == "python":
        cmd = [sys.executable] + cmd[1:]

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=_env_with_src(repo_root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--runner_module", default="pifsl.runner.bench_run_one")
    ap.add_argument("--method", default="pi_fsl")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--label_mode", default="3class", choices=["3class","binary"])
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--seed_count", type=int, default=5)
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt","stft"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    data_root = cfg.get("bosch_mi", {}).get("data_root", cfg.get("bosch_milling", {}).get("data_root", None))
    if data_root is None:
        raise ValueError("Missing bosch_mi/bosch_milling data_root in configs/paths.yaml")

    results_jsonl = Path("artifacts/results/multilevel_bosch_milling.jsonl")

    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        cmd = [
            "python", "-m", args.runner_module,
            "--dataset", "bosch_mi",
            "--data_root", str(data_root),
            "--method", args.method,
            "--variant", args.variant,
            "--seed", str(seed),
            "--n_way", "3",
            "--k_shot", "5",
            "--q_query", "5",
            "--train_episodes", "300",
            "--eval_episodes", "300",
            "--results_jsonl", str(results_jsonl),
            "--tf_rep", args.tf_rep,
            "--comprehensive_metrics",
            "--save_eval_artifacts_dir", "outputs/eval_artifacts",
            "--bosch_mi_label_mode", args.label_mode,
        ]
        run(cmd)


if __name__ == "__main__":
    main()
