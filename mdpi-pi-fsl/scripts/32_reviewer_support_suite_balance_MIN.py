from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_with_src(repo_root: Path) -> dict:
    env = os.environ.copy()
    src = str((repo_root / "src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + ((";" + prev) if prev else "")
    return env


def _run(cmd: list[str]) -> None:
    repo_root = _repo_root()

    if cmd and str(cmd[0]).lower() == "python":
        cmd = [sys.executable] + cmd[1:]

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=_env_with_src(repo_root))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--runner_module", default="pifsl.runner.bench_run_one")

    ap.add_argument("--suite_datasets", default="bosch,pu")
    ap.add_argument("--seeds", default="1,2,3")

    ap.add_argument("--train_episodes", type=int, default=200)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt", "stft"])

    ap.add_argument("--out_jsonl", default="artifacts/results/jsonl/table12_balance.jsonl")
    ap.add_argument("--save_eval_artifacts_dir", default="outputs/eval_artifacts")

    args = ap.parse_args()

    root = _repo_root()
    cfg = _load_yaml(root / args.paths)

    suite = cfg.get("phase_c", {}).get("suite", [])
    if not suite:
        raise SystemExit("configs/paths.yaml missing phase_c.suite")

    want = {s.strip().lower() for s in args.suite_datasets.split(",") if s.strip()}
    suite_sel = [s for s in suite if str(s.get("dataset", "")).lower() in want]
    if not suite_sel:
        raise SystemExit(f"No suite entries matched --suite_datasets={args.suite_datasets}")

    ds_roots = cfg.get("datasets", {})

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    method = "pi_fsl"
    variant = "full"

    modes = [
        ("none", ["--balance_mode", "none"]),
        ("psd", ["--balance_mode", "psd"]),
        ("random", ["--balance_mode", "random"]),
    ]

    for entry in suite_sel:
        dataset = str(entry["dataset"]).lower()
        data_root_key = str(entry["data_root"]).strip()
        data_root = ds_roots.get(data_root_key)
        if not data_root:
            raise KeyError(f"paths.yaml missing datasets.{data_root_key}")

        scenario_base = str(entry.get("scenario", f"{dataset}_suite")).strip()
        exp_id = str(entry.get("exp_id", "SUP_BAL")).strip()

        source_domain = str(entry.get("source_domain", "")).strip()
        target_domain = str(entry.get("target_domain", "")).strip()
        q_query = int(entry.get("q_query", 10))
        extra_args = list(entry.get("extra_args", []))

        for seed in seeds:
            for mode_name, mode_args in modes:
                scenario = f"{scenario_base}__bal_{mode_name}"

                cmd = [
                    "python", "-m", args.runner_module,
                    "--dataset", dataset,
                    "--data_root", str(data_root),
                    "--method", method,
                    "--variant", variant,
                    "--source_domain", source_domain,
                    "--target_domain", target_domain,
                    "--n_way", "2",
                    "--k_shot", "5",
                    "--q_query", str(q_query),
                    "--train_episodes", str(args.train_episodes),
                    "--eval_episodes", str(args.eval_episodes),
                    "--seed", str(seed),
                    "--tf_rep", args.tf_rep,
                    "--results_jsonl", str((root / args.out_jsonl).as_posix()),
                    "--exp_id", exp_id,
                    "--scenario", scenario,
                    "--comprehensive_metrics",
                    "--save_eval_artifacts_dir", args.save_eval_artifacts_dir,
                    "--notes", f"table12_balance;dataset={dataset};variant=full;bal={mode_name}",
                ] + mode_args + extra_args

                cmd += ["--transfer_finetune", "--target_adapt_ratio", "0.40"]

                _run(cmd)


if __name__ == "__main__":
    main()
