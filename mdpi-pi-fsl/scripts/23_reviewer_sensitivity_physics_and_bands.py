from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path
import yaml
import json
import json
import os, sys

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def domain_str(d: dict) -> str:
    return f"{d['machine']}_{d['operation']}"

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]  # repo root 

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
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt", "stft"])
    ap.add_argument("--out_jsonl", default="artifacts/results/jsonl/sensitivity_physics_bands.jsonl")
    ap.add_argument("--override_seeds", default=None)
    ap.add_argument("--only_exp", default="E3", help="Run sensitivity on a single Bosch experiment id (default E3).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    bosch_root = cfg["datasets"]["bosch_drilling_root"]
    yaml_dir = Path(cfg["phase_a"]["bosch_yaml_dir"])
    yaml_files = cfg["phase_a"]["bosch_yaml_files"]

    device = cfg["defaults"]["device"]
    normalization = cfg["defaults"]["normalization"]

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    lambda_energy_grid = [0.0, 0.05, 0.1]
    lambda_spectral_grid = [0.0, 0.05, 0.1]
    lambda_envelope_grid = [0.0, 0.02, 0.05]

    # Band sets: manuscript (0-75/75-300/300-1000) vs loader defaults (0-50/50-200/200-800)
    bands_a = {"low":[0.0,75.0], "mid":[75.0,300.0], "high":[300.0,1000.0]}
    bands_b = {"low":[0.0,50.0], "mid":[50.0,200.0], "high":[200.0,800.0]}
    bands_grid = [bands_a, bands_b]

    for y in yaml_files:
        exp_id = y.split("_")[0]
        if args.only_exp and exp_id != args.only_exp:
            continue

        ypath = yaml_dir / y
        ycfg = load_yaml(ypath)

        seeds = ycfg.get("seeds", [])
        if args.override_seeds:
            seeds = [int(x.strip()) for x in args.override_seeds.split(",") if x.strip()]

        fs = ycfg.get("few_shot_setting", {})
        n_way = int(fs.get("n_way", 2))

        k_raw = fs.get("k_shot", 5)
        k_shot = int(k_raw[0] if isinstance(k_raw, list) else k_raw)

        q_raw = fs.get("q_query", fs.get("query_size", 10))
        q_query = int(q_raw[0] if isinstance(q_raw, list) else q_raw)

        train_cfg = ycfg.get("training_config", {})
        eval_cfg = ycfg.get("evaluation_config", {})

        train_raw = train_cfg.get("train_episodes", fs.get("episodes", cfg["defaults"]["train_episodes"]))
        train_episodes = int(train_raw[0] if isinstance(train_raw, list) else train_raw)

        eval_raw = eval_cfg.get("eval_episodes", eval_cfg.get("episodes", cfg["defaults"]["eval_episodes"]))
        eval_episodes = int(eval_raw[0] if isinstance(eval_raw, list) else eval_raw)

        src_domains = [domain_str(d) for d in ycfg.get("source_domains", [])]
        tgt_domains = [domain_str(d) for d in ycfg.get("target_domains", [])]
        src_domain_arg = ",".join(src_domains)

        for tgt in tgt_domains:
            scenario = f"{exp_id}_{src_domain_arg}_to_{tgt}"

            for seed in seeds:
                for le, ls, la, bands in itertools.product(lambda_energy_grid, lambda_spectral_grid, lambda_envelope_grid, bands_grid):
                    cmd = [
                        "python", "-m", args.runner_module,
                        "--dataset", "bosch",
                        "--data_root", bosch_root,
                        "--method", "pi_fsl",
                        "--variant", "full",
                        "--source_domain", src_domain_arg,
                        "--target_domain", tgt,
                        "--n_way", str(n_way),
                        "--k_shot", str(k_shot),
                        "--q_query", str(q_query),
                        "--train_episodes", str(train_episodes),
                        "--eval_episodes", str(eval_episodes),
                        "--seed", str(seed),
                        "--device", device,
                        "--normalization", normalization,
                        "--results_jsonl", str(out_jsonl),
                        "--exp_id", exp_id,
                        "--scenario", scenario,
                        "--tf_rep", args.tf_rep,
                        "--time_profile",
                        "--transfer_finetune", "--target_adapt_ratio", "0.40",
                        "--lambda_energy", str(le),
                        "--lambda_spectral", str(ls),
                        "--lambda_envelope", str(la),
                        "--spectral_bands_json", json.dumps(bands),
                    ]
                    # Persist detailed metrics/artifacts for reviewer analysis
                    cmd += ["--comprehensive_metrics", "--save_eval_artifacts_dir", "outputs/eval_artifacts"]
                    run(cmd)

if __name__ == "__main__":
    main()
