from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml
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
    ap.add_argument("--out_jsonl", default="artifacts/results/jsonl/latency_profile.jsonl")
    ap.add_argument("--exp", default="E3", help="Which Bosch experiment id to profile (default E3).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method_list", default="pi_fsl,protonet,matchingnet,deep_coral")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    bosch_root = cfg["datasets"]["bosch_drilling_root"]

    yaml_dir = Path(cfg["phase_a"]["bosch_yaml_dir"])
    yaml_files = cfg["phase_a"]["bosch_yaml_files"]

    device = cfg["defaults"]["device"]
    normalization = cfg["defaults"]["normalization"]

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.method_list.split(",") if m.strip()]

    # Select first Phase-A YAML whose filename prefix matches --exp (E1..E4)
    chosen = None
    for y in yaml_files:
        if y.split("_")[0] == args.exp:
            chosen = y
            break
    if chosen is None:
        raise SystemExit(f"No YAML found for exp={args.exp}. Check configs/paths.yaml phase_a settings.")

    ycfg = load_yaml(yaml_dir / chosen)

    fs = ycfg.get("few_shot_setting", {})
    n_way = int(fs.get("n_way", 2))

    k_raw = fs.get("k_shot", 5)
    k_shot = int(k_raw[0] if isinstance(k_raw, list) else k_raw)

    q_raw = fs.get("q_query", fs.get("query_size", 10))
    q_query = int(q_raw[0] if isinstance(q_raw, list) else q_raw)

    train_cfg = ycfg.get("training_config", {})
    eval_cfg = ycfg.get("evaluation_config", {})

    train_episodes = int(train_cfg.get("train_episodes", cfg["defaults"]["train_episodes"]))
    eval_episodes = int(eval_cfg.get("eval_episodes", cfg["defaults"]["eval_episodes"]))

    src_domains = [domain_str(d) for d in ycfg.get("source_domains", [])]
    tgt_domains = [domain_str(d) for d in ycfg.get("target_domains", [])]
    src_domain_arg = ",".join(src_domains)

    tgt = tgt_domains[0]
    scenario = f"{args.exp}_{src_domain_arg}_to_{tgt}"

    for method in methods:
        cmd = [
            "python", "-m", args.runner_module,
            "--dataset", "bosch",
            "--data_root", bosch_root,
            "--method", method,
            "--variant", "full",
            "--source_domain", src_domain_arg,
            "--target_domain", tgt,
            "--n_way", str(n_way),
            "--k_shot", str(k_shot),
            "--q_query", str(q_query),
            "--train_episodes", str(train_episodes),
            "--eval_episodes", str(eval_episodes),
            "--seed", str(args.seed),
            "--device", device,
            "--normalization", normalization,
            "--results_jsonl", str(out_jsonl),
            "--exp_id", args.exp,
            "--scenario", scenario,
            "--tf_rep", args.tf_rep,
            "--time_profile",
        ]

        # PI-FSL: enable target-domain fine-tune step
        if method == "pi_fsl":
            cmd += ["--transfer_finetune", "--target_adapt_ratio", "0.40"]

        # Persist detailed metrics/artifacts for profiling and auditability
        cmd += ["--comprehensive_metrics", "--save_eval_artifacts_dir", "outputs/eval_artifacts"]

        run(cmd)

    # Extract per-method timing stats from JSONL into CSV for plotting
    run(["python", "tools/jsonl_extract_latency_to_csv.py", "--jsonl", str(out_jsonl)])

if __name__ == "__main__":
    main()
