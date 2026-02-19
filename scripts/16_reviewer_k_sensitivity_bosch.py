from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def domain_str(d: dict) -> str:
    return f"{d['machine']}_{d['operation']}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def env_with_src(root: Path) -> dict:
    env = os.environ.copy()
    src = str((root / "src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + ((";" + prev) if prev else "")
    return env


def load_done_pairs(jsonl_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        scenario = r.get("scenario")
        seed = r.get("seed")
        if isinstance(scenario, str) and isinstance(seed, int):
            done.add((scenario, seed))
    return done


def run(cmd: list[str], root: Path) -> None:
    if cmd and str(cmd[0]).lower() == "python":
        cmd = [sys.executable] + cmd[1:]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env_with_src(root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--runner_module", default="pifsl.runner.bench_run_one")
    ap.add_argument("--method", default="pi_fsl")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--exp", default="E3", choices=["E1", "E2", "E3", "E4"])
    ap.add_argument("--k_values", default="1,5")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--seed_count", type=int, default=5)
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt", "stft"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--normalization", default="per_window")
    ap.add_argument("--balance_mode", default="none", choices=["none", "psd", "random"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    cfg = load_yaml(root / args.paths)

    bosch_root = cfg["datasets"]["bosch_drilling_root"]
    yaml_dir = Path(cfg["phase_a"]["bosch_yaml_dir"])
    yaml_files = cfg["phase_a"]["bosch_yaml_files"]

    yname = next((y for y in yaml_files if y.split("_")[0] == args.exp), None)
    if yname is None:
        raise KeyError(f"Cannot find YAML for exp={args.exp} in phase_a.bosch_yaml_files")
    ycfg = load_yaml(yaml_dir / yname)

    src_domains = [domain_str(d) for d in ycfg.get("source_domains", [])]
    tgt_domains = [domain_str(d) for d in ycfg.get("target_domains", [])]
    if not src_domains or not tgt_domains:
        raise KeyError(f"YAML {yname} missing source_domains or target_domains")

    src_domain_arg = ",".join(src_domains)

    fs = ycfg.get("few_shot_setting", {})
    n_way = int(fs.get("n_way", 2))
    q_raw = fs.get("q_query", fs.get("query_size", 15))
    q_query = int(q_raw[0] if isinstance(q_raw, list) else q_raw)

    train_cfg = ycfg.get("training_config", {})
    eval_cfg = ycfg.get("evaluation_config", {})
    train_raw = train_cfg.get("train_episodes", cfg["defaults"]["train_episodes"])
    train_episodes = int(train_raw[0] if isinstance(train_raw, list) else train_raw)
    eval_raw = eval_cfg.get("eval_episodes", cfg["defaults"]["eval_episodes"])
    eval_episodes = int(eval_raw[0] if isinstance(eval_raw, list) else eval_raw)

    ks = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    if not ks:
        raise ValueError("Empty --k_values")

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = ycfg.get("seeds", [])
        if not seeds:
            seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    results_jsonl = root / "artifacts" / "results" / "jsonl" / "k_sensitivity_bosch.jsonl"
    results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_pairs(results_jsonl) if args.resume else set()

    eval_dir = root / "outputs" / "eval_artifacts"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for tgt in tgt_domains:
        for k in ks:
            for seed in seeds:
                scenario = f"{args.exp}_{src_domain_arg}_to_{tgt}__k{k}__bal_{args.balance_mode}"
                if args.resume and (scenario, int(seed)) in done:
                    print(f"[SKIP] {scenario} seed={seed} (already in JSONL)")
                    continue

                cmd = [
                    "python", "-m", args.runner_module,
                    "--dataset", "bosch",
                    "--data_root", str(bosch_root),
                    "--method", args.method,
                    "--variant", args.variant,
                    "--source_domain", src_domain_arg,
                    "--target_domain", str(tgt),
                    "--n_way", str(n_way),
                    "--k_shot", str(k),
                    "--q_query", str(q_query),
                    "--train_episodes", str(train_episodes),
                    "--eval_episodes", str(eval_episodes),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--normalization", args.normalization,
                    "--tf_rep", args.tf_rep,
                    "--results_jsonl", str(results_jsonl),
                    "--exp_id", args.exp,
                    "--scenario", scenario,
                    "--comprehensive_metrics",
                    "--save_eval_artifacts_dir", str(eval_dir),
                ]

                if args.method == "pi_fsl":
                    cmd += ["--transfer_finetune", "--target_adapt_ratio", "0.40"]

                if args.balance_mode == "psd":
                    cmd += ["--psd_guided_balance"]
                elif args.balance_mode == "random":
                    cmd += ["--balance_mode", "random"]

                run(cmd, root)


if __name__ == "__main__":
    main()
