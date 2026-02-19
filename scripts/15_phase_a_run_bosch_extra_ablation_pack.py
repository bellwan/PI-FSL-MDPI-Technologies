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

def load_done_runs(jsonl_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            done.add((str(o.get("scenario", "")), int(o.get("seed", 0))))
        except Exception:
            continue
    return done


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]  

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
    ap.add_argument("--method", default="pi_fsl", choices=["pi_fsl", "relationnet", "protonet", "matchingnet", "maml"])
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt", "stft"], help="Representation ablation")
    ap.add_argument(
        "--variants",
        default="full,wo_physics,energy_only,spectral_only,envelope_only",
        help="Comma-separated variants to run (bench_run_one --variant)",
    )
    ap.add_argument(
        "--psd_balance",
        default="both",
        choices=["off", "on", "both"],
        help="Run PSD-guided source balancing ablation (Bosch/PI-FSL only).",
    )
    ap.add_argument("--override_seeds", default=None, help="Override YAML seeds (comma-separated, e.g. 1,2,3).")
    ap.add_argument("--resume", action="store_true", help="Skip (scenario,seed) already present in output JSONL.")
    ap.add_argument(
        "--only_exp_ids",
        default="",
        help="Comma-separated exp IDs to run (e.g. E1,E2,E3). Empty = run all YAMLs.",
    )

    ap.add_argument(
        "--save_eval_artifacts_dir",
        default="outputs/eval_artifacts",
        help="Where to save per-run eval_*.npz (requires --comprehensive_metrics)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))

    bosch_root = cfg["datasets"]["bosch_drilling_root"]
    out_jsonl = Path(cfg["phase_a"]["results_jsonl"]).with_name("phase_a_bosch_extra_ablations.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_runs(out_jsonl) if args.resume else set()

    yaml_dir = Path(cfg["phase_a"]["bosch_yaml_dir"])
    yaml_files = cfg["phase_a"]["bosch_yaml_files"]
    if args.only_exp_ids.strip():
        allow = {x.strip() for x in args.only_exp_ids.split(",") if x.strip()}
        yaml_files = [y for y in yaml_files if y.split("_")[0] in allow]


    device = cfg["defaults"]["device"]
    normalization = cfg["defaults"]["normalization"]

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    if args.psd_balance == "both":
        psd_modes = ["off", "on"]
    else:
        psd_modes = [args.psd_balance]

    for y in yaml_files:
        ypath = yaml_dir / y
        ycfg = load_yaml(ypath)

        seeds = ycfg.get("seeds", [])
        if args.override_seeds:
            seeds = [int(x.strip()) for x in args.override_seeds.split(",") if x.strip()]

        fs_cfg = ycfg.get("few_shot_setting", {})
        n_way = int(fs_cfg.get("n_way", 2))

        k_raw = fs_cfg.get("k_shot", 5)
        k_shot = int(k_raw[0] if isinstance(k_raw, list) else k_raw)

        q_raw = fs_cfg.get("q_query", fs_cfg.get("query_size", 10))
        q_query = int(q_raw[0] if isinstance(q_raw, list) else q_raw)

        train_cfg = ycfg.get("training_config", {})
        eval_cfg = ycfg.get("evaluation_config", {})

        train_raw = train_cfg.get("train_episodes", fs_cfg.get("episodes", cfg["defaults"]["train_episodes"]))
        train_episodes = int(train_raw[0] if isinstance(train_raw, list) else train_raw)

        eval_raw = eval_cfg.get("eval_episodes", eval_cfg.get("episodes", cfg["defaults"]["eval_episodes"]))
        eval_episodes = int(eval_raw[0] if isinstance(eval_raw, list) else eval_raw)

        src_domains = [domain_str(d) for d in ycfg.get("source_domains", [])]
        tgt_domains = [domain_str(d) for d in ycfg.get("target_domains", [])]

        src_domain_arg = ",".join(src_domains)
        exp_id = y.split("_")[0]  # E1..E4

        for tgt in tgt_domains:
            for variant in variants:
                for psd_mode in psd_modes:
                    # PSD-guided balance applies only to PI-FSL variants using FSL
                    psd_flag = []
                    notes = []
                    if args.method == "pi_fsl" and psd_mode == "on" and variant != "wo_fsl":
                        psd_flag = ["--psd_guided_balance"]
                        notes.append("psd_guided_balance")

                    scenario = f"{exp_id}_{src_domain_arg}_to_{tgt}__{args.tf_rep}__{variant}__psd_{psd_mode}"

                    for seed in seeds:
                        if args.resume and (scenario, int(seed)) in done:
                            print(f"[SKIP] {scenario} seed={seed} (already in JSONL)")
                            continue
                        cmd = [
                            "python", "-m", args.runner_module,
                            "--dataset", "bosch",
                            "--data_root", bosch_root,
                            "--method", args.method,
                            "--variant", variant,
                            "--tf_rep", args.tf_rep,
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
                            "--comprehensive_metrics",
                            "--save_eval_artifacts_dir", str(Path(args.save_eval_artifacts_dir)),
                        ]

                        # PI-FSL: optional target-domain fine-tune (kept consistent with phase-a runner)
                        if args.method == "pi_fsl" and variant not in ("wo_fsl", "wo_fsl_ft"):
                            cmd += ["--transfer_finetune", "--target_adapt_ratio", "0.40"]

                        if notes:
                            cmd += ["--notes", ";".join(notes)]

                        cmd += psd_flag
                        run(cmd)


if __name__ == "__main__":
    main()
