from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def domain_str(d: dict) -> str:
    # Domain key format: M01_OP07
    return f"{d['machine']}_{d['operation']}"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--runner_module", default="pifsl.runner.bench_run_one",
                    help="Runner module used to execute a single benchmark run.")
    ap.add_argument("--method", default="pi_fsl")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--override_seeds", default=None,
                    help="Override YAML seeds (comma-separated, e.g. 1,2,3).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))

    bosch_root = cfg["datasets"]["bosch_drilling_root"]
    out_jsonl = Path(cfg["phase_a"]["results_jsonl"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    yaml_dir = Path(cfg["phase_a"]["bosch_yaml_dir"])
    yaml_files = cfg["phase_a"]["bosch_yaml_files"]

    device = cfg["defaults"]["device"]
    normalization = cfg["defaults"]["normalization"]

    for y in yaml_files:
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

        exp_id = y.split("_")[0]  # "E1", "E2", "E3", "E4"

        for tgt in tgt_domains:
            scenario = f"{exp_id}_{src_domain_arg}_to_{tgt}"

            for seed in seeds:
                cmd = [
                    "python", "-m", args.runner_module,
                    "--dataset", "bosch",
                    "--data_root", bosch_root,
                    "--method", args.method,
                    "--variant", args.variant,
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
                ]
                # PI-FSL: transfer fine-tune on target samples
                if args.method == "pi_fsl" and args.variant not in ("wo_fsl", "wo_fsl_ft"):
                    cmd += ["--transfer_finetune", "--target_adapt_ratio", "0.40"]

                run(cmd)


if __name__ == "__main__":
    main()
