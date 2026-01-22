from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def pick_experiment_yaml(exp_id: str, exp_dir: Path) -> Path:
    matches = sorted(exp_dir.glob(f"{exp_id}_*.yaml"))
    if not matches:
        raise FileNotFoundError(f"Cannot find experiment YAML for {exp_id} in: {exp_dir}")
    return matches[0]


def get_ds_root(cfg: dict, ds: str) -> str:
    ds0 = ds.lower()
    d = cfg["datasets"]
    if ds0 == "bosch":
        return d["bosch_drilling_root"]
    if ds0 == "cwru":
        return d["cwru_root"]
    if ds0 == "pu":
        return d["pu_root"]
    if ds0 == "hust_cn":
        return d["hust_cn_root"]
    if ds0 == "hust_vn":
        return d["hust_vn_root"]
    raise KeyError(f"Dataset root not defined in paths.yaml for dataset: {ds}")


def as_int(v, default: int) -> int:
    if v is None:
        return default
    if isinstance(v, list):
        return int(v[0])
    return int(v)


def build_one_run(
    cfg: dict,
    exp_id: str,
    src_dataset: str,
    src_domain: str,
    tgt_dataset: str,
    tgt_domain: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    method: str,
    seed: int,
) -> list[str]:
    device = str(cfg["defaults"]["device"])
    normalization = str(cfg["defaults"].get("normalization", "per_window"))
    train_episodes = int(cfg["defaults"].get("train_episodes", 500))
    eval_episodes = int(cfg["defaults"].get("eval_episodes", 200))

    results_jsonl = cfg["phase_b"]["results_jsonl"]

    scenario = f"{exp_id}_{method}_{src_dataset}_{src_domain}_to_{tgt_dataset}_{tgt_domain}"

    cmd = [
        "python", "-m", "pifsl.runner.bench_run_one",
        "--method", method,
        "--device", device,
        "--normalization", normalization,
        "--train_episodes", str(train_episodes),
        "--eval_episodes", str(eval_episodes),
        "--n_way", str(n_way),
        "--k_shot", str(k_shot),
        "--q_query", str(q_query),
        "--seed", str(seed),
        "--results_jsonl", results_jsonl,
        "--exp_id", exp_id,
        "--scenario", scenario,
        "--source_domain", src_domain,
        "--target_domain", tgt_domain,
    ]

    if src_dataset.lower() == tgt_dataset.lower():
        cmd += ["--dataset", src_dataset, "--data_root", get_ds_root(cfg, src_dataset)]
    else:
        cmd += [
            "--src_dataset", src_dataset,
            "--tgt_dataset", tgt_dataset,
            "--src_data_root", get_ds_root(cfg, src_dataset),
            "--tgt_data_root", get_ds_root(cfg, tgt_dataset),
        ]

    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--experiments", default=None, help="Comma list like E9,E10. If omitted, uses paths.yaml phase_b.legacy_experiments")
    ap.add_argument("--methods", default="pi_fsl,maml", help="Comma list, default: pi_fsl,maml")
    ap.add_argument("--swap_direction", action="store_true", help="Also run the swapped direction (target->source) for cross-dataset experiments")
    ap.add_argument("--make_table", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))

    exp_ids = cfg["phase_b"]["legacy_experiments"]
    if args.experiments:
        exp_ids = [x.strip() for x in args.experiments.split(",") if x.strip()]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = cfg["defaults"].get("seeds", [1])

    exp_dir = Path(cfg["phase_a"]["bosch_yaml_dir"])

    for exp_id in exp_ids:
        ypath = pick_experiment_yaml(exp_id, exp_dir)
        ycfg = yaml.safe_load(ypath.read_text(encoding="utf-8"))

        src = ycfg["source"]
        tgt = ycfg["target"]
        fs = ycfg.get("few_shot", ycfg.get("few_shot_setting", {}))

        src_dataset = str(src["dataset"])
        src_domain = str(src.get("domain", "ALL"))
        tgt_dataset = str(tgt["dataset"])
        tgt_domain = str(tgt.get("domain", "ALL"))

        n_way = as_int(fs.get("n_way"), 2)
        k_shot = as_int(fs.get("k_shot"), 5)
        q_query = as_int(fs.get("q_query", fs.get("query_size")), 16)

        for method in methods:
            for seed in seeds:
                cmd = build_one_run(
                    cfg, exp_id,
                    src_dataset, src_domain,
                    tgt_dataset, tgt_domain,
                    n_way, k_shot, q_query,
                    method, seed
                )
                run(cmd)

                if args.swap_direction and src_dataset.lower() != tgt_dataset.lower():
                    cmd2 = build_one_run(
                        cfg, exp_id,
                        tgt_dataset, tgt_domain,
                        src_dataset, src_domain,
                        n_way, k_shot, q_query,
                        method, seed
                    )
                    run(cmd2)

    if args.make_table:
        out_dir = str(Path(cfg["artifacts"]["tables"]) / "phase_b")
        run(["python", "scripts/80_make_tables.py", "--results_jsonl", cfg["phase_b"]["results_jsonl"], "--out_dir", out_dir])


if __name__ == "__main__":
    main()
