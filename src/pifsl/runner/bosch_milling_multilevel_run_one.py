from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from pifsl.eval.ci import bootstrap
from pifsl.runner.bench_utils import set_seed
from pifsl.eval.schema import append_jsonl, utc_now_iso

from pifsl.runner.bench_run_one import main as bench_run_one_main


def parse_args():
    p = argparse.ArgumentParser("Bosch milling multilevel wrapper -> calls bench_run_one with explicit data_root")
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--data_root", type=str, required=True)

    # common protocol
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_way", type=int, default=3)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=5)
    p.add_argument("--train_episodes", type=int, default=300)
    p.add_argument("--eval_episodes", type=int, default=300)

    # dataset specifics
    p.add_argument("--bosch_mi_label_mode", type=str, default="3class", choices=["binary", "3class"])
    p.add_argument("--tf_rep", type=str, default="cwt", choices=["cwt", "stft"])

    # output
    p.add_argument("--results_jsonl", type=str, default="artifacts/results/jsonl/multilevel_bosch_milling.jsonl")
    p.add_argument("--save_eval_artifacts_dir", type=str, default="outputs/eval_artifacts")
    p.add_argument("--scenario", type=str, default="")
    p.add_argument("--exp_id", type=str, default="")
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    bootstrap(project_root)
    set_seed(int(args.seed))

    import sys

    scenario = args.scenario.strip() or f"bosch_milling_multilevel_seed{args.seed}"

    sys.argv = [
        "bench_run_one",
        "--dataset", "bosch_mi",
        "--data_root", str(args.data_root),

        "--method", "pi_fsl",
        "--variant", "full",

        "--bosch_mi_label_mode", str(args.bosch_mi_label_mode),
        "--tf_rep", str(args.tf_rep),

        "--seed", str(args.seed),
        "--n_way", str(args.n_way),
        "--k_shot", str(args.k_shot),
        "--q_query", str(args.q_query),
        "--train_episodes", str(args.train_episodes),
        "--eval_episodes", str(args.eval_episodes),

        "--comprehensive_metrics",
        "--save_eval_artifacts_dir", str(args.save_eval_artifacts_dir),
        "--results_jsonl", str(args.results_jsonl),

        "--exp_id", str(args.exp_id),
        "--scenario", str(scenario),
        "--notes", str(args.notes),
    ]

    bench_run_one_main()


if __name__ == "__main__":
    main()
