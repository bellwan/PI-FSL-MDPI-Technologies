from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


METHODS_5 = [
    ("relationnet", "full"),
    ("protonet", "full"),
    ("matchingnet", "full"),
    ("maml", "full"),
    ("pi_fsl", "full"),
]


def _csv_set(s: str | None) -> set[str] | None:
    if not s:
        return None
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def _milling_has_feature_table(r: Path) -> bool:
    if r.is_file():
        return True
    if not r.exists():
        return False
    for pat in ("*.parquet", "*.pq", "*feature*.csv", "*features*.csv", "*processed*.csv", "*table*.csv"):
        if any(r.rglob(pat)):
            return True
    return False


def choose_milling_domains_option_y(
    milling_root: str,
    *,
    max_rows: int = 50000,
    min_per_class: int = 15,
) -> Tuple[str, str, bool, Optional[int]]:
    from pifsl.data.bosch_milling.loader import load_bosch_mi_source

    root = Path(milling_root)
    has_ft = _milling_has_feature_table(root)

    if has_ft:
        sel_max_files = int(max_rows) if max_rows else None
    else:
        sel_max_files = int(min(max_rows, 80)) if max_rows else 80

    b = load_bosch_mi_source(
        data_root=str(milling_root),
        normalization="zscore",
        seed=0,
        max_files=sel_max_files,
        label_mode="3class",
        prefer_feature_table=True,
        include_domains=None,
    )

    counts: Dict[str, Dict[int, int]] = {}
    for dom, y in zip(b.domain, b.y):
        d = counts.setdefault(str(dom), {})
        d[int(y)] = d.get(int(y), 0) + 1

    feasible: List[Tuple[int, str]] = []
    for dom, c in counts.items():
        c0 = c.get(0, 0)
        c2 = c.get(2, 0)
        if min(c0, c2) >= int(min_per_class):
            feasible.append((min(c0, c2), dom))

    feasible.sort(reverse=True)
    if len(feasible) >= 2:
        src, tgt = feasible[0][1], feasible[1][1]
    else:
        fallback = sorted([(min(c.get(0, 0), c.get(2, 0)), dom) for dom, c in counts.items()], reverse=True)
        if len(fallback) < 2:
            raise RuntimeError("[bosch_mi] Not enough domains found for AUTO selection.")
        src, tgt = fallback[0][1], fallback[1][1]

    cap_for_runs = None if has_ft else int(max_rows)
    return src, tgt, has_ft, cap_for_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--make_tables", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--milling_max_rows", type=int, default=50000)
    ap.add_argument("--milling_min_per_class", type=int, default=15)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))

    device = args.device or cfg["defaults"]["device"]
    seeds = args.seeds or ",".join(str(s) for s in cfg["defaults"]["seeds"])
    seeds_list = [int(x.strip()) for x in seeds.split(",") if x.strip()]

    train_episodes = int(cfg["defaults"]["train_episodes"])
    eval_episodes = int(cfg["defaults"]["eval_episodes"])
    normalization = str(cfg["defaults"].get("normalization", "per_window"))

    results_jsonl = str(cfg["phase_c"]["results_jsonl"])
    tables_dir = str(cfg["phase_c"]["tables_dir"])

    dcfg = cfg["datasets"]
    suite = cfg["phase_c"]["suite"]

    only = _csv_set(args.only)
    skip = _csv_set(args.skip)

    for item in suite:
        dataset = str(item["dataset"]).lower()

        if only is not None and dataset not in only:
            continue
        if skip is not None and dataset in skip:
            continue

        data_root = str(item["data_root"])
        if data_root in dcfg:
            data_root = str(dcfg[data_root])

        source_domain = str(item["source_domain"])
        target_domain = str(item["target_domain"])

        n_way = int(item.get("n_way", 2))
        k_shot = int(item.get("k_shot", 5))
        q_query = int(item.get("q_query", 10))

        extra = item.get("extra_args", []) or []
        extra = [str(x) for x in extra]

        exp_id = str(item.get("exp_id", f"SUP_{dataset.upper()}"))
        scenario = str(item.get("scenario", f"{dataset}_{source_domain}_to_{target_domain}_{n_way}way_{k_shot}shot"))

        if dataset == "bosch_mi" and source_domain.upper() == "AUTO" and target_domain.upper() == "AUTO":
            mi_src, mi_tgt, has_ft, cap_for_runs = choose_milling_domains_option_y(
                data_root,
                max_rows=int(args.milling_max_rows),
                min_per_class=int(args.milling_min_per_class),
            )
            source_domain = mi_src
            target_domain = mi_tgt
            scenario = f"BoschMilling_{mi_src}_to_{mi_tgt}_2way_5shot_dropmid"
            if not has_ft and cap_for_runs is not None:
                extra = extra + ["--max_files", str(int(cap_for_runs))]

        for seed in seeds_list:
            for method, variant in METHODS_5:
                cmd = [
                    "python", "-m", "pifsl.runner.bench_run_one",
                    "--dataset", dataset,
                    "--data_root", data_root,
                    "--method", method,
                    "--variant", variant,
                    "--source_domain", source_domain,
                    "--target_domain", target_domain,
                    "--n_way", str(n_way),
                    "--k_shot", str(k_shot),
                    "--q_query", str(q_query),
                    "--train_episodes", str(train_episodes),
                    "--eval_episodes", str(eval_episodes),
                    "--seed", str(seed),
                    "--device", device,
                    "--normalization", normalization,
                    "--results_jsonl", results_jsonl,
                    "--exp_id", exp_id,
                    "--scenario", scenario,
                ]
                cmd += extra
                run(cmd)

    if args.make_tables:
        run([
            "python", "-m", "pifsl.eval.make_tables",
            "--results_jsonl", results_jsonl,
            "--out_dir", tables_dir,
        ])


if __name__ == "__main__":
    main()
