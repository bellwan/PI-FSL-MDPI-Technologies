from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_with_src(repo_root: Path) -> dict:
    env = os.environ.copy()
    src = str((repo_root / "src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + ((";" + prev) if prev else "")
    return env


def _run(cmd: List[str]) -> None:
    repo_root = _repo_root()
    if cmd and str(cmd[0]).lower() == "python":
        cmd = [sys.executable] + cmd[1:]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=_env_with_src(repo_root))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                # ignore malformed lines
                continue
    return out


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _key(dataset: str, method: str, variant: str, seed: int) -> Tuple[str, str, str, int]:
    return (dataset.lower(), method.lower(), variant.lower(), int(seed))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--runner_module", default="pifsl.runner.bench_run_one")

    ap.add_argument("--out_jsonl", default="artifacts/results/jsonl/table12_ablation.jsonl")

    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--train_episodes", type=int, default=200)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--tf_rep", default="cwt", choices=["cwt", "stft"])
    ap.add_argument("--device", default="auto")

    ap.add_argument("--save_eval_artifacts_dir", default="outputs/eval_artifacts")

    args = ap.parse_args()

    root = _repo_root()
    cfg = _load_yaml(root / args.paths)

    suite = cfg.get("phase_c", {}).get("suite", [])
    pu_entries = [e for e in suite if str(e.get("dataset", "")).lower() == "pu"]
    if not pu_entries:
        raise SystemExit("configs/paths.yaml has no phase_c.suite entry for dataset=pu")
    pu = pu_entries[0]

    # Dataset root indirection via paths.yaml datasets.<key>
    data_root_key = str(pu["data_root"]).strip()
    data_root = cfg.get("datasets", {}).get(data_root_key)
    if not data_root:
        raise KeyError(f"configs/paths.yaml missing datasets.{data_root_key}")

    source_domain = str(pu.get("source_domain", "")).strip()
    target_domain = str(pu.get("target_domain", "")).strip()
    q_query = int(pu.get("q_query", 10))
    extra_args = list(pu.get("extra_args", []))

    exp_id = str(pu.get("exp_id", "SUP_PU_SHIFT")).strip()
    scenario_base = str(pu.get("scenario", "PU_setting_shift_2way_5shot")).strip()

    out_jsonl_path = root / args.out_jsonl
    existing = _load_jsonl(out_jsonl_path)
    existing_keys = {
        _key(r.get("dataset", ""), r.get("method", ""), r.get("variant", ""), r.get("seed", -999))
        for r in existing
    }

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    method = "pi_fsl"

    # Missing PI-FSL variants required for Table 12a (PU only).
    want_variants = ["wo_physics", "energy_only", "spectral_only", "envelope_only"]

    tmp_dir = root / "outputs" / "tmp_runs_pu_missing"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ran_any = False
    for variant in want_variants:
        for seed in seeds:
            if _key("pu", method, variant, seed) in existing_keys:
                print(f"[SKIP] already present: dataset=pu method={method} variant={variant} seed={seed}")
                continue

            # Runs into a temp JSONL, then appends a single normalized record to out_jsonl.
            tmp_jsonl = tmp_dir / f"pu_{variant}_seed{seed}.jsonl"
            if tmp_jsonl.exists():
                tmp_jsonl.unlink()

            effective_variant = variant
            physics_weight = None
            notes_variant = variant
            notes_extra = ""

            # wo_physics is Bosch-only in the runner; emulate by disabling physics via physics_weight=0.0.
            if variant == "wo_physics":
                effective_variant = "full"
                physics_weight = 0.0
                notes_extra = "emulated_wo_physics: ran variant=full with physics_weight=0.0"

            scenario = f"{scenario_base}__abl_{notes_variant}"

            cmd = [
                "python", "-m", args.runner_module,
                "--dataset", "pu",
                "--data_root", str(data_root),
                "--method", method,
                "--variant", effective_variant,
                "--source_domain", source_domain,
                "--target_domain", target_domain,
                "--n_way", "2",
                "--k_shot", "5",
                "--q_query", str(q_query),
                "--train_episodes", str(args.train_episodes),
                "--eval_episodes", str(args.eval_episodes),
                "--seed", str(seed),
                "--tf_rep", args.tf_rep,
                "--device", args.device,
                "--results_jsonl", str(tmp_jsonl),
                "--exp_id", exp_id,
                "--scenario", scenario,
                "--comprehensive_metrics",
                "--save_eval_artifacts_dir", args.save_eval_artifacts_dir,
                "--notes", f"table12_ablation;dataset=pu;variant={notes_variant};bal=none;{notes_extra}".strip(";"),
            ] + extra_args

            cmd += ["--transfer_finetune", "--target_adapt_ratio", "0.40"]

            if physics_weight is not None:
                cmd += ["--physics_weight", str(physics_weight)]

            _run(cmd)

            tmp_rows = _load_jsonl(tmp_jsonl)
            if not tmp_rows:
                raise RuntimeError(f"Runner produced no JSONL output: {tmp_jsonl}")

            rec = tmp_rows[-1]

            # Normalizes output variant name for downstream table grouping when emulating wo_physics.
            if variant == "wo_physics":
                rec["variant"] = "wo_physics"
                rec["notes"] = str(rec.get("notes", "")) + ";emulated_variant=wo_physics"

            _append_jsonl(out_jsonl_path, rec)
            existing_keys.add(_key("pu", method, variant, seed))
            ran_any = True
            print(f"[APPEND] dataset=pu method={method} variant={variant} seed={seed} -> {out_jsonl_path}")

    if not ran_any:
        print("Nothing to run: all PU missing variants/seeds are already present in out_jsonl.")


if __name__ == "__main__":
    main()
