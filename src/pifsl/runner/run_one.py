from __future__ import annotations
import argparse
from pathlib import Path

from pifsl.eval.bootstrap import add_project_roots_to_syspath
from pifsl.core.utils import set_global_seed
from pifsl.eval.schema import RunResult, append_jsonl

from pifsl.data.bosch_drilling.bundle import BoschAdapter
from pifsl.data.cwru.bundle import CWRUAdapter
from pifsl.data.hust_cn.bundle import HUSTCNAdapter
from pifsl.data.hust_vn.bundle import HUSTVNAdapter
from pifsl.data.pu.bundle import PUAdapter

from pifsl.runner.pi_fsl_relation import run_pi_fsl, PIFSLConfig
from pifsl.methods.baselines.maml_1d import run_maml, MAMLConfig

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, default=None)

    p.add_argument("--dataset", required=True, choices=["bosch","cwru","hust_cn","hust_vn","pu"])
    p.add_argument("--method", required=True, choices=["pi_fsl","maml"])
    p.add_argument("--variant", default="full", choices=["full","wo_physics"])

    p.add_argument("--data_root", required=True)
    p.add_argument("--source_domain", required=True)
    p.add_argument("--target_domain", required=True)

    p.add_argument("--tier", default="C", choices=["A","B","C"])
    p.add_argument("--n_way", type=int, default=2)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=16)
    p.add_argument("--train_episodes", type=int, default=500)
    p.add_argument("--eval_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--window_seconds", type=float, default=0.08)
    p.add_argument("--overlap_ratio", type=float, default=0.5)
    p.add_argument("--normalization", type=str, default="per_window", choices=["per_window","none"])
    p.add_argument("--input_representation", type=str, default="scalogram")
    p.add_argument("--label_mode", type=str, default="binary", choices=["binary","multiclass","simple","full"])

    p.add_argument("--hust_vn_domain_axis", type=str, default="load", choices=["load","bearing"])
    p.add_argument("--out_jsonl", type=str, default=str(Path("artifacts/results") / "results.jsonl"))
    args = p.parse_args()

    add_project_roots_to_syspath(args.project_root)
    set_global_seed(args.seed)

    # dataset load
    if args.dataset == "bosch":
        adapter = BoschAdapter()
        src, tgt = adapter.load_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            normalization=args.normalization,
        )
        scenario = "cross_domain"

    elif args.dataset == "cwru":
        adapter = CWRUAdapter()
        src, tgt = adapter.load_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            window_seconds=args.window_seconds,
            overlap_ratio=args.overlap_ratio,
            normalization=args.normalization,
            seed=args.seed,
            label_mode=args.label_mode,
        )
        scenario = "load_shift"

    elif args.dataset == "hust_cn":
        adapter = HUSTCNAdapter()
        src, tgt = adapter.load_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            window_seconds=args.window_seconds,
            overlap_ratio=args.overlap_ratio,
            normalization=args.normalization,
            label_mode=args.label_mode,
        )
        scenario = "condition_shift"

    elif args.dataset == "hust_vn":
        adapter = HUSTVNAdapter()
        src, tgt = adapter.load_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            window_seconds=args.window_seconds,
            overlap_ratio=args.overlap_ratio,
            normalization=args.normalization,
            domain_axis=args.hust_vn_domain_axis,
            label_mode=args.label_mode,
        )
        scenario = f"{args.hust_vn_domain_axis}_shift"

    else:
        adapter = PUAdapter()
        src, tgt = adapter.load_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
        )
        scenario = "pu_shift"

    # method run
    if args.method == "pi_fsl":
        cfg = PIFSLConfig(
            device=args.device,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
        )
        use_physics = (args.variant != "wo_physics")
        metrics = run_pi_fsl(src, tgt, cfg, use_physics=use_physics)

    else:
        cfg = MAMLConfig(
            device=args.device,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
        )
        metrics = run_maml(src, tgt, cfg)

    rr = RunResult(
        timestamp=RunResult.now_iso(),
        dataset=args.dataset,
        scenario=scenario,
        tier=args.tier,
        method=args.method,
        variant=args.variant,
        result_source="Own",

        source_domain=args.source_domain,
        target_domain=args.target_domain,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,

        window_seconds=args.window_seconds,
        window_samples=int(len(src.X[0])) if src.X else 0,
        overlap_ratio=args.overlap_ratio,
        normalization=args.normalization,
        input_representation=args.input_representation,
        label_mode=args.label_mode,

        acc_mean=float(metrics["acc_mean"]),
        acc_ci95=metrics.get("acc_ci95"),
        bacc_mean=float(metrics["bacc_mean"]),
        bacc_ci95=metrics.get("bacc_ci95"),
        macro_f1_mean=float(metrics["macro_f1_mean"]),
        macro_f1_ci95=metrics.get("macro_f1_ci95"),
        extra={"src_n": len(src.y), "tgt_n": len(tgt.y)},
    )

    out_path = Path(args.out_jsonl)
    append_jsonl(out_path, rr.to_dict())
    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
