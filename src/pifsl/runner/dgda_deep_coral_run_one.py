from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from pifsl.eval.ci import bootstrap
from pifsl.runner.bench_utils import set_seed
from pifsl.eval.schema import append_jsonl, utc_now_iso

from pifsl.data.bosch_drilling.loader import load_bosch_windows, Bundle
from pifsl.data.bosch_drilling.raw_processing.gating import scalogram_64x64, stftogram_64x64

from pifsl.core.fs_multiclass import sample_episode
from pifsl.core.utils import mean_ci95

from pifsl.core.models import ConvEmbedding
from pifsl.methods.baselines.m_deep_coral import (
    DeepCORALArgs,
    run_deep_coral_supervised_source,
    run_deep_coral_episode,
    evaluate_logits,
    compute_all_basic,
)

RawWindow = Union[np.ndarray, Dict[str, np.ndarray]]


def _resolve_device(device_str: str) -> str:
    if str(device_str).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _first_len(x: Any) -> int:
    if isinstance(x, dict):
        for _, v in x.items():
            return int(np.asarray(v).reshape(-1).shape[0])
        return 0
    return int(np.asarray(x).reshape(-1).shape[0])


def _parse_modalities(mods: str) -> List[str]:
    mods = (mods or "").strip()
    if not mods:
        return ["vibration"]
    out = [m.strip() for m in mods.split(",") if m.strip()]
    return out or ["vibration"]


def _get_signal_from_window(
    w: RawWindow,
    key: str,
    pad_policy: str,
    ref_len: int,
    fallback_first: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(w, dict):
        if key in w:
            arr = np.asarray(w[key], dtype=np.float32).reshape(-1)
            return arr
        if pad_policy == "duplicate_first" and fallback_first is not None:
            arr = np.asarray(fallback_first, dtype=np.float32).reshape(-1)
            return arr
        return np.zeros((ref_len,), dtype=np.float32)

    arr0 = np.asarray(w, dtype=np.float32).reshape(-1)
    if key == "vibration":
        return arr0
    if pad_policy == "duplicate_first":
        return arr0
    return np.zeros((ref_len,), dtype=np.float32)


def _windows_to_scalograms(
    windows: List[RawWindow],
    fs: float,
    modalities: List[str],
    pad_missing_modalities: str,
    tf_rep: str = "cwt",
) -> torch.Tensor:
    if not windows:
        raise ValueError("No windows provided for scalogram conversion")

    ref_len = _first_len(windows[0])
    if ref_len <= 0:
        ref_len = 1024

    batch: List[torch.Tensor] = []
    for w in windows:
        fallback_first = None
        if isinstance(w, dict) and len(w) > 0:
            fallback_first = np.asarray(next(iter(w.values())), dtype=np.float32).reshape(-1)

        chans: List[torch.Tensor] = []
        for m in modalities:
            sig = _get_signal_from_window(
                w=w,
                key=m,
                pad_policy=pad_missing_modalities,
                ref_len=ref_len,
                fallback_first=fallback_first,
            )
            if str(tf_rep).lower() == "stft":
                S = stftogram_64x64(sig, float(fs))
            else:
                S = scalogram_64x64(sig, float(fs))
            chans.append(torch.from_numpy(S).float())
        x = torch.stack(chans, dim=0)  # [C,64,64]
        batch.append(x)

    return torch.stack(batch, dim=0)  # [B,C,64,64]


def _maybe_extra_metrics_and_save(
    base_metrics: Dict[str, Any],
    y_true_all: List[int],
    y_pred_all: List[int],
    y_proba_all: Optional[List[float]],
    args,
    tag: str,
) -> Dict[str, Any]:
    out = dict(base_metrics)
    extra: Dict[str, Any] = {}
    y_true_np = np.asarray(y_true_all, dtype=int) if len(y_true_all) else None
    y_pred_np = np.asarray(y_pred_all, dtype=int) if len(y_pred_all) else None
    y_proba_np = None if y_proba_all is None else np.asarray(y_proba_all, dtype=float)

    if y_true_np is not None and y_pred_np is not None:
        try:
            extra["confusion_matrix"] = confusion_matrix(y_true_np, y_pred_np).tolist()
        except Exception:
            pass

    if getattr(args, "comprehensive_metrics", False) and y_proba_np is not None and y_true_np is not None:
        # meaningful for binary labels only
        if set(np.unique(y_true_np)).issubset({0, 1}) and len(np.unique(y_true_np)) == 2:
            try:
                extra["roc_auc"] = float(roc_auc_score(y_true_np, y_proba_np))
            except Exception:
                pass
            try:
                extra["pr_auc"] = float(average_precision_score(y_true_np, y_proba_np))
            except Exception:
                pass

    # Save artifacts for plotting if requested
    save_dir = str(getattr(args, "save_eval_artifacts_dir", "") or "").strip()
    if save_dir and y_true_np is not None and y_pred_np is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fn = (
            f"eval_{str(getattr(args,'dataset',''))}_"
            f"{str(getattr(args,'scenario',''))}_"
            f"{str(getattr(args,'method',''))}_{tag}_seed{int(getattr(args,'seed',0))}.npz"
        )
        np.savez_compressed(
            str(Path(save_dir) / fn),
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_proba=y_proba_np if y_proba_np is not None else np.array([], dtype=float),
        )
        extra["eval_artifacts_npz"] = fn

    if extra:
        out["extra"] = out.get("extra", {}) or {}
        out["extra"].update(extra)
    return out


def parse_args():
    p = argparse.ArgumentParser("Run one DGDA DeepCORAL job (Bosch drilling) and append to results.jsonl")
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--dataset", type=str, default="bosch", choices=["bosch"])
    p.add_argument("--data_root", type=str, required=True)

    p.add_argument("--scenario", type=str, default="")
    p.add_argument("--exp_id", type=str, default="")
    p.add_argument("--result_source", type=str, default="run_one")
    p.add_argument("--notes", type=str, default="")

    # representation
    p.add_argument("--tf_rep", type=str, default="cwt", choices=["cwt", "stft"])
    p.add_argument("--modalities", type=str, default="vibration")
    p.add_argument("--pad_missing_modalities", type=str, default="zeros", choices=["zeros", "duplicate_first"])

    # protocol
    p.add_argument("--source_domain", type=str, required=True)
    p.add_argument("--target_domain", type=str, required=True)
    p.add_argument("--normalization", type=str, default="per_window", choices=["per_window", "none", "zscore"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_way", type=int, default=2)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--device", type=str, default="auto")

    # episodes
    p.add_argument("--train_episodes", type=int, default=0)  # kept for schema compatibility
    p.add_argument("--eval_episodes", type=int, default=200)

    # supervised pretrain (source)
    p.add_argument("--sup_epochs", type=int, default=10)
    p.add_argument("--sup_batch_size", type=int, default=32)
    p.add_argument("--sup_lr", type=float, default=1e-3)

    # CORAL adaptation knobs
    p.add_argument("--coral_steps", type=int, default=200)
    p.add_argument("--coral_weight", type=float, default=1.0)
    p.add_argument("--coral_lr", type=float, default=5e-4)
    p.add_argument("--coral_batch_size", type=int, default=64)
    p.add_argument(
        "--coral_use_target_support_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, CORAL uses only target SUPPORT windows as unlabeled target data.",
    )

    # outputs
    p.add_argument("--comprehensive_metrics", action="store_true")
    p.add_argument("--save_eval_artifacts_dir", type=str, default="")
    p.add_argument("--time_profile", action="store_true")
    p.add_argument("--results_jsonl", type=str, default="artifacts/results/jsonl/dgda_deep_coral.jsonl")
    p.add_argument("--out_jsonl", type=str, default="")  # alias

    return p.parse_args()


def _load_pair(args) -> Tuple[Bundle, Bundle, float, int, float]:
    # Support multi-source domains given as comma-separated values, e.g.
    #   "M01_OP07,M01_OP05" (used by Bosch E4 in this repo).
    src_domains = [d.strip() for d in str(args.source_domain).split(",") if d.strip()]
    if not src_domains:
        src_domains = [str(args.source_domain).strip()]

    src_all_X, src_all_y, src_all_dom, src_all_fid = [], [], [], []
    tgt = None
    for sd in src_domains:
        src_i, tgt_i = load_bosch_windows(
            data_root=str(args.data_root),
            source_domain=str(sd),
            target_domain=str(args.target_domain),
            normalization=str(args.normalization),
        )
        src_all_X.extend(list(src_i.X))
        src_all_y.extend(list(src_i.y))
        src_all_dom.extend(list(src_i.domain))
        src_all_fid.extend(list(src_i.file_id))
        if tgt is None:
            tgt = tgt_i

    assert tgt is not None
    src = Bundle(X=src_all_X, y=src_all_y, domain=src_all_dom, file_id=src_all_fid, fs=float(tgt.fs))
    fs = float(src.fs)
    window_samples = int(np.asarray(src.X[0]).reshape(-1).shape[0]) if src.X else 0
    window_seconds = float(window_samples / fs) if (fs > 0 and window_samples > 0) else 0.0
    return src, tgt, fs, window_samples, window_seconds


def _run_deep_coral(src: Bundle, tgt: Bundle, fs: float, args) -> Dict[str, Any]:
    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)

    # Convert all labeled source windows once
    src_x = _windows_to_scalograms(
        src.X, fs=fs,
        modalities=modalities,
        pad_missing_modalities=args.pad_missing_modalities,
        tf_rep=args.tf_rep,
    ).to(device)
    src_y = torch.as_tensor(np.asarray(src.y, dtype=np.int64), device=device).long()

    # Model
    enc = ConvEmbedding(in_channels=len(modalities)).to(device)
    with torch.no_grad():
        dummy = torch.zeros((1, len(modalities), 64, 64), device=device)
        feat_dim = int(enc(dummy).reshape(1, -1).size(1))
    head = nn.Linear(feat_dim, int(args.n_way)).to(device)

    dc_args = DeepCORALArgs(
        device=str(device),
        sup_epochs=int(args.sup_epochs),
        sup_batch_size=int(args.sup_batch_size),
        sup_lr=float(args.sup_lr),
        coral_steps=int(args.coral_steps),
        coral_batch_size=int(args.coral_batch_size),
        coral_lr=float(args.coral_lr),
        coral_weight=float(args.coral_weight),
        coral_use_target_support_only=bool(args.coral_use_target_support_only),
    )

    # Supervised pretrain on full source
    t0 = time.time()
    run_deep_coral_supervised_source(enc, head, src_x, src_y, dc_args)
    t_sup = time.time() - t0

    # Snapshot weights so each target episode starts from same pretrained model
    base_enc = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    base_head = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    accs: List[float] = []
    baccs: List[float] = []
    f1s: List[float] = []

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_proba_all: Optional[List[float]] = []  # only for binary head

    ep_times: List[float] = []

    for ep in range(int(args.eval_episodes)):
        ep0 = time.time()

        enc.load_state_dict(base_enc, strict=True)
        head.load_state_dict(base_head, strict=True)
        enc.to(device); head.to(device)

        # Target episode sampling
        sup_X, sup_y, qry_X, qry_y = sample_episode(
            tgt.X, tgt.y,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=int(args.seed) * 1000 + ep,
        )

        # unlabeled target for CORAL
        tgt_unl = sup_X if bool(args.coral_use_target_support_only) else tgt.X
        tgt_unl_x = _windows_to_scalograms(
            tgt_unl, fs=fs,
            modalities=modalities,
            pad_missing_modalities=args.pad_missing_modalities,
            tf_rep=args.tf_rep,
        ).to(device)

        # Adaptation
        run_deep_coral_episode(enc, head, src_x, src_y, tgt_unl_x, dc_args)

        # Evaluate on query
        qry_x = _windows_to_scalograms(
            qry_X, fs=fs,
            modalities=modalities,
            pad_missing_modalities=args.pad_missing_modalities,
            tf_rep=args.tf_rep,
        ).to(device)

        y_true, y_pred, y_proba = evaluate_logits(
            enc, head, qry_x,
            np.asarray(qry_y, dtype=np.int64),
            str(device),
        )

        y_true_all.extend([int(v) for v in y_true.tolist()])
        y_pred_all.extend([int(v) for v in y_pred.tolist()])
        if y_proba is not None:
            if y_proba_all is not None:
                y_proba_all.extend([float(v) for v in y_proba.tolist()])
        else:
            y_proba_all = None

        m = compute_all_basic(y_true, y_pred)
        accs.append(float(m["acc"]))
        baccs.append(float(m["bacc"]))
        f1s.append(float(m["macro_f1"]))

        ep_times.append(time.time() - ep0)

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)

    metrics: Dict[str, Any] = {
        "acc_mean": float(acc_m),
        "acc_ci95": acc_ci,
        "bacc_mean": float(bacc_m),
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": float(f1_m),
        "macro_f1_ci95": f1_ci,
    }

    if getattr(args, "time_profile", False):
        metrics["extra"] = metrics.get("extra", {}) or {}
        metrics["extra"]["time_sup_seconds"] = float(t_sup)
        metrics["extra"]["time_eval_mean_seconds"] = float(np.mean(ep_times)) if ep_times else None
        metrics["extra"]["time_eval_total_seconds"] = float(np.sum(ep_times)) if ep_times else None

    metrics = _maybe_extra_metrics_and_save(
        metrics, y_true_all, y_pred_all, y_proba_all, args, tag="deep_coral"
    )
    return metrics


def main():
    args = parse_args()

    # allow --out_jsonl as alias
    if args.out_jsonl:
        args.results_jsonl = args.out_jsonl

    project_root = Path(args.project_root).resolve()
    bootstrap(project_root)
    set_seed(int(args.seed))

    src, tgt, fs, window_samples, window_seconds = _load_pair(args)

    # run
    args.method = "deep_coral"
    args.variant = getattr(args, "variant", "full") or "full"
    metrics = _run_deep_coral(src, tgt, fs, args)

    scenario = str(args.scenario).strip()
    if not scenario:
        scenario = f"{args.source_domain}_to_{args.target_domain}"

    input_rep = "scalogram_64x64"

    record: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "dataset": str(args.dataset),
        "scenario": scenario,
        "method": "deep_coral",
        "variant": str(getattr(args, "variant", "full")),
        "result_source": str(args.result_source),
        "source_domain": str(args.source_domain),
        "target_domain": str(args.target_domain),
        "n_way": int(args.n_way),
        "k_shot": int(args.k_shot),
        "q_query": int(args.q_query),
        "train_episodes": int(args.train_episodes),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "fs": float(fs),
        "window_samples": int(window_samples),
        "window_seconds": float(window_seconds),
        "overlap_ratio": 0.0,
        "normalization": str(args.normalization),
        "input_representation": input_rep,
        "acc_mean": float(metrics.get("acc_mean", float("nan"))),
        "acc_ci95": metrics.get("acc_ci95", None),
        "bacc_mean": float(metrics.get("bacc_mean", float("nan"))),
        "bacc_ci95": metrics.get("bacc_ci95", None),
        "macro_f1_mean": float(metrics.get("macro_f1_mean", float("nan"))),
        "macro_f1_ci95": metrics.get("macro_f1_ci95", None),
        "notes": str(args.notes),
        "extra": {
            "exp_id": str(args.exp_id),
            "modalities": _parse_modalities(args.modalities),
            "pad_missing_modalities": str(args.pad_missing_modalities),
            "device_resolved": _resolve_device(args.device),
            "tf_rep": str(args.tf_rep),
            "sup_epochs": int(args.sup_epochs),
            "sup_batch_size": int(args.sup_batch_size),
            "sup_lr": float(args.sup_lr),
            "coral_steps": int(args.coral_steps),
            "coral_batch_size": int(args.coral_batch_size),
            "coral_lr": float(args.coral_lr),
            "coral_weight": float(args.coral_weight),
            "coral_use_target_support_only": bool(args.coral_use_target_support_only),
        },
    }

    if isinstance(metrics.get("extra", None), dict):
        record["extra"].update(metrics["extra"])

    out_fp = project_root / str(args.results_jsonl)
    append_jsonl(out_fp, record)

    print(f"[OK] wrote {out_fp}")
    print(json.dumps(
        {k: record[k] for k in ["dataset", "scenario", "method", "variant", "acc_mean", "bacc_mean", "macro_f1_mean"]},
        indent=2
    ))


if __name__ == "__main__":
    main()
