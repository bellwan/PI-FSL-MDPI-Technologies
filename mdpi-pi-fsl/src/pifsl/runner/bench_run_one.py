from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json as _json

from pifsl.eval.ci import bootstrap
from pifsl.runner.bench_utils import set_seed
from pifsl.eval.schema import append_jsonl, utc_now_iso

from pifsl.data.bosch_drilling.loader import load_bosch_windows, Bundle
from pifsl.data.cwru.loader import load_cwru_windows
from pifsl.data.hust_cn.loader import load_hust_cn_windows
from pifsl.data.hust_vn.loader import load_hust_vn_windows
from pifsl.data.pu.loader import load_pu_source
from pifsl.data.bosch_milling.loader import load_bosch_mi_source, load_scidata2025_source
from pifsl.runner.config_loader import PhysicsRegularizationConfig
from pifsl.core.physics_regularization import PhysicsInformedRegularizer
from pifsl.core.models import RelationNet, ConvEmbedding
from pifsl.data.bosch_drilling.raw_processing.gating import scalogram_64x64

from pifsl.core.fs_multiclass import (
    sample_episode_disjoint_by_file,
    sample_episode_with_indices_disjoint_by_file,
    sample_episode,
    sample_episode_with_indices,
)
from pifsl.core.utils import mean_ci95, compute_metrics
from pifsl.methods.baselines.m_maml import run_maml, MAMLArgs
from pifsl.core.sampling.psd_sampling import PSDGuidedSampler

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


def _concat_bundles(bundles: List[Bundle], name: str = "src") -> Bundle:
    if not bundles:
        raise ValueError(f"Empty bundle list for {name}")
    fs0 = float(bundles[0].fs)
    X: List[RawWindow] = []
    y: List[int] = []
    dom: List[str] = []
    fid: List[str] = []
    for b in bundles:
        if float(b.fs) != fs0:
            raise ValueError(f"FS mismatch while merging {name}: {fs0} vs {float(b.fs)}")
        X.extend(b.X)
        y.extend(b.y)
        dom.extend(b.domain)
        fid.extend(b.file_id)
    return Bundle(X=X, y=y, domain=dom, file_id=fid, fs=fs0) 


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
        # missing modality
        if pad_policy == "duplicate_first" and fallback_first is not None:
            arr = np.asarray(fallback_first, dtype=np.float32).reshape(-1)
            return arr
        return np.zeros((ref_len,), dtype=np.float32)

    # ndarray window
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
) -> "torch.Tensor":

    if not windows:
        raise ValueError("No windows provided for scalogram conversion")

    ref_len = _first_len(windows[0])
    if ref_len <= 0:
        ref_len = 1024

    batch = []
    for w in windows:
        fallback_first = None
        if isinstance(w, dict) and len(w) > 0:
            fallback_first = np.asarray(next(iter(w.values())), dtype=np.float32).reshape(-1)

        chans = []
        for m in modalities:
            sig = _get_signal_from_window(
                w=w,
                key=m,
                pad_policy=pad_missing_modalities,
                ref_len=ref_len,
                fallback_first=fallback_first,
            )
            S = scalogram_64x64(sig, float(fs))  # [64,64]
            chans.append(torch.from_numpy(S).float())
        x = torch.stack(chans, dim=0)  # [C,64,64]
        batch.append(x)

    return torch.stack(batch, dim=0)  # [B,C,64,64]


def _split_by_file_id(file_ids: List[str], seed: int, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(file_ids))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_test = max(1, int(math.ceil(len(uniq) * float(test_ratio))))
    test_set = set(uniq[:n_test])
    idx = np.arange(len(file_ids))
    test_idx = np.asarray([i for i, f in enumerate(file_ids) if f in test_set], dtype=np.int64)
    train_idx = np.asarray([i for i, f in enumerate(file_ids) if f not in test_set], dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        rng2 = np.random.RandomState(int(seed))
        perm = rng2.permutation(idx)
        cut = max(1, int(len(idx) * (1.0 - float(test_ratio))))
        train_idx = perm[:cut]
        test_idx = perm[cut:]
    return train_idx, test_idx

def _sample_episode_bundle(bundle: Bundle, n_way: int, k_shot: int, q_query: int, seed: int, args, with_indices: bool = False):
    ds = str(getattr(args, "dataset", "")).lower()
    if ds == "cwru":

        if with_indices:
            return sample_episode_with_indices_disjoint_by_file(
                bundle.X, bundle.y, bundle.file_id,
                n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed
            )
        return sample_episode_disjoint_by_file(
            bundle.X, bundle.y, bundle.file_id,
            n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed
        )

    # default behavior for other datasets

    if with_indices:
        return sample_episode_with_indices(
            bundle.X, bundle.y, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed
        )
    return sample_episode(
        bundle.X, bundle.y, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed
    )

# -----------------------------
def _load_single_domain(dataset: str, data_root: str, domain: str, normalization: str, seed: int, args=None, max_files: Optional[int] = None) -> Bundle:
    dataset = dataset.lower()

    if dataset == "bosch":
        doms = [d.strip() for d in (domain or "").split(",") if d.strip()]
        if len(doms) <= 1:
            d = doms[0] if doms else domain
            b, _ = load_bosch_windows(
                data_root=data_root,
                source_domain=d,
                target_domain=d,
                normalization=normalization,
            )
            return b
        bundles: List[Bundle] = []
        for d in doms:
            b, _ = load_bosch_windows(
                data_root=data_root,
                source_domain=d,
                target_domain=d,
                normalization=normalization,
            )
            bundles.append(b)
        return _concat_bundles(bundles, name="src_multi_bosch")

    if dataset == "cwru":
        b, _ = load_cwru_windows(
            data_root=data_root,
            source_domain=domain,
            target_domain=domain,
            time_steps=1024,
            overlap_ratio=0.5,
            normalization=normalization,
            seed=seed,
            label_mode="binary",
        )
        return b

    if dataset == "hust_cn":
        b, _ = load_hust_cn_windows(
            data_root=data_root,
            source_domain=domain,
            target_domain=domain,
            normalization=normalization,
            seed=seed,
            label_mode="binary",
        )
        return b

    if dataset == "hust_vn":
        b, _ = load_hust_vn_windows(
            data_root=data_root,
            source_domain=domain,
            target_domain=domain,
            normalization=normalization,
            seed=seed,
            label_mode="full",
            domain_axis="load",
        )
        return b

    if dataset in ("bosch_mi", "bosch_milling", "scidata2025"):
        lm = str(getattr(args, "bosch_mi_label_mode", "3class")) if args is not None else "3class"
        prefer_ft = bool(getattr(args, "bosch_mi_prefer_feature_table", True)) if args is not None else True
        inc = getattr(args, "bosch_mi_include_domains", None) if args is not None else None
        inc_list = None
        if isinstance(inc, str) and inc.strip():
            inc_list = [s.strip() for s in inc.split(",") if s.strip()]

        norm = normalization
        if str(norm).lower() in ("per_window",):
            norm = "zscore"

        return load_bosch_mi_source(
            data_root=data_root,
            normalization=norm,
            seed=int(seed),
            max_files=int(max_files) if max_files is not None else None,
            label_mode=lm,
            prefer_feature_table=prefer_ft,
            include_domains=inc_list,
        )

    if dataset == "pu":
        ck = str(getattr(args, "current_key", "motor_current")) if args is not None else "motor_current"
        inc_codes = getattr(args, "pu_include_codes", None) if args is not None else None
        inc_settings = getattr(args, "pu_include_settings", None) if args is not None else None
        max_wpf = getattr(args, "pu_max_windows_per_file", None) if args is not None else None

        codes_list = [s.strip() for s in inc_codes.split(",") if s.strip()] if isinstance(inc_codes, str) and inc_codes.strip() else None
        settings_list = [s.strip() for s in inc_settings.split(",") if s.strip()] if isinstance(inc_settings, str) and inc_settings.strip() else None

        return load_pu_source(
            data_root=data_root,
            normalization=normalization,
            current_key=ck,
            max_files=int(max_files) if max_files is not None else None,
            include_codes=codes_list,
            include_settings=settings_list,
            max_windows_per_file=(int(max_wpf) if max_wpf is not None else None),
        )

    raise ValueError(f"Unknown dataset: {dataset}")


def _load_pair(args) -> Tuple[Bundle, Bundle, float, int, float]:
    # Cross-dataset mode
    if args.src_dataset and args.tgt_dataset:
        src_dataset = args.src_dataset.lower()
        tgt_dataset = args.tgt_dataset.lower()

        if args.src_data_root is None or args.tgt_data_root is None:
            raise ValueError("Cross-dataset mode requires --src_data_root and --tgt_data_root")

        src_domain = args.source_domain
        tgt_domain = args.target_domain

        _src_max_files = int(args.src_max_files) if getattr(args, "src_max_files", None) is not None else (int(args.max_files) if getattr(args, "max_files", None) is not None else None)
        _tgt_max_files = int(args.tgt_max_files) if getattr(args, "tgt_max_files", None) is not None else (int(args.max_files) if getattr(args, "max_files", None) is not None else None)

        if src_dataset in ("pu", "scidata2025"):
            src = _load_single_domain(src_dataset, args.src_data_root, "ALL", args.normalization, args.seed, args=args, max_files=_src_max_files)
        else:
            src = _load_single_domain(src_dataset, args.src_data_root, src_domain, args.normalization, args.seed, args=args, max_files=_src_max_files)

        if tgt_dataset in ("pu", "scidata2025"):
            tgt = _load_single_domain(tgt_dataset, args.tgt_data_root, "ALL", args.normalization, args.seed, args=args, max_files=_tgt_max_files)
        else:
            tgt = _load_single_domain(tgt_dataset, args.tgt_data_root, tgt_domain, args.normalization, args.seed, args=args, max_files=_tgt_max_files)

        fs = float(getattr(tgt, 'fs', 0.0) or 0.0)
        if not np.isfinite(fs) or fs <= 0:
            fs = float(getattr(src, 'fs', 0.0) or 0.0)
        if not np.isfinite(fs) or fs <= 0:
            fs = 1.0
        window_samples = _first_len(src.X[0])
        window_seconds = float(window_samples) / float(fs) if fs > 0 else float("nan")
        return src, tgt, fs, window_samples, window_seconds

    # Single dataset mode
    if args.dataset and not args.data_root:
        args.data_root = r"D:\jb\datasets"

    if not ((args.dataset and args.data_root) or (args.src_dataset and args.tgt_dataset)):
        raise ValueError("Provide either --dataset/--data_root OR --src_dataset/--tgt_dataset + roots")

    ds = args.dataset.lower()
    if ds == "bosch":
        src_doms = [d.strip() for d in (args.source_domain or "").split(",") if d.strip()]
        if len(src_doms) <= 1:
            src, tgt = load_bosch_windows(
                data_root=args.data_root,
                source_domain=args.source_domain,
                target_domain=args.target_domain,
                normalization=args.normalization,
            )
        else:
            src_parts: List[Bundle] = []
            tgt_keep: Optional[Bundle] = None
            for d in src_doms:
                s, t = load_bosch_windows(
                    data_root=args.data_root,
                    source_domain=d,
                    target_domain=args.target_domain,
                    normalization=args.normalization,
                )
                src_parts.append(s)
                if tgt_keep is None:
                    tgt_keep = t
            src = _concat_bundles(src_parts, name="src_multi_bosch")
            tgt = tgt_keep if tgt_keep is not None else src_parts[0]
        fs = float(src.fs)

    elif ds == "cwru":
        lm = args.label_mode
        if lm is None and args.cwru_label_mode:
            lm = "binary" if args.cwru_label_mode == "fault" else "multiclass"
        lm = lm or "binary"

        src, tgt = load_cwru_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            time_steps=int(args.cwru_time_steps),
            overlap_ratio=float(args.overlap_ratio),
            normalization=args.normalization,
            seed=int(args.seed),
            label_mode=lm,
            pos_label=(int(args.cwru_pos_label) if getattr(args, "cwru_pos_label", None) is not None else None),
        )
        fs = float(src.fs)

    elif ds == "hust_cn":
        lm = args.label_mode or "binary"
        src, tgt = load_hust_cn_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            normalization=args.normalization,
            seed=int(args.seed),
            label_mode=lm,
        )
        fs = float(src.fs)

    elif ds == "hust_vn":
        lm = args.label_mode or "full"
        src, tgt = load_hust_vn_windows(
            data_root=args.data_root,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            normalization=args.normalization,
            seed=int(args.seed),
            label_mode=lm,
            domain_axis=args.hust_vn_domain_axis,
        )
        fs = float(src.fs)

    elif ds in ("bosch_mi", "bosch_milling", "scidata2025"):
        lm = str(getattr(args, "bosch_mi_label_mode", "3class"))
        prefer_ft = bool(getattr(args, "bosch_mi_prefer_feature_table", True))
        inc = getattr(args, "bosch_mi_include_domains", None)
        inc_list = [s.strip() for s in inc.split(",") if s.strip()] if isinstance(inc, str) and inc.strip() else None

        norm = args.normalization
        if str(norm).lower() in ("per_window",):
            norm = "zscore"

        def _split_csv(s: str | None):
            if s is None:
                return None
            s2 = str(s).strip()
            if not s2 or s2.upper() in ("ALL", "SAME"):
                return None
            return [x.strip() for x in s2.split(",") if x.strip()]

        # Treat source_domain/target_domain as tool/run domain ids when provided
        src_domains = _split_csv(getattr(args, "source_domain", None))
        tgt_domains = _split_csv(getattr(args, "target_domain", None))

        if src_domains or tgt_domains:
            src = load_bosch_mi_source(
                data_root=args.data_root,
                normalization=norm,
                seed=int(args.seed),
                max_files=int(args.max_files) if getattr(args, "max_files", None) is not None else None,
                label_mode=lm,
                prefer_feature_table=prefer_ft,
                include_domains=(src_domains or inc_list),
            )
            tgt = load_bosch_mi_source(
                data_root=args.data_root,
                normalization=norm,
                seed=int(args.seed),
                max_files=int(args.max_files) if getattr(args, "max_files", None) is not None else None,
                label_mode=lm,
                prefer_feature_table=prefer_ft,
                include_domains=(tgt_domains or src_domains or inc_list),
            )
        else:
            src = load_bosch_mi_source(
                data_root=args.data_root,
                normalization=norm,
                seed=int(args.seed),
                max_files=int(args.max_files) if getattr(args, "max_files", None) is not None else None,
                label_mode=lm,
                prefer_feature_table=prefer_ft,
                include_domains=inc_list,
            )
            tgt = src

        # Option-Y analogue: drop mid-life class (1) and binarize {0,2}->{0,1}
        if bool(getattr(args, "bosch_mi_drop_mid", False)):
            y = np.asarray(src.y)
            keep = (y != 1)
            src = Bundle(
                X=src.X[keep],
                y=np.where(y[keep] == 2, 1, y[keep]),
                domain=src.domain[keep],
                file_id=src.file_id[keep],
                fs=src.fs,
            )

            y2 = np.asarray(tgt.y)
            keep2 = (y2 != 1)
            tgt = Bundle(
                X=tgt.X[keep2],
                y=np.where(y2[keep2] == 2, 1, y2[keep2]),
                domain=tgt.domain[keep2],
                file_id=tgt.file_id[keep2],
                fs=tgt.fs,
            )

        fs = float(src.fs)

    elif ds == "pu":
        ck = str(getattr(args, "current_key", "motor_current"))
        max_wpf = getattr(args, "pu_max_windows_per_file", None)

        # Option-Y (2-way): healthy vs a single fault code
        pu_healthy = getattr(args, "pu_healthy_code", None)
        pu_pos = getattr(args, "pu_pos_code", None)

        inc_codes = getattr(args, "pu_include_codes", None)
        inc_settings = getattr(args, "pu_include_settings", None)

        def _split_csv(s: str | None):
            if s is None:
                return None
            s2 = str(s).strip()
            if not s2 or s2.upper() in ("ALL", "SAME"):
                return None
            return [x.strip() for x in s2.split(",") if x.strip()]

        # Use source_domain/target_domain as PU settings (Nxx_Mxx_Fxx) when provided
        src_settings = _split_csv(getattr(args, "source_domain", None))
        tgt_settings = _split_csv(getattr(args, "target_domain", None))

        codes_list = [s.strip() for s in inc_codes.split(",") if s.strip()] if isinstance(inc_codes, str) and inc_codes.strip() else None
        settings_list = [s.strip() for s in inc_settings.split(",") if s.strip()] if isinstance(inc_settings, str) and inc_settings.strip() else None

        mapping = None

        if pu_healthy and pu_pos:
            if ("," not in str(pu_healthy)) and ("," not in str(pu_pos)):
                codes_list = [str(pu_healthy).strip(), str(pu_pos).strip()]
            else:
                codes_list = None

        # Cross-setting split if source_domain/target_domain provided
        if src_settings or tgt_settings:
            src = load_pu_source(
                data_root=args.data_root,
                normalization=args.normalization,
                current_key=ck,
                max_files=int(args.max_files) if getattr(args, "max_files", None) is not None else None,
                include_codes=codes_list,
                include_settings=(src_settings or settings_list),
                max_windows_per_file=(int(max_wpf) if max_wpf is not None else None),
                mapping=mapping,
                healthy_code=pu_healthy,
                pos_code=pu_pos,
                window=int(args.pu_window_samples) if getattr(args, "pu_window_samples", None) else 1000,
                hop=int(args.pu_hop_samples) if getattr(args, "pu_hop_samples", None) else 500,

            )
            tgt = load_pu_source(
                data_root=args.data_root,
                normalization=args.normalization,
                current_key=ck,
                max_files=int(args.max_files) if getattr(args, "max_files", None) is not None else None,
                include_codes=codes_list,
                include_settings=(tgt_settings or src_settings or settings_list),
                max_windows_per_file=(int(max_wpf) if max_wpf is not None else None),
                mapping=mapping,
                healthy_code=pu_healthy,
                pos_code=pu_pos,
                window=int(args.pu_window_samples) if getattr(args, "pu_window_samples", None) else 1000,
                hop=int(args.pu_hop_samples) if getattr(args, "pu_hop_samples", None) else 500,

            )
        else:
            src = load_pu_source(
                data_root=args.data_root,
                normalization=args.normalization,
                current_key=ck,
                max_files=int(args.max_files) if getattr(args, "max_files", None) is not None else None,
                include_codes=codes_list,
                include_settings=settings_list,
                max_windows_per_file=(int(max_wpf) if max_wpf is not None else None),
                mapping=mapping,
            )
            tgt = src

        # Safety: ensure both classes exist in both splits in Option-Y
        if (pu_healthy and pu_pos) and getattr(args, "label_mode", None) in (None, "binary"):
            for name, b in ("src", src), ("tgt", tgt):
                u = set(int(v) for v in np.asarray(b.y).tolist())
                if u != {0, 1}:
                    raise RuntimeError(
                        f"PU Option-Y split has missing class(es): {name} classes={sorted(u)}. "
                        f"Check settings/codes filters."
                    )

        fs = float(src.fs)

    else:
        raise ValueError(f"Unknown dataset: {ds}")

    window_samples = _first_len(src.X[0])
    window_seconds = float(window_samples) / float(fs) if fs > 0 else float("nan")
    return src, tgt, fs, window_samples, window_seconds

def _run_pi_fsl(
    src: Bundle,
    tgt: Bundle,
    args,
    use_physics: bool,
) -> Dict[str, Any]:
    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)

    model = RelationNet(in_channels=len(modalities)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    phys = None
    if use_physics:

        spectral_bands = None
        if getattr(args, "spectral_bands_json", None):
            spectral_bands = _json.loads(str(args.spectral_bands_json))

        le = getattr(args, "lambda_energy", None)
        ls = getattr(args, "lambda_spectral", None)
        la = getattr(args, "lambda_envelope", None)

        pr_cfg = PhysicsRegularizationConfig(
            enabled=True,
            lambda_energy=float(le) if le is not None else PhysicsRegularizationConfig.lambda_energy,
            lambda_spectral=float(ls) if ls is not None else PhysicsRegularizationConfig.lambda_spectral,
            lambda_envelope=float(la) if la is not None else PhysicsRegularizationConfig.lambda_envelope,
            spectral_bands=spectral_bands,
            motor_current_enabled=bool(args.motor_current_enabled),
            lambda_current=float(args.lambda_current),
            current_key=str(args.current_key),
        )

        if str(args.variant) == "energy_only":
            pr_cfg.lambda_spectral = 0.0
            pr_cfg.lambda_envelope = 0.0
        elif str(args.variant) == "spectral_only":
            pr_cfg.lambda_energy = 0.0
            pr_cfg.lambda_envelope = 0.0
        elif str(args.variant) == "envelope_only":
            pr_cfg.lambda_energy = 0.0
            pr_cfg.lambda_spectral = 0.0

        phys = PhysicsInformedRegularizer(pr_cfg).to(device)

    model.train()
    for ep in range(int(args.train_episodes)):
        sup_X, sup_y, qry_X, qry_y, sup_idx, qry_idx = _sample_episode_bundle(
            src,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=ep + int(args.seed) * 1000,
            args=args,
            with_indices=True,
        )

        Sx = _windows_to_scalograms(sup_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Qx = _windows_to_scalograms(qry_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        scores, classes = model.forward_episode(Sx, Sy, Qx)
        loss = F.cross_entropy(scores, Qy)

        if phys is not None and (ep % int(args.physics_every) == 0):
            # Physics MUST use true dataset labels (healthy/worn), not episode-remapped labels.
            # Use indices to recover true labels.
            episode_raw: List[RawWindow] = list(sup_X) + list(qry_X)
            true_labels: List[int] = [int(src.y[int(i)]) for i in list(sup_idx) + list(qry_idx)]

            # embed features
            all_imgs = torch.cat([Sx, Qx], dim=0)
            emb_fn = getattr(model, "embed", None)
            feature_maps = emb_fn(all_imgs) if callable(emb_fn) else model.embed(all_imgs)

            loss_phys = phys(
                pred_outputs=(scores, classes),
                raw_signals=episode_raw,
                labels=true_labels,
                feature_maps=feature_maps,
                fs=float(src.fs),
            )
            loss = loss + float(args.physics_weight) * loss_phys

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    do_transfer = bool(getattr(args, "transfer_finetune", False))
    ratio = float(getattr(args, "target_adapt_ratio", 0.40))
    ratio = min(max(ratio, 0.0), 1.0)

    adapt_dataset_enabled = False
    adapt_bundle = None

    if do_transfer:
        y_np = np.asarray(tgt.y, dtype=int)
        idx0 = np.where(y_np == 0)[0]
        idx1 = np.where(y_np == 1)[0]

        if len(idx0) > 0 and len(idx1) > 0:
            rng = np.random.RandomState(int(args.seed) * 1337 + 17)
            rng.shuffle(idx0)
            rng.shuffle(idx1)

            def _split_one(ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                n = len(ix)
                n_adapt = int(round(ratio * n))
                n_adapt = max(1, n_adapt)
                if n - n_adapt == 0 and n > 1:
                    n_adapt -= 1
                return ix[:n_adapt], ix[n_adapt:]

            a0, e0 = _split_one(idx0)
            a1, e1 = _split_one(idx1)

            if len(e0) > 0 and len(e1) > 0:
                adapt_idx = np.concatenate([a0, a1])
                test_idx = np.concatenate([e0, e1])
                rng.shuffle(adapt_idx)
                rng.shuffle(test_idx)

                adapt_X = [tgt.X[int(i)] for i in adapt_idx]
                adapt_y = [int(tgt.y[int(i)]) for i in adapt_idx]
                adapt_domain = [tgt.domain[int(i)] for i in adapt_idx] if tgt.domain is not None else None
                adapt_fid = [tgt.file_id[int(i)] for i in adapt_idx] if tgt.file_id is not None else None

                test_X = [tgt.X[int(i)] for i in test_idx]
                test_y = [int(tgt.y[int(i)]) for i in test_idx]
                test_domain = [tgt.domain[int(i)] for i in test_idx] if tgt.domain is not None else None
                test_fid = [tgt.file_id[int(i)] for i in test_idx] if tgt.file_id is not None else None

                adapt_bundle = Bundle(X=adapt_X, y=adapt_y, fs=tgt.fs, domain=adapt_domain, file_id=adapt_fid)
                tgt = Bundle(X=test_X, y=test_y, fs=tgt.fs, domain=test_domain, file_id=test_fid)
                adapt_dataset_enabled = True

    if adapt_dataset_enabled and adapt_bundle is not None:
        transfer_episodes = 200
        tlr = 5e-4
        twd = 1e-4
        tstep = 100
        tgamma = 0.5

        model.train()
        opt_t = torch.optim.Adam(model.parameters(), lr=tlr, weight_decay=twd)
        sch_t = torch.optim.lr_scheduler.StepLR(opt_t, step_size=tstep, gamma=tgamma)

        for ep in range(int(transfer_episodes)):
            sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
                adapt_bundle,
                n_way=int(args.n_way),
                k_shot=int(args.k_shot),
                q_query=int(args.q_query),
                seed=50_000 + ep + int(args.seed) * 1000,
                args=args,
                with_indices=False,
            )

            Sx = _windows_to_scalograms(sup_X, adapt_bundle.fs, modalities, args.pad_missing_modalities).to(device)
            Qx = _windows_to_scalograms(qry_X, adapt_bundle.fs, modalities, args.pad_missing_modalities).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            scores, _classes = model.forward_episode(Sx, Sy, Qx)
            loss_t = F.cross_entropy(scores, Qy)

            opt_t.zero_grad(set_to_none=True)
            loss_t.backward()
            opt_t.step()
            sch_t.step()

    model.eval()
    accs: List[float] = []
    baccs: List[float] = []
    f1s: List[float] = []
    with torch.no_grad():
        for ep in range(int(args.eval_episodes)):
            sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
                tgt,
                n_way=int(args.n_way),
                k_shot=int(args.k_shot),
                q_query=int(args.q_query),
                seed=10_000 + ep + int(args.seed) * 1000,
                args=args,
                with_indices=False,
            )

            Sx = _windows_to_scalograms(sup_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Qx = _windows_to_scalograms(qry_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            scores, _classes = model.forward_episode(Sx, Sy, Qx)
            y_pred = torch.argmax(scores, dim=1).cpu().numpy()
            y_true = Qy.cpu().numpy()
            m = compute_metrics(y_true, y_pred)
            accs.append(m["acc"])
            baccs.append(m["bacc"])
            f1s.append(m["macro_f1"])

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)

    return {
        "acc_mean": acc_m,
        "acc_ci95": acc_ci,
        "bacc_mean": bacc_m,
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m,
        "macro_f1_ci95": f1_ci,
    }

def _run_pi_fsl_matching(src: Bundle, tgt: Bundle, args, use_physics: bool) -> Dict[str, Any]:
    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)
    pool = str(getattr(args, "embed_pool", "flatten"))

    enc = ConvEmbedding(in_channels=len(modalities)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    phys = None
    if use_physics:
        pr_cfg = PhysicsRegularizationConfig(
            enabled=True,
            motor_current_enabled=bool(args.motor_current_enabled),
            lambda_current=float(args.lambda_current),
            current_key=str(args.current_key),
        )
        phys = PhysicsInformedRegularizer(pr_cfg).to(device)

    enc.train()
    for ep in range(int(args.train_episodes)):
        sup_X, sup_y, qry_X, qry_y, sup_idx, qry_idx = _sample_episode_bundle(
            src,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=ep + int(args.seed) * 1000,
            args=args,
            with_indices=True,
        )

        Sx = _windows_to_scalograms(sup_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Qx = _windows_to_scalograms(qry_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        feat_s = _pool_feat(enc(Sx), pool)
        feat_q = _pool_feat(enc(Qx), pool)

        probs = _class_probs(feat_s, Sy, feat_q, int(args.n_way))
        loss = F.nll_loss(torch.log(probs), Qy)

        if use_physics and phys is not None:
            episode_raw = list(sup_X) + list(qry_X)
            true_labels = [int(src.y[int(i)]) for i in list(sup_idx) + list(qry_idx)]
            all_imgs = torch.cat([Sx, Qx], dim=0)
            feature_maps = enc(all_imgs)
            loss_phys = phys(
                pred_outputs=(probs, None),
                raw_signals=episode_raw,
                labels=true_labels,
                feature_maps=feature_maps,
                fs=float(src.fs),
            )
            loss = loss + float(args.physics_weight) * loss_phys

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    do_transfer = bool(getattr(args, "transfer_finetune", False))
    ratio = float(getattr(args, "target_adapt_ratio", 0.40))
    ratio = min(max(ratio, 0.0), 1.0)

    adapt_bundle = None
    if do_transfer:
        y_np = np.asarray(tgt.y, dtype=int)
        idx0 = np.where(y_np == 0)[0]
        idx1 = np.where(y_np == 1)[0]

        if len(idx0) > 0 and len(idx1) > 0:
            rng = np.random.RandomState(int(args.seed) * 1337 + 17)
            rng.shuffle(idx0)
            rng.shuffle(idx1)

            def _split_one(ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                n = len(ix)
                n_adapt = int(round(ratio * n))
                n_adapt = max(1, n_adapt)
                if n - n_adapt == 0 and n > 1:
                    n_adapt -= 1
                return ix[:n_adapt], ix[n_adapt:]

            a0, e0 = _split_one(idx0)
            a1, e1 = _split_one(idx1)

            if len(e0) > 0 and len(e1) > 0:
                adapt_idx = np.concatenate([a0, a1])
                test_idx = np.concatenate([e0, e1])
                rng.shuffle(adapt_idx)
                rng.shuffle(test_idx)

                adapt_X = [tgt.X[int(i)] for i in adapt_idx]
                adapt_y = [int(tgt.y[int(i)]) for i in adapt_idx]
                adapt_domain = [tgt.domain[int(i)] for i in adapt_idx] if tgt.domain is not None else None
                adapt_fid = [tgt.file_id[int(i)] for i in adapt_idx] if tgt.file_id is not None else None

                test_X = [tgt.X[int(i)] for i in test_idx]
                test_y = [int(tgt.y[int(i)]) for i in test_idx]
                test_domain = [tgt.domain[int(i)] for i in test_idx] if tgt.domain is not None else None
                test_fid = [tgt.file_id[int(i)] for i in test_idx] if tgt.file_id is not None else None

                adapt_bundle = Bundle(X=adapt_X, y=adapt_y, fs=tgt.fs, domain=adapt_domain, file_id=adapt_fid)
                tgt = Bundle(X=test_X, y=test_y, fs=tgt.fs, domain=test_domain, file_id=test_fid)

    if adapt_bundle is not None:
        transfer_episodes = 200
        tlr = 5e-4
        twd = 1e-4
        tstep = 100
        tgamma = 0.5

        enc.train()
        opt_t = torch.optim.Adam(enc.parameters(), lr=tlr, weight_decay=twd)
        sch_t = torch.optim.lr_scheduler.StepLR(opt_t, step_size=tstep, gamma=tgamma)

        for ep in range(int(transfer_episodes)):
            sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
                adapt_bundle,
                n_way=int(args.n_way),
                k_shot=int(args.k_shot),
                q_query=int(args.q_query),
                seed=50_000 + ep + int(args.seed) * 1000,
                args=args,
                with_indices=False,
            )

            Sx = _windows_to_scalograms(sup_X, adapt_bundle.fs, modalities, args.pad_missing_modalities).to(device)
            Qx = _windows_to_scalograms(qry_X, adapt_bundle.fs, modalities, args.pad_missing_modalities).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            feat_s = _pool_feat(enc(Sx), pool)
            feat_q = _pool_feat(enc(Qx), pool)
            probs = _class_probs(feat_s, Sy, feat_q, int(args.n_way))
            loss_t = F.nll_loss(torch.log(probs), Qy)

            opt_t.zero_grad(set_to_none=True)
            loss_t.backward()
            opt_t.step()
            sch_t.step()

        enc.eval()

    enc.eval()
    accs, baccs, f1s = [], [], []
    with torch.no_grad():
        for ep in range(int(args.eval_episodes)):
            sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
                tgt,
                n_way=int(args.n_way),
                k_shot=int(args.k_shot),
                q_query=int(args.q_query),
                seed=10_000 + ep + int(args.seed) * 1000,
                args=args,
                with_indices=False,
            )

            Sx = _windows_to_scalograms(sup_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Qx = _windows_to_scalograms(qry_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            feat_s = _pool_feat(enc(Sx), pool)
            feat_q = _pool_feat(enc(Qx), pool)
            probs = _class_probs(feat_s, Sy, feat_q, int(args.n_way))

            y_pred = torch.argmax(probs, dim=1).cpu().numpy()
            y_true = Qy.cpu().numpy()

            m = compute_metrics(y_true, y_pred)
            accs.append(m["acc"])
            baccs.append(m["bacc"])
            f1s.append(m["macro_f1"])

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)
    return {
        "acc_mean": acc_m,
        "acc_ci95": acc_ci,
        "bacc_mean": bacc_m,
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m,
        "macro_f1_ci95": f1_ci,
    }


def _run_pi_fsl_target_finetune(
    src: Bundle,
    tgt: Bundle,
    args,
    use_physics: bool,
) -> Dict[str, Any]:
    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)
    pool = str(getattr(args, "embed_pool", "flatten"))

    enc = ConvEmbedding(in_channels=len(modalities)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    phys = None
    if use_physics:
        pr_cfg = PhysicsRegularizationConfig(
            enabled=True,
            motor_current_enabled=bool(args.motor_current_enabled),
            lambda_current=float(args.lambda_current),
            current_key=str(args.current_key),
        )
        phys = PhysicsInformedRegularizer(pr_cfg).to(device)

    enc.train()
    for ep in range(int(args.train_episodes)):
        sup_X, sup_y, qry_X, qry_y, sup_idx, qry_idx = _sample_episode_bundle(
            src,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=ep + int(args.seed) * 1000,
            args=args,
            with_indices=True,
        )

        Sx = _windows_to_scalograms(sup_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Qx = _windows_to_scalograms(qry_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        featmaps_s = enc(Sx)
        featmaps_q = enc(Qx)
        feat_s = _pool_feat(featmaps_s, pool)
        feat_q = _pool_feat(featmaps_q, pool)

        protos = torch.stack([feat_s[Sy == c].mean(dim=0) for c in range(int(args.n_way))], dim=0)
        logits = -(torch.cdist(feat_q, protos, p=2.0) ** 2)
        loss = F.cross_entropy(logits, Qy)

        if use_physics and phys is not None:
            episode_raw = list(sup_X) + list(qry_X)
            true_labels = [int(src.y[int(i)]) for i in list(sup_idx) + list(qry_idx)]
            all_imgs = torch.cat([Sx, Qx], dim=0)
            feature_maps = enc(all_imgs)
            loss_phys = phys(
                pred_outputs=(logits, None),
                raw_signals=episode_raw,
                labels=true_labels,
                feature_maps=feature_maps,
                fs=float(src.fs),
            )
            loss = loss + float(args.physics_weight) * loss_phys

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    enc.eval()
    accs, baccs, f1s = [], [], []
    for ep in range(int(args.eval_episodes)):
        sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
            tgt,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=10_000 + ep + int(args.seed) * 1000,
            args=args,
            with_indices=False,
        )

        Sx = _windows_to_scalograms(sup_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
        Qx = _windows_to_scalograms(qry_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        with torch.no_grad():
            feat_s = _pool_feat(enc(Sx), pool)
            feat_q = _pool_feat(enc(Qx), pool)

        head = nn.Linear(feat_s.size(1), int(args.n_way)).to(device)
        opt_h = torch.optim.SGD(head.parameters(), lr=float(getattr(args, "ft_lr", 0.1)))

        head.train()
        for _ in range(int(getattr(args, "ft_steps", 100))):
            logits_s = head(feat_s.detach())
            loss_h = F.cross_entropy(logits_s, Sy)
            opt_h.zero_grad(set_to_none=True)
            loss_h.backward()
            opt_h.step()

        head.eval()
        with torch.no_grad():
            logits_q = head(feat_q)
            y_pred = torch.argmax(logits_q, dim=1).cpu().numpy()
            y_true = Qy.cpu().numpy()

        m = compute_metrics(y_true, y_pred)
        accs.append(m["acc"])
        baccs.append(m["bacc"])
        f1s.append(m["macro_f1"])

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)
    return {
        "acc_mean": acc_m,
        "acc_ci95": acc_ci,
        "bacc_mean": bacc_m,
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m,
        "macro_f1_ci95": f1_ci,
    }


def _pool_feat(feat: "torch.Tensor", mode: str) -> "torch.Tensor":
    # feat: (B, C, 4, 4)
    if mode == "gap":
        return feat.mean(dim=(2, 3))
    return feat.reshape(feat.size(0), -1)


def _run_protonet(src: Bundle, tgt: Bundle, args) -> Dict[str, Any]:

    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)
    pool = str(getattr(args, "embed_pool", "flatten"))

    enc = ConvEmbedding(in_channels=len(modalities)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    enc.train()
    for ep in range(int(args.train_episodes)):
        sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
            src,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=ep + int(args.seed) * 1000,
            args=args,
            with_indices=False,
        )

        Sx = _windows_to_scalograms(sup_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Qx = _windows_to_scalograms(qry_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        feat_s = _pool_feat(enc(Sx), pool)  # (Ns, D)
        feat_q = _pool_feat(enc(Qx), pool)  # (Nq, D)

        protos = []
        for c in range(int(args.n_way)):
            protos.append(feat_s[Sy == c].mean(dim=0))
        protos = torch.stack(protos, dim=0)

        logits = -(torch.cdist(feat_q, protos, p=2.0) ** 2)
        loss = F.cross_entropy(logits, Qy)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    enc.eval()
    accs, baccs, f1s = [], [], []
    with torch.no_grad():
        for ep in range(int(args.eval_episodes)):
            sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
                tgt,
                n_way=int(args.n_way),
                k_shot=int(args.k_shot),
                q_query=int(args.q_query),
                seed=10_000 + ep + int(args.seed) * 1000,
                args=args,
                with_indices=False,
            )

            Sx = _windows_to_scalograms(sup_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Qx = _windows_to_scalograms(qry_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            feat_s = _pool_feat(enc(Sx), pool)
            feat_q = _pool_feat(enc(Qx), pool)

            protos = torch.stack([feat_s[Sy == c].mean(dim=0) for c in range(int(args.n_way))], dim=0)
            logits = -(torch.cdist(feat_q, protos, p=2.0) ** 2)

            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = Qy.cpu().numpy()

            m = compute_metrics(y_true, y_pred)
            accs.append(m["acc"])
            baccs.append(m["bacc"])
            f1s.append(m["macro_f1"])

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)
    return {
        "acc_mean": acc_m,
        "acc_ci95": acc_ci,
        "bacc_mean": bacc_m,
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m,
        "macro_f1_ci95": f1_ci,
    }

def _class_probs(feat_s, Sy, feat_q, n_way: int, temperature: float = 10.0):
    feat_s_n = F.normalize(feat_s, dim=1)
    feat_q_n = F.normalize(feat_q, dim=1)

    sim = (feat_q_n @ feat_s_n.t()) * float(temperature)   # (Nq, Ns)
    attn = F.softmax(sim, dim=1)                            # (Nq, Ns)

    probs = []
    for c in range(int(n_way)):
        probs.append(attn[:, (Sy == c)].sum(dim=1, keepdim=True))
    probs = torch.cat(probs, dim=1)

    return probs.clamp_min(1e-9)


def _run_matchingnet(src: Bundle, tgt: Bundle, args) -> Dict[str, Any]:

    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)
    pool = str(getattr(args, "embed_pool", "flatten"))

    enc = ConvEmbedding(in_channels=len(modalities)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    enc.train()
    for ep in range(int(args.train_episodes)):
        sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
            src,
            n_way=int(args.n_way),
            k_shot=int(args.k_shot),
            q_query=int(args.q_query),
            seed=ep + int(args.seed) * 1000,
            args=args,
            with_indices=False,
        )

        Sx = _windows_to_scalograms(sup_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Qx = _windows_to_scalograms(qry_X, src.fs, modalities, args.pad_missing_modalities).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        feat_s = _pool_feat(enc(Sx), pool)
        feat_q = _pool_feat(enc(Qx), pool)

        probs = _class_probs(feat_s, Sy, feat_q, int(args.n_way))
        loss = F.nll_loss(torch.log(probs), Qy)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    enc.eval()
    accs, baccs, f1s = [], [], []
    with torch.no_grad():
        for ep in range(int(args.eval_episodes)):
            sup_X, sup_y, qry_X, qry_y = _sample_episode_bundle(
                tgt,
                n_way=int(args.n_way),
                k_shot=int(args.k_shot),
                q_query=int(args.q_query),
                seed=10_000 + ep + int(args.seed) * 1000,
                args=args,
                with_indices=False,
            )

            Sx = _windows_to_scalograms(sup_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Qx = _windows_to_scalograms(qry_X, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            feat_s = _pool_feat(enc(Sx), pool)
            feat_q = _pool_feat(enc(Qx), pool)

            probs = _class_probs(feat_s, Sy, feat_q, int(args.n_way))
            y_pred = torch.argmax(probs, dim=1).cpu().numpy()
            y_true = Qy.cpu().numpy()

            m = compute_metrics(y_true, y_pred)
            accs.append(m["acc"])
            baccs.append(m["bacc"])
            f1s.append(m["macro_f1"])

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)
    return {
        "acc_mean": acc_m,
        "acc_ci95": acc_ci,
        "bacc_mean": bacc_m,
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m,
        "macro_f1_ci95": f1_ci,
    }

def _run_supervised_target_only(tgt: Bundle, args) -> Dict[str, Any]:
    device = torch.device(_resolve_device(args.device))
    modalities = _parse_modalities(args.modalities)

    # map labels -> 0..C-1
    y_raw = np.asarray(tgt.y, dtype=np.int64)
    classes = sorted(np.unique(y_raw).tolist())
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cls2id[int(v)] for v in y_raw], dtype=np.int64)
    n_classes = int(len(classes))

    train_idx, test_idx = _split_by_file_id(tgt.file_id, seed=int(args.seed), test_ratio=0.2)

    enc = ConvEmbedding(in_channels=len(modalities)).to(device)
    head = nn.Linear(64 * 4 * 4, n_classes).to(device)  # ConvEmbedding default ends at 64ch 4x4

    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=float(getattr(args, "sup_lr", 1e-3)))

    def _batch(ix: np.ndarray, bs: int, rng: np.random.RandomState):
        pick = rng.choice(ix, size=min(bs, ix.size), replace=(ix.size < bs))
        Xb = [tgt.X[int(i)] for i in pick]
        yb = y[pick]
        Xb = _windows_to_scalograms(Xb, tgt.fs, modalities, args.pad_missing_modalities)
        return Xb, torch.from_numpy(yb).long()

    enc.train(); head.train()
    rng = np.random.RandomState(int(args.seed) + 123)
    steps_per_epoch = max(1, int(np.ceil(train_idx.size / int(getattr(args, "sup_batch_size", 32)))))
    for _ep in range(int(getattr(args, "sup_epochs", 10))):
        for _ in range(steps_per_epoch):
            Xb, yb = _batch(train_idx, int(getattr(args, "sup_batch_size", 32)), rng)
            Xb = Xb.to(device); yb = yb.to(device)
            feat = enc(Xb).reshape(Xb.size(0), -1)
            logits = head(feat)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    enc.eval(); head.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        # evaluate in chunks to control memory
        bs = int(getattr(args, "sup_batch_size", 32))
        for start in range(0, test_idx.size, bs):
            pick = test_idx[start:start+bs]
            Xb = [tgt.X[int(i)] for i in pick]
            yb = y[pick]
            Xb = _windows_to_scalograms(Xb, tgt.fs, modalities, args.pad_missing_modalities).to(device)
            feat = enc(Xb).reshape(Xb.size(0), -1)
            logits = head(feat)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb)

    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    m = compute_metrics(y_true, y_pred)
    return {"acc_mean": m["acc"], "acc_ci95": None, "bacc_mean": m["bacc"], "bacc_ci95": None, "macro_f1_mean": m["macro_f1"], "macro_f1_ci95": None}


def _run_maml(src: Bundle, tgt: Bundle, args) -> Dict[str, Any]:
    def _to_1d_list(X: List[RawWindow], key: str = "vibration") -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for w in X:
            if isinstance(w, dict):
                if key in w:
                    out.append(np.asarray(w[key], dtype=np.float32).reshape(-1))
                else:
                    out.append(np.asarray(next(iter(w.values())), dtype=np.float32).reshape(-1))
            else:
                out.append(np.asarray(w, dtype=np.float32).reshape(-1))
        return out

    def _pad_or_crop_1d(a: np.ndarray, L: int) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.shape[0] == L:
            return a
        if a.shape[0] > L:
            return a[:L]
        out = np.zeros((L,), dtype=np.float32)
        out[: a.shape[0]] = a
        return out

    maml_args = MAMLArgs(
        device=_resolve_device(args.device),
        train_episodes=int(getattr(args, "maml_train_episodes", args.train_episodes)),
        eval_episodes=int(getattr(args, "maml_eval_episodes", args.eval_episodes)),
        n_way=int(args.n_way),
        k_shot=int(args.k_shot),
        q_query=int(args.q_query),
        inner_steps=int(getattr(args, "maml_inner_steps", 5)),
        inner_lr=float(getattr(args, "maml_inner_lr", 0.01)),
        outer_lr=float(getattr(args, "maml_outer_lr", 1e-3)),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
        grad_clip=float(getattr(args, "maml_grad_clip", 5.0)),
        z_norm=not bool(getattr(args, "maml_no_z_norm", False)),
    )


    src_X_1d = _to_1d_list(src.X)
    tgt_X_1d = _to_1d_list(tgt.X)

    lens = [x.shape[0] for x in src_X_1d] + [x.shape[0] for x in tgt_X_1d]
    target_len = int(np.median(lens)) if lens else 1024
    target_len = max(128, target_len)

    src_X_1d = [_pad_or_crop_1d(x, target_len) for x in src_X_1d]
    tgt_X_1d = [_pad_or_crop_1d(x, target_len) for x in tgt_X_1d]

    return run_maml(
        src_X=src_X_1d,
        src_y=list(src.y),
        tgt_X=tgt_X_1d,
        tgt_y=list(tgt.y),
        args=maml_args,
    )

def parse_args():
    p = argparse.ArgumentParser("Run one benchmark job and append to results.jsonl")

    p.add_argument("--project_root", type=str, default=".")

    p.add_argument("--dataset", type=str, choices=["bosch", "cwru", "hust_cn", "hust_vn", "pu", "bosch_mi", "bosch_milling", "scidata2025"])
    p.add_argument("--data_root", type=str)

    p.add_argument("--src_dataset", type=str, choices=["bosch", "cwru", "hust_cn", "hust_vn", "pu", "bosch_mi", "bosch_milling", "scidata2025"])
    p.add_argument("--tgt_dataset", type=str, choices=["bosch", "cwru", "hust_cn", "hust_vn", "pu", "bosch_mi", "bosch_milling", "scidata2025"])
    p.add_argument("--src_data_root", type=str)
    p.add_argument("--tgt_data_root", type=str)

    p.add_argument("--max_files", type=int, default=None, help="Cap number of files/rows loaded in single-dataset mode")
    p.add_argument("--src_max_files", type=int, default=None, help="Cap number of files/rows loaded for src_dataset in cross-dataset mode")
    p.add_argument("--tgt_max_files", type=int, default=None, help="Cap number of files/rows loaded for tgt_dataset in cross-dataset mode")

    # experiment identity
    p.add_argument("--scenario", type=str, default="")
    p.add_argument("--exp_id", type=str, default="")
    p.add_argument("--result_source", type=str, default="Own")
    p.add_argument("--notes", type=str, default="")

    # protocol
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "pi_fsl",        # RelationNet + physics (episodic)
            "relationnet",   # RelationNet baseline (episodic, no physics)
            "protonet",      # classic ProtoNet (episodic)
            "matchingnet",   # classic MatchingNet (episodic)
            "maml",          # baseline (binary only in this repo)
        ],
    )

    # NOTE:
    # - "wo_physics": episodic FSL without physics
    # - "wo_fsl": supervised target training (target plentiful ablation)
    # - *_only: physics-term-only ablations (episodic FSL)
    p.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=[
            "full",
            "wo_physics",
            "wo_fsl",
            "wo_fsl_ft",
            "energy_only",
            "spectral_only",
            "envelope_only",
        ],
    )

    # physics term knobs (for term-only ablations / band sweeps)
    p.add_argument("--lambda_energy", type=float, default=None)
    p.add_argument("--lambda_spectral", type=float, default=None)
    p.add_argument("--lambda_envelope", type=float, default=None)
    p.add_argument("--spectral_bands_json", type=str, default=None,
                   help='JSON dict like {"low":[0,75],"mid":[75,300],"high":[300,1000]}')

    # classic FSL baseline knobs
    p.add_argument("--embed_pool", type=str, default="flatten", choices=["flatten", "gap"],
                   help="How to pool ConvEmbedding outputs for ProtoNet/MatchingNet")
    p.add_argument("--matching_temperature", type=float, default=10.0,
                   help="Softmax temperature for MatchingNet attention")

    # PSD-guided source balancing (Bosch / PI-FSL only)
    p.add_argument("--psd_guided_balance", action="store_true",
                help="Apply PSD-guided source balancing (PI-FSL only, Bosch only)")
    p.add_argument("--psd_target_ratio", type=float, default=0.3,
                help="Target ratio for sparse sampling majority class (default 0.3)")
    p.add_argument("--psd_n_clusters", type=int, default=None,
                help="Number of clusters for PSD sampling (default None => derived)")
    p.add_argument("--psd_random_state", type=int, default=42,
                help="Random state for PSD sampling")

    # wo_fsl supervised target training knobs (target plentiful ablation)
    p.add_argument("--sup_epochs", type=int, default=10)
    p.add_argument("--sup_batch_size", type=int, default=32)
    p.add_argument("--sup_lr", type=float, default=1e-3)

    p.add_argument("--source_domain", type=str, required=True)
    p.add_argument("--target_domain", type=str, required=True)
    p.add_argument("--normalization", type=str, default="per_window", choices=["per_window", "none", "zscore"])
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--n_way", type=int, default=2)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=16)

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | cuda:0 | auto (auto uses cuda if available)",
    )
    p.add_argument("--train_episodes", type=int, default=1000)
    p.add_argument("--eval_episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # physics knobs
    p.add_argument("--physics_every", type=int, default=1)
    p.add_argument("--physics_weight", type=float, default=1.0)

    # PI-FSL v01-style target transfer fine-tune (only when enabled by runner)
    p.add_argument("--transfer_finetune", action="store_true")
    p.add_argument("--target_adapt_ratio", type=float, default=0.40)

    # multimodal knobs
    p.add_argument("--modalities", type=str, default="vibration")
    p.add_argument("--pad_missing_modalities", type=str, default="zeros", choices=["zeros", "duplicate_first"])
    p.add_argument("--motor_current_enabled", action="store_true")
    p.add_argument("--lambda_current", type=float, default=0.1)
    p.add_argument("--current_key", type=str, default="motor_current")

    # dataset-specific knobs (compat)
    p.add_argument("--cwru_time_steps", type=int, default=1024)
    p.add_argument("--overlap_ratio", type=float, default=0.5)

    # label mode + cwru alias
    p.add_argument("--label_mode", type=str, default=None)
    p.add_argument("--cwru_label_mode", type=str, default=None, choices=["fault", "severity"])

    p.add_argument(
        "--cwru_pos_label",
        type=int,
        default=None,
        help="Option-Y (2-way): keep only healthy label 0 and this single fault label (e.g., 1).",
    )

    p.add_argument(
        "--bosch_mi_drop_mid",
        action="store_true",
        help="Option-Y analogue for milling: drop mid-life class and binarize early vs late (requires bosch_mi_label_mode=3class).",
    )

    p.add_argument("--pu_healthy_code", type=str, default=None, help="Option-Y: healthy code (e.g., K001)")
    p.add_argument("--pu_pos_code", type=str, default=None, help="Option-Y: single fault code (e.g., KI14)")

    # hust-vn
    p.add_argument("--hust_vn_domain_axis", type=str, default="load", choices=["load", "speed"])

    # Bosch milling
    p.add_argument("--bosch_mi_label_mode", type=str, default="3class", choices=["binary", "3class"])
    p.add_argument("--bosch_mi_prefer_feature_table", action=argparse.BooleanOptionalAction, default=True, help="Prefer loading a processed feature table if found")
    p.add_argument("--bosch_mi_include_domains", type=str, default=None, help="Comma-separated domain ids to keep (advanced)")

    # PU (Paderborn) subsetting
    p.add_argument("--pu_include_codes", type=str, default=None, help="Comma-separated bearing codes to keep (e.g., K001,KA01,KB23)")
    p.add_argument("--pu_include_settings", type=str, default=None, help="Comma-separated settings to keep (e.g., N15_M07_F10)")
    p.add_argument("--pu_max_windows_per_file", type=int, default=None, help="Cap windows generated per MAT file")
    p.add_argument("--pu_window_samples", type=int, default=None)
    p.add_argument("--pu_hop_samples", type=int, default=None)

    # MAML hyperparams
    # MAML hyperparams (only used when method==maml)
    p.add_argument("--maml_inner_steps", type=int, default=5)
    p.add_argument("--maml_train_episodes", type=int, default=4000)
    p.add_argument("--maml_eval_episodes", type=int, default=200)
    p.add_argument("--maml_grad_clip", type=float, default=5.0)
    p.add_argument("--maml_no_z_norm", action="store_true")
    p.add_argument("--maml_inner_lr", type=float, default=0.01)
    p.add_argument("--maml_outer_lr", type=float, default=1e-3)

    # output:
    p.add_argument("--results_jsonl", type=str, default="artifacts/results/results.jsonl")
    p.add_argument("--out_jsonl", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()


    if args.out_jsonl:
        args.results_jsonl = args.out_jsonl

    project_root = Path(args.project_root).resolve()
    bootstrap(project_root)

    set_seed(int(args.seed))

    src, tgt, fs, window_samples, window_seconds = _load_pair(args)

    if str(args.method) == "pi_fsl":
        if bool(getattr(args, "psd_guided_balance", False)) and str(args.dataset).lower() == "bosch":
            v = str(getattr(args, "variant", "full"))
            # If variant is wo_fsl, src is not used anyway; skip to avoid confusion
            if v != "wo_fsl":

                sampler = PSDGuidedSampler(
                    target_imbalance_ratio=float(getattr(args, "psd_target_ratio", 0.3)),
                    n_clusters=getattr(args, "psd_n_clusters", None),
                    clustering_method="kmeans",
                    random_state=int(getattr(args, "psd_random_state", 42)),
                )

                Xb, yb, sel = sampler.balance_dataset_with_indices(src.X, src.y)

                # keep metadata aligned
                src = Bundle(
                    X=Xb,
                    y=yb,
                    domain=[src.domain[int(i)] for i in sel],
                    file_id=[src.file_id[int(i)] for i in sel],
                    fs=float(src.fs),
                )

                print(f"[PSD-balance] src windows: {len(src.X)} (after PSD-guided balancing)")


    if args.method == "maml":
        metrics = _run_maml(src, tgt, args)
        input_rep = "raw_1d"

    elif args.method == "pi_fsl":
        v = str(args.variant)

        BOSCH_ONLY = {"wo_fsl_ft", "wo_physics"}  
        if str(getattr(args, "dataset", "")).lower() != "bosch" and v in BOSCH_ONLY:
            raise ValueError(f"Variant '{v}' is Bosch-only in this setup (dataset={args.dataset}).")
        if v == "wo_fsl":
            metrics = _run_supervised_target_only(tgt, args)
        elif v == "wo_fsl_ft":

            metrics = _run_pi_fsl_target_finetune(src, tgt, args, use_physics=True)
        elif v in ("wo_physics",):  

            metrics = _run_pi_fsl_matching(src, tgt, args, use_physics=False)
        else:
            use_physics = (v != "wo_physics")
            metrics = _run_pi_fsl_matching(src, tgt, args, use_physics=use_physics)

        input_rep = "scalogram_64x64"

    elif args.method == "relationnet":
        # classic RelationNet baseline (no physics, episodic)
        metrics = _run_pi_fsl(src, tgt, args, use_physics=False)
        input_rep = "scalogram_64x64"

    elif args.method == "protonet":
        metrics = _run_protonet(src, tgt, args)
        input_rep = "scalogram_64x64"

    elif args.method == "matchingnet":
        metrics = _run_matchingnet(src, tgt, args)
        input_rep = "scalogram_64x64"

    else:
        raise ValueError(f"Unknown method: {args.method}")


    scenario = args.scenario.strip()
    if not scenario:
        if args.src_dataset and args.tgt_dataset:
            scenario = f"{args.src_dataset}_to_{args.tgt_dataset}"
        else:
            scenario = "single_dataset"

    dataset_label = args.dataset if args.dataset else f"{args.src_dataset}->{args.tgt_dataset}"

    record: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "dataset": dataset_label,
        "scenario": scenario,
        "method": args.method,
        "variant": args.variant,
        "result_source": args.result_source,
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "n_way": int(args.n_way),
        "k_shot": int(args.k_shot),
        "q_query": int(args.q_query),
        "train_episodes": int(args.train_episodes),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "fs": float(fs),
        "window_samples": int(window_samples),
        "window_seconds": float(window_seconds),
        "overlap_ratio": float(args.overlap_ratio),
        "normalization": args.normalization,
        "input_representation": input_rep,
        "acc_mean": float(metrics.get("acc_mean", float("nan"))),
        "acc_ci95": metrics.get("acc_ci95", None),
        "bacc_mean": float(metrics.get("bacc_mean", float("nan"))),
        "bacc_ci95": metrics.get("bacc_ci95", None),
        "macro_f1_mean": float(metrics.get("macro_f1_mean", float("nan"))),
        "macro_f1_ci95": metrics.get("macro_f1_ci95", None),
        "notes": args.notes,
        "extra": {
            "exp_id": args.exp_id,
            "modalities": _parse_modalities(args.modalities),
            "pad_missing_modalities": args.pad_missing_modalities,
            "motor_current_enabled": bool(args.motor_current_enabled),
            "lambda_current": float(args.lambda_current),
            "current_key": str(args.current_key),
            "physics_weight": float(args.physics_weight),
            "physics_every": int(args.physics_every),
            "device_resolved": _resolve_device(args.device),
            "src_dataset": args.src_dataset,
            "tgt_dataset": args.tgt_dataset,
        },
    }

    out_fp = project_root / args.results_jsonl
    append_jsonl(out_fp, record)

    print(f"[OK] wrote {out_fp}")
    print(
        json.dumps(
            {k: record[k] for k in ["dataset", "scenario", "method", "variant", "acc_mean", "bacc_mean", "macro_f1_mean"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
