from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from pifsl.core.base import DatasetBundle
from pifsl.core.physics_regularization import (
    PhysicsInformedRegularizer,
    PhysicsRegularizationConfig,
)
from pifsl.core.models import RelationNet
from pifsl.runner.train_eval_bosch import evaluate_relation

from pifsl.core.fs_multiclass import EpisodicScalogramSet, Modalities, RawWindow
from pifsl.runner.bench_utils import mean_ci95


@dataclass
class PIFSLArgs:
    device: str = "cpu"
    train_episodes: int = 1000
    eval_episodes: int = 200
    n_way: int = 2
    k_shot: int = 5
    q_query: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0

    # physics
    physics_every: int = 1
    physics_weight: float = 1.0

    # modalities
    modalities: str = "vibration"             # e.g. "vibration,current"
    pad_missing_modalities: str = "zeros"    # zeros | duplicate_first

    # spectral bands override (optional)
    spectral_bands: Optional[Dict[str, List[float]]] = None
    motor_current_enabled: bool = False
    lambda_current: float = 0.1
    current_key: str = "motor_current"
    current_spectral_bands: Optional[Dict[str, List[float]]] = None


def _parse_modalities(args: PIFSLArgs) -> Modalities:
    keys = [k.strip() for k in (args.modalities or "vibration").split(",") if k.strip()]
    if not keys:
        keys = ["vibration"]
    return Modalities(keys=keys, pad_missing=args.pad_missing_modalities)


def run_pi_fsl(
    src_X: List[RawWindow],
    src_y: List[int],
    tgt_X: List[RawWindow],
    tgt_y: List[int],
    fs: float,
    args: PIFSLArgs,
    use_physics: bool = True,
) -> Dict[str, float]:
    device = torch.device(args.device)
    modalities = _parse_modalities(args)

    train_set = EpisodicScalogramSet(
        src_X, src_y, fs=fs, n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, modalities=modalities
    )
    test_set = EpisodicScalogramSet(
        tgt_X, tgt_y, fs=fs, n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, modalities=modalities
    )

    model = RelationNet(in_channels=len(modalities.keys)).to(device)

    phys = None
    if use_physics:
        phys_cfg = PhysicsRegularizationConfig(
            enabled=True,
            lambda_energy=0.1,
            lambda_spectral=0.1,
            lambda_envelope=0.05,
            spectral_bands=args.spectral_bands,
            motor_current_enabled=bool(args.motor_current_enabled),
            lambda_current=float(args.lambda_current),
            current_key=str(args.current_key),
            current_spectral_bands=args.current_spectral_bands,
        )
        phys = PhysicsInformedRegularizer(phys_cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    # Training loop
    for ep in range(1, int(args.train_episodes) + 1):
        Sx, Sy, Qx, Qy, raw_support, raw_support_y = train_set.sample_episode(seed=100000 + ep, return_raw=True)

        Sx = Sx.to(device)
        Sy = Sy.to(device)
        Qx = Qx.to(device)
        Qy = Qy.to(device)

        opt.zero_grad(set_to_none=True)

        scores, classes = model.forward_episode(Sx, Sy, Qx) 
        loss = F.cross_entropy(scores, Qy)

        if phys is not None and args.physics_every > 0 and (ep % int(args.physics_every) == 0):
            # feature maps: use query embeddings as a stable representation
            with torch.no_grad():
                feat_q = model.embed(Qx)

            # Raw signals: pass through as-is; regularizer extracts modalities by key
            raw_signals = []
            for w in raw_support:
                if isinstance(w, dict):
                    raw_signals.append({k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in w.items()})
                else:
                    raw_signals.append(np.asarray(w, dtype=np.float32).reshape(-1))

            phys_loss = phys(
                pred_outputs=(scores, Qy),
                raw_signals=raw_signals,
                labels=raw_support_y,
                feature_maps=feat_q,
                fs=float(fs),
            )
            loss = loss + float(args.physics_weight) * phys_loss

        loss.backward()
        opt.step()

    # Evaluation on target episodes
    eval_stats = evaluate_relation(
        model=model,
        data=test_set,
        K=args.k_shot,
        Q=args.q_query,
        episodes=int(args.eval_episodes),
        seed=123,
        device=str(device),
    )

    out: Dict[str, float] = dict(eval_stats)
    if "acc" in out and isinstance(out["acc"], (float, int)):
        # Try bootstrap CI across episodes by re-evaluating quickly
        accs = []
        for i in range(int(args.eval_episodes)):
            Sx, Sy, Qx, Qy, _, _ = test_set.sample_episode(seed=200000 + i)
            Sx, Sy, Qx, Qy = Sx.to(device), Sy.to(device), Qx.to(device), Qy.to(device)
            with torch.no_grad():
                scores, _ = model.forward_episode(Sx, Sy, Qx)
                pred = scores.argmax(dim=1)
                accs.append(float((pred == Qy).float().mean().item()))
        m, lo, hi = mean_ci95(accs)
        out["acc_mean"] = float(m)
        out["acc_ci95_lo"] = float(lo)
        out["acc_ci95_hi"] = float(hi)

    if phys is not None:
        out.update({f"phys_{k}": float(v) for k, v in phys.metrics().items()})

    return out
