from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from pifsl.core.base import DatasetBundle
from pifsl.core.utils import mean_ci95, compute_metrics
from pifsl.core.fs_multiclass import sample_episode, sample_episode_with_indices
from pifsl.data.bosch_drilling.raw_processing.gating import scalogram_64x64
from pifsl.core.models import RelationNet
from pifsl.core.physics_regularization import PhysicsInformedRegularizer
from pifsl.runner.config_loader import PhysicsRegularizationConfig

@dataclass
class PIFSLConfig:
    device: str = "cpu"
    train_episodes: int = 500
    eval_episodes: int = 200
    n_way: int = 2
    k_shot: int = 1
    q_query: int = 16
    lr: float = 1e-3
    wd: float = 0.0
    physics_weight: float = 1.0

def _to_scalogram_tensor(signals: List[np.ndarray], fs: float) -> torch.Tensor:
    # Reuse Bosch scalogram implementation (expects 1D float32)

    batch = []
    for sig in signals:
        S = scalogram_64x64(np.asarray(sig, dtype=np.float32).reshape(-1), float(fs))
        batch.append(torch.from_numpy(S[None, ...]).float())
    return torch.stack(batch, dim=0)  # [B,1,64,64]

def run_pi_fsl(
    source: DatasetBundle,
    target: DatasetBundle,
    cfg: PIFSLConfig,
    use_physics: bool = True,
) -> Dict[str, float]:

    device = torch.device(cfg.device)

    model = RelationNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    phys = None
    if use_physics:
        # PhysicsRegularizationConfig in this repo is a plain class (not a dataclass),
        pr_cfg = PhysicsRegularizationConfig()
        pr_cfg.enabled = True

        # Match fields that actually exist in bosch/experiments/config_loader.py
        pr_cfg.lambda_energy = 0.1
        pr_cfg.lambda_spectral = 0.1
        pr_cfg.lambda_envelope = 0.05

        # __post_init__ exists in the class but is not auto-called (not a dataclass)
        if hasattr(pr_cfg, "__post_init__"):
            pr_cfg.__post_init__()

        phys = PhysicsInformedRegularizer(pr_cfg).to(device)

    raw_src = [np.asarray(x, dtype=np.float32).reshape(-1) for x in source.X]
    y_src_phys = [0 if int(v) == 0 else 1 for v in source.y]

    # ---- train on SOURCE episodes
    model.train()
    for ep in range(int(cfg.train_episodes)):
        sup_X, sup_y, qry_X, qry_y, sup_idx, qry_idx = sample_episode_with_indices(
            source.X, source.y, cfg.n_way, cfg.k_shot, cfg.q_query, seed=ep
        )
        Sx = _to_scalogram_tensor(sup_X, source.fs).to(device)
        Qx = _to_scalogram_tensor(qry_X, source.fs).to(device)
        Sy = torch.from_numpy(sup_y).long().to(device)
        Qy = torch.from_numpy(qry_y).long().to(device)

        scores, classes = model.forward_episode(Sx, Sy, Qx)
        loss = F.cross_entropy(scores, Qy)

        if use_physics and phys is not None:
            # physics regularizer expects batch-aligned:
            # raw_signals length == feature_maps batch size == labels length
            episode_raw = [np.asarray(x, dtype=np.float32).reshape(-1) for x in (sup_X + qry_X)]
            episode_labels = [int(source.y[int(i)]) for i in list(sup_idx) + list(qry_idx)]

            # Use the model embedding as feature_maps 
            all_imgs = torch.cat([Sx, Qx], dim=0)          # [Ns+Nq, 1, 64, 64]
            feature_maps = model.emb(all_imgs)             # [Ns+Nq, C, h, w]

            loss_phys = phys(
                pred_outputs=(scores, classes),
                raw_signals=episode_raw,
                labels=episode_labels,
                feature_maps=feature_maps,
                fs=float(source.fs),
            )
            loss = loss + cfg.physics_weight * loss_phys


        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # ---- eval on TARGET episodes
    model.eval()
    accs, baccs, f1s = [], [], []
    with torch.no_grad():
        for ep in range(int(cfg.eval_episodes)):
            sup_X, sup_y, qry_X, qry_y, sup_idx, qry_idx = sample_episode_with_indices(
                target.X, target.y, cfg.n_way, cfg.k_shot, cfg.q_query, seed=10_000 + ep
            )
            Sx = _to_scalogram_tensor(sup_X, target.fs).to(device)
            Qx = _to_scalogram_tensor(qry_X, target.fs).to(device)
            Sy = torch.from_numpy(sup_y).long().to(device)
            Qy = torch.from_numpy(qry_y).long().to(device)

            scores, _ = model.forward_episode(Sx, Sy, Qx)
            y_pred = torch.argmax(scores, dim=1).cpu().numpy()
            y_true = Qy.cpu().numpy()

            m = compute_metrics(y_true, y_pred)
            accs.append(m["acc"]); baccs.append(m["bacc"]); f1s.append(m["macro_f1"])

    acc_m, acc_ci = mean_ci95(accs)
    bacc_m, bacc_ci = mean_ci95(baccs)
    f1_m, f1_ci = mean_ci95(f1s)

    return {
        "acc_mean": acc_m, "acc_ci95": acc_ci,
        "bacc_mean": bacc_m, "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m, "macro_f1_ci95": f1_ci,
    }
