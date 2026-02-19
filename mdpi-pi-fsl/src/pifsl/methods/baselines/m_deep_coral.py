from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pifsl.core.models import ConvEmbedding
from pifsl.core.utils import compute_metrics
from pifsl.runner.bench_utils import set_seed

RawWindow = Union[np.ndarray, Dict[str, np.ndarray]]

@dataclass
class DeepCORALArgs:
    device: str = "cpu"
    sup_epochs: int = 10
    sup_batch_size: int = 32
    sup_lr: float = 1e-3

    coral_steps: int = 200
    coral_batch_size: int = 64
    coral_lr: float = 5e-4
    coral_weight: float = 1.0

    # If true, target-unlabeled set is restricted to the episode target-support subset.
    coral_use_target_support_only: bool = True


def _coral_loss(src_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
    if src_feat.ndim != 2 or tgt_feat.ndim != 2:
        raise ValueError("Features must be 2D (N,D)")
    ns = src_feat.size(0)
    nt = tgt_feat.size(0)
    if ns < 2 or nt < 2:
        return torch.tensor(0.0, device=src_feat.device)

    src = src_feat - src_feat.mean(dim=0, keepdim=True)
    tgt = tgt_feat - tgt_feat.mean(dim=0, keepdim=True)

    cs = (src.t() @ src) / (ns - 1)
    ct = (tgt.t() @ tgt) / (nt - 1)

    loss = (cs - ct).pow(2).mean()
    return loss


def run_deep_coral_episode(
    enc: ConvEmbedding,
    head: nn.Linear,
    src_x: torch.Tensor,
    src_y: torch.Tensor,
    tgt_x_unl: torch.Tensor,
    args: DeepCORALArgs,
) -> None:
    device = torch.device(args.device)
    enc.train(); head.train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=float(args.coral_lr))

    for _ in range(int(args.coral_steps)):
        # Independent minibatch sampling per alignment step.
        bs_s = min(int(args.coral_batch_size), src_x.size(0))
        bs_t = min(int(args.coral_batch_size), tgt_x_unl.size(0))
        if bs_s < 2 or bs_t < 2:
            return
        idx_s = torch.randint(0, src_x.size(0), (bs_s,), device=device)
        idx_t = torch.randint(0, tgt_x_unl.size(0), (bs_t,), device=device)

        xs = src_x[idx_s]
        ys = src_y[idx_s]
        xt = tgt_x_unl[idx_t]

        fs = enc(xs).reshape(xs.size(0), -1)
        ft = enc(xt).reshape(xt.size(0), -1)

        logits = head(fs)
        cls_loss = F.cross_entropy(logits, ys)
        coral = _coral_loss(fs, ft)
        loss = cls_loss + float(args.coral_weight) * coral

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def run_deep_coral_supervised_source(
    enc: ConvEmbedding,
    head: nn.Linear,
    src_x: torch.Tensor,
    src_y: torch.Tensor,
    args: DeepCORALArgs,
) -> None:
    device = torch.device(args.device)
    enc.train(); head.train()
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=float(args.sup_lr))

    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    for _ep in range(int(args.sup_epochs)):
        steps = max(1, int(np.ceil(src_x.size(0) / float(args.sup_batch_size))))
        for _ in range(steps):
            bs = min(int(args.sup_batch_size), src_x.size(0))
            idx = torch.randint(0, src_x.size(0), (bs,), generator=rng, device=device)
            xb = src_x[idx]
            yb = src_y[idx]
            feat = enc(xb).reshape(xb.size(0), -1)
            logits = head(feat)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def evaluate_logits(
    enc: ConvEmbedding,
    head: nn.Linear,
    x: torch.Tensor,
    y: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    enc.eval(); head.eval()
    dev = torch.device(device)
    with torch.no_grad():
        feat = enc(x.to(dev)).reshape(x.size(0), -1)
        logits = head(feat)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        proba = None
        if logits.size(1) == 2:
            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return np.asarray(y, dtype=np.int64), pred, proba


def compute_all_basic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = compute_metrics(y_true, y_pred)
    return {"acc": float(m["acc"]), "bacc": float(m["bacc"]), "macro_f1": float(m["macro_f1"])}
