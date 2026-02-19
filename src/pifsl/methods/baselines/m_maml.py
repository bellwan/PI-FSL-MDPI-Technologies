from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from pifsl.core.utils import compute_metrics

@dataclass
class MAMLArgs:
    device: str = "cpu"
    train_episodes: int = 4000
    eval_episodes: int = 200

    n_way: int = 2
    k_shot: int = 5
    q_query: int = 16

    # First-order MAML (no second-order gradients).
    inner_steps: int = 5
    inner_lr: float = 0.01
    outer_lr: float = 1e-3
    weight_decay: float = 0.0

    grad_clip: float = 5.0

    z_norm: bool = True
    z_norm_eps: float = 1e-6


class _MAML1DNet(nn.Module):
    def __init__(self, input_len: int, n_way: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            x = torch.zeros(1, 1, input_len)
            z = self.conv(x)
            feat_dim = int(z.reshape(1, -1).shape[1])

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_way),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.conv(x).reshape(x.size(0), -1)
        return self.head(z)


def _sample_nway_episode(
    X: List[np.ndarray],
    y: List[int],
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if y.size == 0:
        raise ValueError("Empty y for episode sampling")

    classes = np.unique(y)
    if classes.size < n_way:
        raise ValueError(f"Not enough classes for {n_way}-way episode: have {classes.size}")

    chosen = rng.choice(classes, size=n_way, replace=False)

    sup_x, sup_y, qry_x, qry_y = [], [], [], []
    for new_id, c in enumerate(chosen):
        idx = np.where(y == c)[0]
        need = k_shot + q_query
        pick = rng.choice(idx, size=need, replace=(idx.size < need))
        sup_pick, qry_pick = pick[:k_shot], pick[k_shot:]
        for i in sup_pick:
            sup_x.append(X[int(i)])
            sup_y.append(new_id)
        for i in qry_pick:
            qry_x.append(X[int(i)])
            qry_y.append(new_id)

    Sx = np.stack([np.asarray(a, dtype=np.float32).reshape(-1) for a in sup_x], axis=0)
    Qx = np.stack([np.asarray(a, dtype=np.float32).reshape(-1) for a in qry_x], axis=0)
    Sy = np.asarray(sup_y, dtype=np.int64)
    Qy = np.asarray(qry_y, dtype=np.int64)

    s_perm = rng.permutation(len(Sy))
    q_perm = rng.permutation(len(Qy))
    return Sx[s_perm], Sy[s_perm], Qx[q_perm], Qy[q_perm]


def run_maml(
    src_X: List[np.ndarray],
    src_y: List[int],
    tgt_X: List[np.ndarray],
    tgt_y: List[int],
    args: MAMLArgs,
) -> dict:
    device = torch.device(args.device)

    if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
        raise RuntimeError(
            "MAML requires gradients but torch.inference_mode() is enabled in the caller. "
            "Remove inference_mode around the MAML run."
        )

    if len(src_X) == 0:
        raise ValueError("Empty src_X")

    input_len = int(np.asarray(src_X[0]).reshape(-1).shape[0])
    n_way = int(args.n_way)

    model = _MAML1DNet(input_len=input_len, n_way=n_way).to(device)
    outer_opt = torch.optim.Adam(
        model.parameters(),
        lr=float(args.outer_lr),
        weight_decay=float(args.weight_decay),
    )

    def _to_tensor(a: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
        if args.z_norm:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(args.z_norm_eps))
            x = (x - mu) / sd
        return x

    # Meta-train on source episodes.
    with torch.enable_grad():
        model.train()
        for ep in range(int(args.train_episodes)):
            Sx, Sy, Qx, Qy = _sample_nway_episode(
                src_X, src_y, n_way, int(args.k_shot), int(args.q_query), seed=1000 + ep
            )
            Sx_t = _to_tensor(Sx)
            Qx_t = _to_tensor(Qx)
            Sy_t = torch.from_numpy(Sy).long().to(device)
            Qy_t = torch.from_numpy(Qy).long().to(device)

            fast_params = dict(model.named_parameters())

            for _ in range(int(args.inner_steps)):
                logits_s = functional_call(model, fast_params, (Sx_t,))
                loss_s = F.cross_entropy(logits_s, Sy_t)

                grads = torch.autograd.grad(
                    loss_s,
                    tuple(fast_params.values()),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                fast_params = {
                    k: v - float(args.inner_lr) * g
                    for (k, v), g in zip(fast_params.items(), grads)
                }

            logits_q = functional_call(model, fast_params, (Qx_t,))
            loss_q = F.cross_entropy(logits_q, Qy_t)

            outer_opt.zero_grad(set_to_none=True)
            loss_q.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            outer_opt.step()


    # Meta-test: adapt on target support, evaluate on target query.
    model.eval()
    accs, baccs, f1s = [], [], []

    for ep in range(int(args.eval_episodes)):
        Sx, Sy, Qx, Qy = _sample_nway_episode(
            tgt_X, tgt_y, n_way, int(args.k_shot), int(args.q_query), seed=7777 + ep
        )
        Sx_t = _to_tensor(Sx)
        Qx_t = _to_tensor(Qx)
        Sy_t = torch.from_numpy(Sy).long().to(device)

        with torch.enable_grad():
            fast_params = dict(model.named_parameters())
            for _ in range(int(args.inner_steps)):
                logits_s = functional_call(model, fast_params, (Sx_t,))
                loss_s = F.cross_entropy(logits_s, Sy_t)
                grads = torch.autograd.grad(
                    loss_s,
                    tuple(fast_params.values()),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                fast_params = {
                    k: v - float(args.inner_lr) * g
                    for (k, v), g in zip(fast_params.items(), grads)
                }

        with torch.no_grad():
            logits_q = functional_call(model, fast_params, (Qx_t,))
            pred = torch.argmax(logits_q, dim=1).detach().cpu().numpy()
            y_true = np.asarray(Qy, dtype=np.int64)
            m = compute_metrics(y_true, pred)
            accs.append(float(m["acc"]))
            baccs.append(float(m["bacc"]))
            f1s.append(float(m["macro_f1"]))

    acc_m, acc_ci = (float(np.mean(accs)), float("nan")) if len(accs) < 2 else compute_metrics(np.ones(2), np.ones(2))["acc"], 0.0
    acc_m, acc_ci = compute_metrics(np.asarray(accs), np.asarray(accs))["acc"], 0.0 if False else (float(np.mean(accs)), float("nan"))
    m = compute_metrics(np.asarray(accs), np.asarray(accs)) if False else None
    acc_m, acc_ci = (float(np.mean(accs)), float("nan")) if len(accs) < 2 else (float(np.mean(accs)), 1.96 * float(np.std(accs, ddof=1) / math.sqrt(len(accs))))
    bacc_m, bacc_ci = (float(np.mean(baccs)), float("nan")) if len(baccs) < 2 else (float(np.mean(baccs)), 1.96 * float(np.std(baccs, ddof=1) / math.sqrt(len(baccs))))
    f1_m, f1_ci = (float(np.mean(f1s)), float("nan")) if len(f1s) < 2 else (float(np.mean(f1s)), 1.96 * float(np.std(f1s, ddof=1) / math.sqrt(len(f1s))))
    return {
        "acc_mean": acc_m,
        "acc_ci95": acc_ci,
        "bacc_mean": bacc_m,
        "bacc_ci95": bacc_ci,
        "macro_f1_mean": f1_m,
        "macro_f1_ci95": f1_ci,
    }
