from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch.func import functional_call  
except Exception:
    from torch.nn.utils.stateless import functional_call 
from pifsl.methods.baselines.cwru_models import CNN1D

from pifsl.core.base import DatasetBundle
from pifsl.core.utils import mean_ci95, compute_metrics

@dataclass
class MAMLConfig:
    device: str = "cpu"
    train_episodes: int = 500
    eval_episodes: int = 200
    n_way: int = 2
    k_shot: int = 1
    q_query: int = 16
    inner_lr: float = 0.01
    outer_lr: float = 1e-3
    inner_steps: int = 1

def _prep_batch(X: List[np.ndarray], idx: np.ndarray, device) -> torch.Tensor:
    arr = np.stack([X[int(i)] for i in idx]).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(1).to(device)  # [B, 1, L]

def _sample_binary_episode(X: List[np.ndarray], y: List[int], k: int, q: int, seed: int):
    rng = np.random.RandomState(seed)
    pos = np.where(np.asarray(y) == 1)[0]
    neg = np.where(np.asarray(y) == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Need both classes for binary MAML episode.")
    take_pos = rng.choice(pos, size=k+q, replace=(len(pos) < (k+q)))
    take_neg = rng.choice(neg, size=k+q, replace=(len(neg) < (k+q)))
    sup = np.r_[take_pos[:k], take_neg[:k]]
    qry = np.r_[take_pos[k:], take_neg[k:]]
    Sy = np.array([1]*k + [0]*k, dtype=np.int64)
    Qy = np.array([1]*q + [0]*q, dtype=np.int64)
    return sup, Sy, qry, Qy

def run_maml(source: DatasetBundle, target: DatasetBundle, cfg: MAMLConfig) -> Dict[str, float]:
    if cfg.n_way != 2:
        raise ValueError("This MAML baseline supports binary only (n_way=2).")

    device = torch.device(cfg.device)
    model = CNN1D(output_size=cfg.n_way).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)

    def named_params():
        return dict(model.named_parameters())

    # Meta-train (first-order MAML) on source episodes.
    model.train()
    for ep in range(int(cfg.train_episodes)):
        sup_ix, Sy_np, qry_ix, Qy_np = _sample_binary_episode(source.X, source.y, cfg.k_shot, cfg.q_query, seed=ep)
        Sx = _prep_batch(source.X, sup_ix, device)
        Qx = _prep_batch(source.X, qry_ix, device)
        Sy = torch.from_numpy(Sy_np).to(device)
        Qy = torch.from_numpy(Qy_np).to(device)

        params = named_params()
        for _ in range(cfg.inner_steps):
            logits = functional_call(model, params, (Sx,))
            loss_s = F.cross_entropy(logits, Sy)
            grads = torch.autograd.grad(loss_s, params.values(), create_graph=False)
            params = {k: v - cfg.inner_lr * g.detach() for (k, v), g in zip(params.items(), grads)}

        logits_q = functional_call(model, params, (Qx,))
        loss_q = F.cross_entropy(logits_q, Qy)

        opt.zero_grad(set_to_none=True)
        loss_q.backward()
        opt.step()

    # Meta-test: adapt on target support, evaluate on target query.
    model.eval()
    accs, baccs, f1s = [], [], []

    for ep in range(int(cfg.eval_episodes)):
        sup_ix, Sy_np, qry_ix, Qy_np = _sample_binary_episode(
            target.X, target.y, cfg.k_shot, cfg.q_query, seed=10_000 + ep
        )
        Sx = _prep_batch(target.X, sup_ix, device)
        Qx = _prep_batch(target.X, qry_ix, device)
        Sy = torch.from_numpy(Sy_np).to(device)
        Qy = torch.from_numpy(Qy_np).to(device)

        # Inner-loop adaptation requires gradients.
        params = named_params()
        with torch.enable_grad():
            for _ in range(cfg.inner_steps):
                logits = functional_call(model, params, (Sx,))
                loss_s = F.cross_entropy(logits, Sy)
                grads = torch.autograd.grad(loss_s, params.values(), create_graph=False)
                params = {k: v - cfg.inner_lr * g.detach() for (k, v), g in zip(params.items(), grads)}

        # Query evaluation can be torch.no_grad().
        with torch.no_grad():
            logits_q = functional_call(model, params, (Qx,))
            y_pred = torch.argmax(logits_q, dim=1).cpu().numpy()
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
