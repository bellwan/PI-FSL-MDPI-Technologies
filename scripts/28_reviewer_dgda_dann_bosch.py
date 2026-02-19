from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))
from pifsl.data.bosch_drilling.loader import load_bosch_windows
from pifsl.data.bosch_drilling.raw_processing.gating import scalogram_64x64, stftogram_64x64


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class FeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)  # [B,64,4,4]
        return z.view(z.size(0), -1)


class LabelClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 2),  # src vs tgt
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _to_img(sig: np.ndarray, fs: float, tf_rep: str) -> np.ndarray:
    if str(tf_rep).lower() == "stft":
        return stftogram_64x64(sig, fs)
    return scalogram_64x64(sig, fs)


def _bundle_to_tensors(bundle, tf_rep: str, max_n: int | None, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(bundle.y))
    if max_n is not None and len(idx) > int(max_n):
        idx = rng.choice(idx, size=int(max_n), replace=False)
    xs = []
    ys = []
    for i in idx.tolist():
        w = bundle.X[i]
        # Raw window may be multi-channel dict; uses first available channel.
        if isinstance(w, dict):
            sig = np.asarray(next(iter(w.values())), dtype=np.float32).reshape(-1)
        else:
            sig = np.asarray(w, dtype=np.float32).reshape(-1)
        img = _to_img(sig, float(bundle.fs), tf_rep=tf_rep)
        xs.append(img[None, ...])  # [1,64,64]
        ys.append(int(bundle.y[i]))
    X = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.asarray(ys, dtype=np.int64))
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/bosch_drilling")
    ap.add_argument("--source_domain", type=str, default="M01_OP05")
    ap.add_argument("--target_domain", type=str, default="M02_OP05")
    ap.add_argument("--tf_rep", type=str, default="cwt", choices=["cwt","stft"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_eval", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_domain", type=float, default=0.2)
    ap.add_argument("--out_dir", type=str, default="artifacts/results/dgda")
    ap.add_argument("--save_npz_dir", type=str, default="outputs/eval_artifacts")
    args = ap.parse_args()

    t0 = time.time()

    src, tgt = load_bosch_windows(
        data_root=str(args.data_root),
        source_domain=str(args.source_domain),
        target_domain=str(args.target_domain),
        normalization="per_window",
    )

    Xs, ys = _bundle_to_tensors(src, tf_rep=args.tf_rep, max_n=args.max_train, seed=args.seed)
    Xt, yt = _bundle_to_tensors(tgt, tf_rep=args.tf_rep, max_n=args.max_train, seed=args.seed+13)

    # Evaluation uses labeled target subset (supervised eval on target domain).
    Xe, ye = _bundle_to_tensors(tgt, tf_rep=args.tf_rep, max_n=args.max_eval, seed=args.seed+7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs, ys = Xs.to(device), ys.to(device)
    Xt = Xt.to(device)
    Xe = Xe.to(device)

    feat = FeatureExtractor(in_ch=1).to(device)
    in_dim = 64 * 4 * 4
    clf = LabelClassifier(in_dim=in_dim, n_classes=2).to(device)
    dom = DomainClassifier(in_dim=in_dim).to(device)

    opt = optim.Adam(list(feat.parameters()) + list(clf.parameters()) + list(dom.parameters()), lr=float(args.lr))
    ce = nn.CrossEntropyLoss()

    n_s = Xs.size(0)
    n_t = Xt.size(0)
    rng = np.random.RandomState(int(args.seed))

    feat.train(); clf.train(); dom.train()
    for ep in range(int(args.epochs)):
        perm_s = rng.permutation(n_s)
        perm_t = rng.permutation(n_t)
        n_steps = max(int(np.ceil(n_s / args.batch_size)), int(np.ceil(n_t / args.batch_size)))
        for step in range(n_steps):
            bs = int(args.batch_size)
            s_idx = perm_s[(step*bs) % n_s : min(((step+1)*bs) % n_s if ((step+1)*bs)%n_s!=0 else n_s, n_s)]
            t_idx = perm_t[(step*bs) % n_t : min(((step+1)*bs) % n_t if ((step+1)*bs)%n_t!=0 else n_t, n_t)]
            if len(s_idx)==0:
                s_idx = perm_s[:bs]
            if len(t_idx)==0:
                t_idx = perm_t[:bs]

            xs = Xs[s_idx]
            ys_b = ys[s_idx]
            xt = Xt[t_idx]

            z_s = feat(xs)
            z_t = feat(xt)

            logits_y = clf(z_s)
            loss_y = ce(logits_y, ys_b)

            # Domain labels: source=0, target=1.
            z_mix = torch.cat([z_s, z_t], dim=0)
            d_in = grad_reverse(z_mix, float(args.lambda_domain))
            logits_d = dom(d_in)
            d_lab = torch.cat([torch.zeros(z_s.size(0), dtype=torch.long, device=device),
                               torch.ones(z_t.size(0), dtype=torch.long, device=device)], dim=0)
            loss_d = ce(logits_d, d_lab)

            loss = loss_y + loss_d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    feat.eval(); clf.eval()
    with torch.no_grad():
        z = feat(Xe)
        logits = clf(z)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

    y_true = ye.numpy()
    acc = float(np.mean(pred == y_true))
    bacc = float(0.5 * ((pred[y_true==0]==0).mean() if np.any(y_true==0) else 0.0) +
                 0.5 * ((pred[y_true==1]==1).mean() if np.any(y_true==1) else 0.0))
    roc_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true))==2 else None
    pr_auc  = float(average_precision_score(y_true, prob)) if len(np.unique(y_true))==2 else None
    cm = confusion_matrix(y_true, pred).tolist()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    save_npz = Path(args.save_npz_dir); save_npz.mkdir(parents=True, exist_ok=True)
    npz_name = f"eval_bosch_dann_{args.source_domain}_to_{args.target_domain}_seed{int(args.seed)}.npz"
    np.savez_compressed(str(save_npz / npz_name), y_true=y_true, y_pred=pred, y_proba=prob)

    payload: Dict[str, Any] = {
        "dataset": "bosch",
        "baseline": "dann",
        "source_domain": str(args.source_domain),
        "target_domain": str(args.target_domain),
        "tf_rep": str(args.tf_rep),
        "seed": int(args.seed),
        "metrics": {
            "acc": acc,
            "bacc": bacc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
        },
        "eval_npz": str((save_npz / npz_name)).replace("\\", "/"),
        "wall_sec": float(time.time() - t0),
    }

    js_path = out_dir / f"bosch_dann_{args.source_domain}_to_{args.target_domain}_seed{int(args.seed)}.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
