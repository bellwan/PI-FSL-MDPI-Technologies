from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from pifsl.eval.ci import bootstrap
from pifsl.runner.bench_utils import set_seed
from pifsl.eval.schema import append_jsonl, utc_now_iso

from pifsl.data.bosch_drilling.loader import load_bosch_windows, Bundle
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
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z.reshape(z.size(0), -1)  # [B, 64*4*4]


class LabelClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _bundle_to_tensors(bundle: Bundle, tf_rep: str, max_n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(bundle.y))
    rng.shuffle(idx)
    if int(max_n) > 0:
        idx = idx[: int(max_n)]

    X = []
    y = []
    for i in idx:
        sig = np.asarray(bundle.X[i], dtype=np.float32).reshape(-1)
        if str(tf_rep).lower() == "stft":
            S = stftogram_64x64(sig, float(bundle.fs))
        else:
            S = scalogram_64x64(sig, float(bundle.fs))
        X.append(S[None, :, :])  # [1,64,64]
        y.append(int(bundle.y[i]))

    X_t = torch.from_numpy(np.stack(X, axis=0)).float()  # [B,1,64,64]
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    return X_t, y_t


def parse_args():
    ap = argparse.ArgumentParser("DGDA DANN run-one (Bosch drilling) -> append results.jsonl + save npz")
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--dataset", type=str, default="bosch", choices=["bosch"])
    ap.add_argument("--data_root", type=str, required=True)

    ap.add_argument("--source_domain", type=str, required=True)
    ap.add_argument("--target_domain", type=str, required=True)
    ap.add_argument("--tf_rep", type=str, default="cwt", choices=["cwt", "stft"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_eval", type=int, default=2000)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_domain", type=float, default=0.2)

    ap.add_argument("--scenario", type=str, default="")
    ap.add_argument("--exp_id", type=str, default="")
    ap.add_argument("--notes", type=str, default="")

    ap.add_argument("--results_jsonl", type=str, default="artifacts/results/jsonl/dgda_dann_bosch.jsonl")
    ap.add_argument("--save_eval_artifacts_dir", type=str, default="outputs/eval_artifacts")
    return ap.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    bootstrap(project_root)
    set_seed(int(args.seed))

    t0 = time.time()

    src, tgt = load_bosch_windows(
        data_root=str(args.data_root),
        source_domain=str(args.source_domain),
        target_domain=str(args.target_domain),
        normalization="per_window",
    )

    # train tensors
    Xs, ys = _bundle_to_tensors(src, tf_rep=args.tf_rep, max_n=args.max_train, seed=args.seed)
    Xt, _  = _bundle_to_tensors(tgt, tf_rep=args.tf_rep, max_n=args.max_train, seed=args.seed + 13)

    # eval tensors (target labeled)
    Xe, ye = _bundle_to_tensors(tgt, tf_rep=args.tf_rep, max_n=args.max_eval, seed=args.seed + 7)

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
        bs = int(args.batch_size)

        for step in range(n_steps):
            s_idx = perm_s[(step * bs) % n_s : min(((step + 1) * bs) % n_s if ((step + 1) * bs) % n_s != 0 else n_s, n_s)]
            t_idx = perm_t[(step * bs) % n_t : min(((step + 1) * bs) % n_t if ((step + 1) * bs) % n_t != 0 else n_t, n_t)]
            if len(s_idx) == 0: s_idx = perm_s[:bs]
            if len(t_idx) == 0: t_idx = perm_t[:bs]

            xs = Xs[s_idx]
            ys_b = ys[s_idx]
            xt = Xt[t_idx]

            z_s = feat(xs)
            z_t = feat(xt)

            # label loss (source)
            logits_y = clf(z_s)
            loss_y = ce(logits_y, ys_b)

            # domain loss (source=0, target=1)
            z_mix = torch.cat([z_s, z_t], dim=0)
            d_in = grad_reverse(z_mix, float(args.lambda_domain))
            logits_d = dom(d_in)
            d_lab = torch.cat(
                [
                    torch.zeros(z_s.size(0), dtype=torch.long, device=device),
                    torch.ones(z_t.size(0), dtype=torch.long, device=device),
                ],
                dim=0,
            )
            loss_d = ce(logits_d, d_lab)

            loss = loss_y + loss_d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # evaluate on target labeled set
    feat.eval(); clf.eval()
    with torch.no_grad():
        z = feat(Xe)
        logits = clf(z)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

    y_true = ye.cpu().numpy()
    acc = float(np.mean(pred == y_true))
    bacc = float(
        0.5 * ((pred[y_true == 0] == 0).mean() if np.any(y_true == 0) else 0.0)
        + 0.5 * ((pred[y_true == 1] == 1).mean() if np.any(y_true == 1) else 0.0)
    )
    roc_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None
    pr_auc = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None
    cm = confusion_matrix(y_true, pred).tolist()

    # save npz
    save_dir = Path(str(args.save_eval_artifacts_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"eval_bosch_dann_{args.source_domain}_to_{args.target_domain}_seed{int(args.seed)}.npz"
    np.savez_compressed(str(save_dir / npz_name), y_true=y_true, y_pred=pred, y_proba=prob)

    scenario = str(args.scenario).strip() or f"{args.source_domain}_to_{args.target_domain}"
    record: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "dataset": "bosch",
        "scenario": scenario,
        "method": "dann",
        "variant": "dgda",
        "source_domain": str(args.source_domain),
        "target_domain": str(args.target_domain),
        "seed": int(args.seed),
        "input_representation": "scalogram_64x64",
        "tf_rep": str(args.tf_rep),
        "acc_mean": acc,
        "bacc_mean": bacc,
        "macro_f1_mean": None,
        "notes": str(args.notes),
        "extra": {
            "exp_id": str(args.exp_id),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "eval_npz": str((save_dir / npz_name)).replace("\\", "/"),
            "wall_sec": float(time.time() - t0),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "lambda_domain": float(args.lambda_domain),
            "max_train": int(args.max_train),
            "max_eval": int(args.max_eval),
        },
    }

    out_fp = project_root / str(args.results_jsonl)
    append_jsonl(out_fp, record)
    print(f"[OK] wrote {out_fp}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
