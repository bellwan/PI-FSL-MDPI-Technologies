from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))
from pifsl.data.bosch_milling.loader import load_bosch_mi_source


def _split_by_domain(domains: list[str], train_domain: str, test_domain: str) -> Tuple[list[int], list[int]]:
    tr = [i for i,d in enumerate(domains) if str(d)==str(train_domain)]
    te = [i for i,d in enumerate(domains) if str(d)==str(test_domain)]
    if not tr or not te:
        raise ValueError(f"Domain split failed. train_domain={train_domain} n={len(tr)}; test_domain={test_domain} n={len(te)}")
    return tr, te


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64).reshape(-1)
    y_pred = y_pred.astype(np.float64).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R^2 (guarded for constant targets)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    r2 = float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/bosch_milling", help="Bosch milling data root (feature table inside)")
    ap.add_argument("--train_domain", type=str, default="", help="Domain name for training (tool id). If empty, uses first domain found.")
    ap.add_argument("--test_domain", type=str, default="", help="Domain name for test (tool id). If empty, uses second domain found.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="artifacts/results/regression")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    bundle = load_bosch_mi_source(
        data_root=str(args.data_root),
        seed=int(args.seed),
        label_mode="regression",
        prefer_feature_table=True,
    )

    # bundle.X: per-sample feature vectors; bundle.y: scalar regression targets
    X = np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in bundle.X]).astype(np.float32)
    y = np.asarray(bundle.y, dtype=np.float32).reshape(-1)

    domains = [str(d) for d in bundle.domain]
    uniq = sorted(set(domains))
    if len(uniq) < 2:
        raise ValueError(f"Need at least 2 domains for a cross-domain regression test. Found domains={uniq}")

    train_dom = args.train_domain or uniq[0]
    test_dom  = args.test_domain or uniq[1]

    tr_idx, te_idx = _split_by_domain(domains, train_dom, test_dom)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(in_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    model.train()
    for ep in range(int(args.epochs)):
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        yhat = model(Xte_t).cpu().numpy()

    met = _metrics(yte, yhat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-sample predictions (CSV)
    csv_path = out_dir / f"bosch_milling_regression_{train_dom}_to_{test_dom}_seed{int(args.seed)}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("y_true,y_pred\n")
        for a,b in zip(yte.tolist(), yhat.tolist()):
            f.write(f"{float(a)},{float(b)}\n")

    # Run summary (JSON)
    js_path = out_dir / f"bosch_milling_regression_{train_dom}_to_{test_dom}_seed{int(args.seed)}.json"
    payload: Dict[str, Any] = {
        "dataset": "bosch_milling",
        "task": "regression_ctf_norm",
        "train_domain": train_dom,
        "test_domain": test_dom,
        "seed": int(args.seed),
        "metrics": met,
        "csv": str(csv_path).replace("\\", "/"),
    }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
