from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverseFn.apply(x, lambd)


class SimpleConvEncoder(nn.Module):

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class ClassHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class DomainHead(nn.Module):
    def __init__(self, in_dim: int, n_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, n_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class DANNConfig:
    feat_dim: int = 128
    lambda_grl: float = 0.5
    lambda_domain: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 200


class DANNTrainer:

    def __init__(self, n_classes: int, device: torch.device, cfg: Optional[DANNConfig] = None):
        self.cfg = cfg or DANNConfig()
        self.device = device
        self.encoder = SimpleConvEncoder(out_dim=self.cfg.feat_dim).to(device)
        self.clf = ClassHead(self.cfg.feat_dim, n_classes).to(device)
        self.dom = DomainHead(self.cfg.feat_dim, n_domains=2).to(device)

        params = list(self.encoder.parameters()) + list(self.clf.parameters()) + list(self.dom.parameters())
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def train(self, xs: torch.Tensor, ys: torch.Tensor, xt: torch.Tensor) -> Dict[str, float]:
        # xs, xt: [B, 1, 64, 64]; ys: [B]
        self.encoder.train(); self.clf.train(); self.dom.train()

        # Domain labels: source=0, target=1.
        ds = torch.zeros(xs.size(0), dtype=torch.long, device=self.device)
        dt = torch.ones(xt.size(0), dtype=torch.long, device=self.device)

        # Fixed-step optimization on provided batches.
        last = {}
        for _ in range(int(self.cfg.steps)):
            self.opt.zero_grad(set_to_none=True)

            zs = self.encoder(xs)
            zt = self.encoder(xt)

            logits_y = self.clf(zs)
            loss_cls = F.cross_entropy(logits_y, ys)

            z_all = torch.cat([zs, zt], dim=0)
            d_all = torch.cat([ds, dt], dim=0)
            z_rev = grad_reverse(z_all, self.cfg.lambda_grl)
            logits_d = self.dom(z_rev)
            loss_dom = F.cross_entropy(logits_d, d_all)

            loss = loss_cls + self.cfg.lambda_domain * loss_dom
            loss.backward()
            self.opt.step()

            last = {
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss_dom": float(loss_dom.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
            }
        return last

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.encoder.eval(); self.clf.eval()
        z = self.encoder(x)
        logits = self.clf(z)
        proba = torch.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1)
        return pred, proba
