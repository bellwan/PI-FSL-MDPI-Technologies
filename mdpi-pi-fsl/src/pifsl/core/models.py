from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, channels: List[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [32, 32, 64, 64]

        C = [in_channels] + list(channels)
        layers: List[nn.Module] = []
        for i in range(4):
            layers += [
                nn.Conv2d(C[i], C[i + 1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(C[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RelationModule(nn.Module):
    def __init__(self, feat_channels: int):
        super().__init__()
        feat_dim = feat_channels * 4 * 4
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat_q: torch.Tensor, feat_p: torch.Tensor) -> torch.Tensor:
        Nq = feat_q.size(0)
        Np = feat_p.size(0)

        fq = feat_q.reshape(Nq, -1)
        fp = feat_p.reshape(Np, -1)

        fq2 = fq.unsqueeze(1).expand(Nq, Np, fq.size(1))
        fp2 = fp.unsqueeze(0).expand(Nq, Np, fp.size(1))
        pair = torch.cat([fq2, fp2], dim=-1)
        scores = self.mlp(pair).squeeze(-1)
        return scores


class RelationNet(nn.Module):
    def __init__(self, in_channels: int = 1, channels: List[int] | None = None):
        super().__init__()
        if channels is None:
            channels = [32, 32, 64, 64]
        self.embed = ConvEmbedding(in_channels=in_channels, channels=channels)
        self.emb = self.embed 
        self.relation = RelationModule(feat_channels=channels[-1])

    @torch.no_grad()
    def _compute_prototypes(
        self,
        feat_s: torch.Tensor,
        y_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        classes = torch.unique(y_s)
        protos = []
        for c in classes:
            mask = (y_s == c)
            if mask.sum() == 0:
                continue
            protos.append(feat_s[mask].mean(dim=0, keepdim=True))
        if len(protos) == 0:
            # fallback: mean of everything
            protos = [feat_s.mean(dim=0, keepdim=True)]
            classes = torch.tensor([0], device=y_s.device, dtype=y_s.dtype)
        prototypes = torch.cat(protos, dim=0)
        return prototypes, classes

    def forward_episode(
        self,
        Sx: torch.Tensor,
        Sy: torch.Tensor,
        Qx: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        feat_s = self.embed(Sx)
        feat_q = self.embed(Qx)

        prototypes, classes_t = self._compute_prototypes(feat_s, Sy)
        scores = self.relation(feat_q, prototypes)

        # Return python list of int class labels for compatibility with evaluate_relation()
        classes = [int(c.item()) for c in classes_t]
        return scores, classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.embed(x)
