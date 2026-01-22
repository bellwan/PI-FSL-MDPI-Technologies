
from __future__ import annotations
import numpy as np
import torch
from .gating import scalogram_64x64
from pifsl.data.bosch_drilling import config
import numpy as _np

class FewShotScalogramSet(torch.utils.data.Dataset):
    def __init__(self, X_windows, y, fs: float):
        self.fs = fs
        self.y = _np.asarray(y, dtype=_np.int64)
        self.X = X_windows
        self.pos_ix = _np.where(self.y==1)[0].tolist()
        self.neg_ix = _np.where(self.y==0)[0].tolist()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        sig = self.X[int(idx)]; lab = int(self.y[int(idx)])
        S = scalogram_64x64(sig, self.fs)
        return torch.from_numpy(S[None,...]), torch.tensor(lab, dtype=torch.long)
    def sample_episode(self, K:int, Q:int, seed:int=None):

        if len(self.pos_ix)==0 or len(self.neg_ix)==0:
            raise RuntimeError("FewShotScalogramSet: one class has zero samples")
        rng = _np.random.RandomState(seed if seed is not None else _np.random.randint(0,1<<31))
        pos = rng.choice(self.pos_ix, size=(K+Q), replace=(len(self.pos_ix)<(K+Q)))
        neg = rng.choice(self.neg_ix, size=(K+Q), replace=(len(self.neg_ix)<(K+Q)))
        pos_s, pos_q = pos[:K], pos[K:K+Q]
        neg_s, neg_q = neg[:K], neg[K:K+Q]
        sup_ix = _np.asarray(_np.r_[pos_s, neg_s], dtype=_np.int64)
        qry_ix = _np.asarray(_np.r_[pos_q, neg_q], dtype=_np.int64)
        Sx = torch.stack([self[int(i)][0] for i in sup_ix.tolist()]).float()
        Sy = torch.tensor([int(self.y[int(i)]) for i in sup_ix.tolist()], dtype=torch.long)
        Qx = torch.stack([self[int(i)][0] for i in qry_ix.tolist()]).float()
        Qy = torch.tensor([int(self.y[int(i)]) for i in qry_ix.tolist()], dtype=torch.long)
        return Sx, Sy, Qx, Qy
