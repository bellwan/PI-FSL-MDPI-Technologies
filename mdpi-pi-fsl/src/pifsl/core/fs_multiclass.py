from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch

from pifsl.data.bosch_drilling.raw_processing.gating import scalogram_64x64

RawWindow = Union[np.ndarray, Dict[str, np.ndarray]]


def build_class_index(y: List[int]) -> Dict[int, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.int64)
    out: Dict[int, np.ndarray] = {}
    for c in np.unique(y_arr):
        out[int(c)] = np.where(y_arr == c)[0]
    return out


def sample_episode(
    X: List[RawWindow],
    y: List[int],
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
) -> Tuple[List[RawWindow], np.ndarray, List[RawWindow], np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = build_class_index(y)

    candidates = [c for c, ix in idx.items() if len(ix) >= (k_shot + q_query)]
    if len(candidates) < n_way:
        candidates = list(idx.keys())

    chosen = rng.choice(candidates, size=n_way, replace=(len(candidates) < n_way))

    sup_X: List[RawWindow] = []
    sup_y: List[int] = []
    qry_X: List[RawWindow] = []
    qry_y: List[int] = []

    for cls_i, c in enumerate(chosen):
        ix = idx[int(c)]
        take = rng.choice(ix, size=k_shot + q_query, replace=(len(ix) < (k_shot + q_query)))
        sup_ix = take[:k_shot]
        qry_ix = take[k_shot:]
        for j in sup_ix:
            sup_X.append(X[int(j)])
            sup_y.append(cls_i)
        for j in qry_ix:
            qry_X.append(X[int(j)])
            qry_y.append(cls_i)

    return sup_X, np.asarray(sup_y, dtype=np.int64), qry_X, np.asarray(qry_y, dtype=np.int64)


@dataclass
class Modalities:
    keys: List[str]
    pad_missing: str = "zeros"  


class EpisodicScalogramSet(torch.utils.data.Dataset):
    def __init__(
        self,
        X_windows: List[RawWindow],
        y: List[int],
        fs: float,
        n_way: int = 2,
        k_shot: int = 5,
        q_query: int = 16,
        modalities: Optional[Modalities] = None,
    ):
        super().__init__()
        self.fs = float(fs)
        self.X = X_windows
        self.y = [int(v) for v in y]
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)
        self.modalities = modalities or Modalities(keys=["vibration"], pad_missing="zeros")

    def __len__(self) -> int:
        return len(self.y)

    def _extract_modality(self, w: RawWindow, key: str) -> Optional[np.ndarray]:
        if isinstance(w, dict):
            if key in w:
                return np.asarray(w[key], dtype=np.float32).reshape(-1)
            return None
        # single array: treat as vibration
        if key == "vibration":
            return np.asarray(w, dtype=np.float32).reshape(-1)
        return None

    def _stack_scalograms(self, w: RawWindow) -> np.ndarray:
        # Build per-modality scalogram (C,64,64)
        scalos = []
        first = None
        for k in self.modalities.keys:
            sig = self._extract_modality(w, k)
            if sig is None:
                if self.modalities.pad_missing == "duplicate_first" and first is not None:
                    scalos.append(first)
                else:
                    scalos.append(np.zeros((64, 64), dtype=np.float32))
                continue
            S = scalogram_64x64(sig, self.fs).astype(np.float32)
            if first is None:
                first = S
            scalos.append(S)
        return np.stack(scalos, axis=0)

    def to_tensor(self, windows: List[RawWindow]) -> torch.Tensor:
        arr = np.stack([self._stack_scalograms(w) for w in windows], axis=0)
        return torch.from_numpy(arr)

    def sample_episode(self, K: int | None = None, Q: int | None = None, seed: int | None = None, return_raw: bool = False):
        sup_X, sup_y, qry_X, qry_y = sample_episode(
            self.X, self.y, n_way=self.n_way, k_shot=int(K if K is not None else self.k_shot), q_query=int(Q if Q is not None else self.q_query), seed=int(seed if seed is not None else 0)
        )
        Sx = self.to_tensor(sup_X)
        Qx = self.to_tensor(qry_X)
        Sy = torch.tensor(sup_y, dtype=torch.long)
        Qy = torch.tensor(qry_y, dtype=torch.long)
        # Return also raw (untransformed) support signals for physics loss (source episodes)
        if return_raw:
            return Sx, Sy, Qx, Qy, sup_X, sup_y.tolist()
        return Sx, Sy, Qx, Qy

def sample_episode_with_indices(
    X: List[RawWindow],
    y: List[int],
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
):
    rng = np.random.RandomState(seed)
    idx = build_class_index(y)

    candidates = [c for c, ix in idx.items() if len(ix) >= (k_shot + q_query)]
    if len(candidates) < n_way:
        candidates = list(idx.keys())

    chosen = rng.choice(candidates, size=n_way, replace=(len(candidates) < n_way))

    sup_X, sup_y, qry_X, qry_y = [], [], [], []
    sup_idx, qry_idx = [], []

    for cls_i, c in enumerate(chosen):
        ix = idx[int(c)]
        take = rng.choice(ix, size=k_shot + q_query, replace=(len(ix) < (k_shot + q_query)))
        sup_ix = take[:k_shot]
        qry_ix = take[k_shot:]

        for j in sup_ix:
            j = int(j)
            sup_idx.append(j)
            sup_X.append(X[j])
            sup_y.append(cls_i)

        for j in qry_ix:
            j = int(j)
            qry_idx.append(j)
            qry_X.append(X[j])
            qry_y.append(cls_i)

    return (
        sup_X,
        np.asarray(sup_y, dtype=np.int64),
        qry_X,
        np.asarray(qry_y, dtype=np.int64),
        np.asarray(sup_idx, dtype=np.int64),
        np.asarray(qry_idx, dtype=np.int64),
    )

def _cwru_pool(fid: str) -> int | None:
    s = str(fid)
    if "pool0" in s:
        return 0
    if "pool1" in s:
        return 1
    return None


def sample_episode_disjoint_by_file(
    X: List[RawWindow],
    y: List[int],
    file_id: List[str],
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
) -> Tuple[List[RawWindow], np.ndarray, List[RawWindow], np.ndarray]:
    rng = np.random.RandomState(seed)

    # Build class -> pool -> indices
    y_arr = np.asarray(y, dtype=np.int64)
    pools = np.asarray([_cwru_pool(f) for f in file_id], dtype=object)

    idx = build_class_index(y)

    # verify pool availability
    ok_classes = []
    for c, ix in idx.items():
        p0 = [int(i) for i in ix if pools[int(i)] == 0]
        p1 = [int(i) for i in ix if pools[int(i)] == 1]
        if len(p0) > 0 and len(p1) > 0:
            ok_classes.append(int(c))

    chosen = rng.choice(ok_classes, size=n_way, replace=(len(ok_classes) < n_way))

    sup_X, sup_y, qry_X, qry_y = [], [], [], []
    for cls_i, c in enumerate(chosen):
        ix = idx[int(c)]
        p0 = np.asarray([int(i) for i in ix if pools[int(i)] == 0], dtype=np.int64)
        p1 = np.asarray([int(i) for i in ix if pools[int(i)] == 1], dtype=np.int64)

        sup_ix = rng.choice(p0, size=k_shot, replace=(len(p0) < k_shot))
        qry_ix = rng.choice(p1, size=q_query, replace=(len(p1) < q_query))

        for j in sup_ix:
            sup_X.append(X[int(j)])
            sup_y.append(cls_i)
        for j in qry_ix:
            qry_X.append(X[int(j)])
            qry_y.append(cls_i)

    return sup_X, np.asarray(sup_y, np.int64), qry_X, np.asarray(qry_y, np.int64)


def sample_episode_with_indices_disjoint_by_file(
    X: List[RawWindow],
    y: List[int],
    file_id: List[str],
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: int,
):
    sup_X, sup_y, qry_X, qry_y = sample_episode_disjoint_by_file(
        X, y, file_id, n_way, k_shot, q_query, seed
    )

    # Rebuild indices by re-sampling deterministically 
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y, dtype=np.int64)
    pools = np.asarray([_cwru_pool(f) for f in file_id], dtype=object)

    idx = build_class_index(y)
    ok_classes = []
    for c, ix in idx.items():
        p0 = [int(i) for i in ix if pools[int(i)] == 0]
        p1 = [int(i) for i in ix if pools[int(i)] == 1]
        if len(p0) > 0 and len(p1) > 0:
            ok_classes.append(int(c))

    chosen = rng.choice(ok_classes, size=n_way, replace=(len(ok_classes) < n_way))

    sup_idx, qry_idx = [], []
    for c in chosen:
        ix = idx[int(c)]
        p0 = np.asarray([int(i) for i in ix if pools[int(i)] == 0], dtype=np.int64)
        p1 = np.asarray([int(i) for i in ix if pools[int(i)] == 1], dtype=np.int64)

        sup_ix = rng.choice(p0, size=k_shot, replace=(len(p0) < k_shot))
        qry_ix = rng.choice(p1, size=q_query, replace=(len(p1) < q_query))

        sup_idx.extend([int(j) for j in sup_ix])
        qry_idx.extend([int(j) for j in qry_ix])

    return (
        sup_X,
        sup_y,
        qry_X,
        qry_y,
        np.asarray(sup_idx, np.int64),
        np.asarray(qry_idx, np.int64),
    )
