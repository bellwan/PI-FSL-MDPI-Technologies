
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any
from pifsl.data.bosch_drilling import config
from pifsl.data.bosch_drilling.raw_processing.gating import make_embed_matrix_for_cnn
from sklearn.neighbors import KNeighborsClassifier

def condensed_nn_subset(Xm: np.ndarray, ym: np.ndarray, k: int = 1, max_iters: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    classes = np.unique(ym)
    S_idx=[]
    for c in classes:
        idx_c = np.where(ym==c)[0]
        if len(idx_c):
            S_idx.append(int(rng.choice(idx_c)))
    S_idx = list(sorted(set(S_idx)))
    for _ in range(max_iters):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Xm[S_idx], ym[S_idx])
        yhat = knn.predict(Xm)
        mis = np.where(yhat != ym)[0]
        add = [int(i) for i in mis if ym[i]==0]
        if not add:
            break
        S_idx.extend(add); S_idx = list(sorted(set(S_idx)))
    mask = np.zeros(len(ym), dtype=bool); mask[S_idx] = True
    return mask

def undersample_ok_safe(X_tr: List[np.ndarray], y_tr: List[int]) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    diag = {"before_ok":0, "before_nok":0, "after_ok":0, "after_nok":0, "ok_kept_pct":None, "condensed_size":None}
    if len(X_tr)==0:
        return X_tr, y_tr, diag
    y = np.asarray(y_tr, dtype=int)
    diag["before_ok"] = int((y==0).sum())
    diag["before_nok"] = int((y==1).sum())
    X_embed = make_embed_matrix_for_cnn(X_tr, config.FS)
    mask = condensed_nn_subset(X_embed, y, k=config.U_CNN_K, max_iters=config.U_CNN_MAX_ITERS, seed=config.U_CNN_SEED)
    keep = np.where(y==1, True, mask)
    yk = y[keep]
    if (yk==0).sum()==0 or (yk==1).sum()==0:
        diag["after_ok"]  = diag["before_ok"]
        diag["after_nok"] = diag["before_nok"]
        diag["ok_kept_pct"] = 100.0
        diag["condensed_size"] = int(mask.sum())
        return X_tr, y_tr, diag
    idx = np.where(keep)[0].tolist()
    diag["after_ok"]  = int((yk==0).sum())
    diag["after_nok"] = int((yk==1).sum())
    diag["ok_kept_pct"] = 100.0 * diag["after_ok"] / max(1, diag["before_ok"])
    diag["condensed_size"] = int(mask.sum())
    return [X_tr[i] for i in idx], [int(y_tr[i]) for i in idx], diag
