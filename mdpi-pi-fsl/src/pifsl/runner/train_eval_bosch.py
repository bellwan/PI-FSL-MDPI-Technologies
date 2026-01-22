
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, matthews_corrcoef, confusion_matrix
)

def episodic_train(model, data, episodes=1000, K=3, Q=8,
                   lr=1e-3, step_every=400, gamma=0.5, wd=0.0,
                   device="cpu", thresh=0.5):
    model.to(device); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_every, gamma=gamma)
    for ep in range(1, episodes+1):
        Sx,Sy,Qx,Qy = data.sample_episode(K,Q)
        if torch.unique(Sy).numel() < 2:
            continue
        Sx,Sy,Qx,Qy = Sx.to(device), Sy.to(device), Qx.to(device), Qy.to(device)
        scores, classes = model.forward_episode(Sx, Sy, Qx)
        cls_sorted, _ = classes.sort()
        target = torch.zeros_like(Qy)
        for i,lab in enumerate(cls_sorted):
            target[Qy==lab] = i
        loss = F.cross_entropy(scores, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % step_every == 0:
            sched.step()

@torch.no_grad()
def evaluate_relation(
    model,
    data,
    K=3,
    Q=16,
    episodes=100,
    seed=123,
    device="cpu",
    thresh=0.5,
):
    model.to(device)
    model.eval()
    rng = np.random.RandomState(seed)

    all_p = []
    all_y = []

    for _ in range(episodes):
        # sample an episode with its own seed for reproducibility
        Sx, Sy, Qx, Qy = data.sample_episode(
            K, Q, seed=int(rng.randint(0, 1 << 31))
        )

        # skip degenerate episodes (only one class in support)
        if torch.unique(Sy).numel() < 2:
            continue

        Sx, Sy, Qx, Qy = (
            Sx.to(device),
            Sy.to(device),
            Qx.to(device),
            Qy.to(device),
        )

        scores, classes = model.forward_episode(Sx, Sy, Qx)
        probs = F.softmax(scores, dim=1)

        # classes contains the class labels corresponding to columns in 'scores'
        cls_sorted, _ = classes.sort()

        # assume label "1" is the worn / NOK class
        if (cls_sorted == 1).any():
            nok_col = (cls_sorted == 1).nonzero(as_tuple=True)[0].item()
            p_nok = probs[:, nok_col]
        else:
            # no worn class present in this episode → skip
            continue

        all_p.append(p_nok.detach().cpu().numpy())
        all_y.append(Qy.detach().cpu().numpy())

    # no valid episodes → return NaNs and zero confusion counts
    if len(all_p) == 0:
        metrics = dict(
            f1=np.nan,
            acc=np.nan,
            bacc=np.nan,
            prec=np.nan,
            rec=np.nan,
            roc_auc=np.nan,
            pr_auc=np.nan,
            brier=np.nan,
            mcc=np.nan,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            n=0,
        )
        return {
            "metrics": metrics,
            "probs": None,
            "labels": None,
        }

    # concatenate across all episodes
    p = np.concatenate(all_p)
    y = np.concatenate(all_y).astype(int)

    # hard decisions
    yhat = (p >= thresh).astype(int)

    # confusion matrix (0 = healthy, 1 = worn)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    # scalar metrics
    f1 = float(f1_score(y, yhat, zero_division=0))
    acc = float(accuracy_score(y, yhat))
    bacc = float(balanced_accuracy_score(y, yhat))
    prec = float(precision_score(y, yhat, zero_division=0))
    rec = float(recall_score(y, yhat, zero_division=0))

    if len(np.unique(y)) > 1:
        roc_auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    brier = float(brier_score_loss(y, p))

    if (tp + fp > 0 and tp + fn > 0 and tn + fp > 0 and tn + fn > 0):
        mcc = float(matthews_corrcoef(y, yhat))
    else:
        mcc = 0.0

    metrics = dict(
        f1=f1,
        acc=acc,
        bacc=bacc,
        prec=prec,
        rec=rec,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier=brier,
        mcc=mcc,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        n=int(len(y)),
    )

    # return metrics + raw outputs
    return {
        "metrics": metrics,
        "probs": p.tolist(),
        "labels": y.tolist(),
    }
