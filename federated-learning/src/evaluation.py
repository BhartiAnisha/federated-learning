import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(y, p, thr=0.5):
    """All binary-classification metrics from probabilities p of the positive class (hard labels at thr)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(
        accuracy=accuracy_score(y, yh),
        precision=precision_score(y, yh, zero_division=0),
        recall=recall_score(y, yh, zero_division=0),
        specificity=spec,
        f1=f1_score(y, yh, zero_division=0),
        macro_f1=f1_score(y, yh, average="macro", zero_division=0),
        weighted_f1=f1_score(y, yh, average="weighted", zero_division=0),
        roc_auc=roc_auc_score(y, p) if both else np.nan,       # from probabilities
        pr_auc=average_precision_score(y, p) if both else np.nan,
        balanced_accuracy=balanced_accuracy_score(y, yh),
        mcc=matthews_corrcoef(y, yh),
        fpr=fp / (fp + tn) if (fp + tn) else np.nan,
        fnr=fn / (fn + tp) if (fn + tp) else np.nan,
        log_loss=log_loss(y, p, labels=[0, 1]),
        brier=brier_score_loss(y, p),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def quick_val(y, p):
    """Cheap per-round validation numbers (loss, accuracy, AUC)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p >= 0.5) == y))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return loss, acc, auc
