"""OOD detection metrics.

INPUT CONVENTION: `id_unc`, `ood_unc` are UNCERTAINTY scores (higher = more OOD-like).
For confidence scores such as MSP, pass `-confidence`.

FPR@95TPR follows OpenOOD (the protocol the paper claims to use): ID samples are the
positives; pick the threshold that keeps 95% of ID samples, report the fraction of OOD
samples that are (wrongly) also kept.  The official HCM script uses the opposite
convention (OOD positive, FPR on ID at 95% OOD recall); it is returned as
`fpr95_hcm_repo` so the difference can be documented, but tables should use `fpr95`.
AUROC is identical under both conventions.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def _fpr_at_tpr(labels, scores, tpr_level=0.95):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_level, side="left")
    return float(fpr[min(idx, len(fpr) - 1)])


def ood_metrics(id_unc, ood_unc):
    id_unc = np.asarray(id_unc, dtype=np.float64)
    ood_unc = np.asarray(ood_unc, dtype=np.float64)
    # OOD as positive class, score = uncertainty
    y_out = np.r_[np.zeros(len(id_unc)), np.ones(len(ood_unc))]
    s = np.r_[id_unc, ood_unc]
    return {
        "auroc": float(roc_auc_score(y_out, s)),
        "aupr_out": float(average_precision_score(y_out, s)),
        "aupr_in": float(average_precision_score(1 - y_out, -s)),
        "fpr95": _fpr_at_tpr(1 - y_out, -s),          # OpenOOD: ID positive
        "fpr95_hcm_repo": _fpr_at_tpr(y_out, s),       # official repo convention
    }


def accuracy(logits, labels):
    return float((np.asarray(logits).argmax(1) == np.asarray(labels)).mean())
