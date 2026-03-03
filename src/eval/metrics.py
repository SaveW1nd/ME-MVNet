"""Metrics for ISRJ parameter estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_overall_metrics(
    *,
    tl_true: np.ndarray,
    tl_pred: np.ndarray,
    tf_true: np.ndarray,
    tf_pred: np.ndarray,
    nf_true: np.ndarray,
    nf_pred: np.ndarray,
) -> dict[str, float]:
    """Compute overall regression/classification metrics."""
    return {
        "Tl_MAE": mae(tl_true, tl_pred),
        "Tl_RMSE": rmse(tl_true, tl_pred),
        "Tf_MAE": mae(tf_true, tf_pred),
        "Tf_RMSE": rmse(tf_true, tf_pred),
        "NF_Acc": float(accuracy_score(nf_true, nf_pred)),
        "NF_MacroF1": float(f1_score(nf_true, nf_pred, average="macro")),
    }


def compute_jnr_bucket_metrics(
    *,
    jnr_db: np.ndarray,
    tl_true: np.ndarray,
    tl_pred: np.ndarray,
    tf_true: np.ndarray,
    tf_pred: np.ndarray,
    nf_true: np.ndarray,
    nf_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute per-JNR bucket metrics."""
    rows = []
    for j in sorted(np.unique(jnr_db).tolist()):
        idx = np.where(jnr_db == j)[0]
        rows.append(
            {
                "JNR_dB": int(j),
                "Count": int(idx.shape[0]),
                "Tl_MAE": mae(tl_true[idx], tl_pred[idx]),
                "Tl_RMSE": rmse(tl_true[idx], tl_pred[idx]),
                "Tf_MAE": mae(tf_true[idx], tf_pred[idx]),
                "Tf_RMSE": rmse(tf_true[idx], tf_pred[idx]),
                "NF_Acc": float(accuracy_score(nf_true[idx], nf_pred[idx])),
                "NF_MacroF1": float(f1_score(nf_true[idx], nf_pred[idx], average="macro")),
            }
        )
    return rows


def compute_nf_confusion(nf_true: np.ndarray, nf_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix in fixed label order [1,2,4]."""
    return confusion_matrix(nf_true, nf_pred, labels=[1, 2, 4])
