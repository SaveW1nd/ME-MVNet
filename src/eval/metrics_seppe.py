"""Evaluation metrics for ME-MVSepPE."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size > 0 else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size > 0 else 0.0


def compute_metrics_overall(
    *,
    tl_true_us: np.ndarray,
    tl_pred_us: np.ndarray,
    ts_true_us: np.ndarray,
    ts_pred_us: np.ndarray,
    nf_true: np.ndarray,
    nf_pred: np.ndarray,
    tl_tol_us: float = 0.15,
    ts_tol_us: float = 0.25,
) -> dict[str, float]:
    """Compute overall metrics."""
    active = nf_true > 0

    tl_true_a = tl_true_us[active]
    tl_pred_a = tl_pred_us[active]
    ts_true_a = ts_true_us[active]
    ts_pred_a = ts_pred_us[active]
    nf_true_a = nf_true[active]
    nf_pred_a = nf_pred[active]

    all_true = nf_true.reshape(-1)
    all_pred = nf_pred.reshape(-1)
    nf_acc = float(accuracy_score(all_true, all_pred)) if all_true.size > 0 else 0.0
    nf_macro = (
        float(f1_score(all_true, all_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        if all_true.size > 0
        else 0.0
    )

    ok_tl = np.abs(tl_pred_a - tl_true_a) <= tl_tol_us
    ok_ts = np.abs(ts_pred_a - ts_true_a) <= ts_tol_us
    ok_nf = nf_pred_a == nf_true_a
    ok_total = ok_tl & ok_ts & ok_nf

    return {
        "Tl_MAE_us": _mae(tl_true_a, tl_pred_a),
        "Tl_RMSE_us": _rmse(tl_true_a, tl_pred_a),
        "Ts_MAE_us": _mae(ts_true_a, ts_pred_a),
        "Ts_RMSE_us": _rmse(ts_true_a, ts_pred_a),
        "NF_Acc": nf_acc,
        "NF_macroF1": nf_macro,
        "A_Tl": float(np.mean(ok_tl)) if ok_tl.size > 0 else 0.0,
        "A_Ts": float(np.mean(ok_ts)) if ok_ts.size > 0 else 0.0,
        "A_NF": float(np.mean(ok_nf)) if ok_nf.size > 0 else 0.0,
        "A_total": float(np.mean(ok_total)) if ok_total.size > 0 else 0.0,
    }


def compute_metrics_by_jnr(
    *,
    tl_true_us: np.ndarray,
    tl_pred_us: np.ndarray,
    ts_true_us: np.ndarray,
    ts_pred_us: np.ndarray,
    nf_true: np.ndarray,
    nf_pred: np.ndarray,
    jnr_db: np.ndarray,
    tl_tol_us: float = 0.15,
    ts_tol_us: float = 0.25,
    bin_step_db: float = 2.0,
) -> list[dict[str, Any]]:
    """Compute active-source metrics binned by JNR."""
    active = nf_true > 0
    if not np.any(active):
        return []

    tl_t = tl_true_us[active]
    tl_p = tl_pred_us[active]
    ts_t = ts_true_us[active]
    ts_p = ts_pred_us[active]
    nf_t = nf_true[active]
    nf_p = nf_pred[active]
    jnr = jnr_db[active]

    jmin = float(np.floor(np.min(jnr)))
    jmax = float(np.ceil(np.max(jnr)))
    edges = np.arange(jmin, jmax + bin_step_db, bin_step_db)
    if edges.size < 2:
        edges = np.array([jmin, jmin + bin_step_db], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        idx = np.where((jnr >= lo) & (jnr < hi))[0]
        if idx.size == 0:
            continue

        ok_tl = np.abs(tl_p[idx] - tl_t[idx]) <= tl_tol_us
        ok_ts = np.abs(ts_p[idx] - ts_t[idx]) <= ts_tol_us
        ok_nf = nf_p[idx] == nf_t[idx]
        ok_total = ok_tl & ok_ts & ok_nf
        rows.append(
            {
                "JNR_lo": float(lo),
                "JNR_hi": float(hi),
                "Count": int(idx.size),
                "Tl_MAE_us": _mae(tl_t[idx], tl_p[idx]),
                "Tl_RMSE_us": _rmse(tl_t[idx], tl_p[idx]),
                "Ts_MAE_us": _mae(ts_t[idx], ts_p[idx]),
                "Ts_RMSE_us": _rmse(ts_t[idx], ts_p[idx]),
                "A_Tl": float(np.mean(ok_tl)),
                "A_Ts": float(np.mean(ok_ts)),
                "A_NF": float(np.mean(ok_nf)),
                "A_total": float(np.mean(ok_total)),
            }
        )
    return rows


def compute_metrics_by_kactive(
    *,
    tl_true_us: np.ndarray,
    tl_pred_us: np.ndarray,
    ts_true_us: np.ndarray,
    ts_pred_us: np.ndarray,
    nf_true: np.ndarray,
    nf_pred: np.ndarray,
    k_active: np.ndarray,
    tl_tol_us: float = 0.15,
    ts_tol_us: float = 0.25,
) -> list[dict[str, Any]]:
    """Compute metrics grouped by K_active in {2,3}."""
    rows = []
    for ka in [2, 3]:
        sidx = np.where(k_active == ka)[0]
        if sidx.size == 0:
            continue

        tl_t = tl_true_us[sidx]
        tl_p = tl_pred_us[sidx]
        ts_t = ts_true_us[sidx]
        ts_p = ts_pred_us[sidx]
        nf_t = nf_true[sidx]
        nf_p = nf_pred[sidx]

        active = nf_t > 0
        tl_ta = tl_t[active]
        tl_pa = tl_p[active]
        ts_ta = ts_t[active]
        ts_pa = ts_p[active]
        nf_ta = nf_t[active]
        nf_pa = nf_p[active]

        ok_tl = np.abs(tl_pa - tl_ta) <= tl_tol_us
        ok_ts = np.abs(ts_pa - ts_ta) <= ts_tol_us
        ok_nf = nf_pa == nf_ta
        ok_total = ok_tl & ok_ts & ok_nf

        nf_acc = float(accuracy_score(nf_t.reshape(-1), nf_p.reshape(-1)))
        nf_macro = float(
            f1_score(
                nf_t.reshape(-1),
                nf_p.reshape(-1),
                average="macro",
                labels=[0, 1, 2, 3],
                zero_division=0,
            )
        )
        rows.append(
            {
                "K_active": int(ka),
                "Count_samples": int(sidx.size),
                "Count_active_sources": int(np.sum(active)),
                "Tl_MAE_us": _mae(tl_ta, tl_pa),
                "Tl_RMSE_us": _rmse(tl_ta, tl_pa),
                "Ts_MAE_us": _mae(ts_ta, ts_pa),
                "Ts_RMSE_us": _rmse(ts_ta, ts_pa),
                "NF_Acc": nf_acc,
                "NF_macroF1": nf_macro,
                "A_Tl": float(np.mean(ok_tl)) if ok_tl.size > 0 else 0.0,
                "A_Ts": float(np.mean(ok_ts)) if ok_ts.size > 0 else 0.0,
                "A_NF": float(np.mean(ok_nf)) if ok_nf.size > 0 else 0.0,
                "A_total": float(np.mean(ok_total)) if ok_total.size > 0 else 0.0,
            }
        )
    return rows


def compute_nf_confusion_4(nf_true: np.ndarray, nf_pred: np.ndarray) -> np.ndarray:
    """Confusion matrix in fixed class order [0,1,2,3]."""
    return confusion_matrix(nf_true.reshape(-1), nf_pred.reshape(-1), labels=[0, 1, 2, 3])
