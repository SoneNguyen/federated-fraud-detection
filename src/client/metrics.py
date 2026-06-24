"""Lightweight binary metrics for client-side validation.

The client process avoids importing sklearn/scipy because large multi-client
runs spawn many Python processes on one machine.
"""

from __future__ import annotations

import numpy as np


def _sorted_binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    order = np.argsort(s, kind="mergesort")[::-1]
    return y[order], s[order]


def precision_recall_curve_np(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _sorted_binary(y_true, y_score)
    positives = int(y.sum())
    if len(y) == 0 or positives == 0:
        return (
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    distinct = np.where(np.diff(s))[0]
    threshold_idxs = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)

    precision_desc = tps / np.maximum(tps + fps, 1.0)
    recall_desc = tps / positives
    thresholds_desc = s[threshold_idxs]

    precision = np.r_[precision_desc[::-1], 1.0]
    recall = np.r_[recall_desc[::-1], 0.0]
    thresholds = thresholds_desc[::-1]
    return precision, recall, thresholds


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve_np(y_true, y_score)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y, s = _sorted_binary(y_true, y_score)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    distinct = np.where(np.diff(s))[0]
    threshold_idxs = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)
    tpr = np.r_[0.0, tps / positives]
    fpr = np.r_[0.0, fps / negatives]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
