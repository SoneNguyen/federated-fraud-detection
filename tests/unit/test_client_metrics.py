import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.client.metrics import (
    average_precision_score_np,
    precision_recall_curve_np,
    roc_auc_score_np,
)


def test_client_metrics_match_sklearn() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    y_score = np.array([0.05, 0.9, 0.1, 0.4, 0.8, 0.2, 0.3, 0.7])

    assert average_precision_score_np(y_true, y_score) == pytest.approx(
        average_precision_score(y_true, y_score)
    )
    assert roc_auc_score_np(y_true, y_score) == pytest.approx(
        roc_auc_score(y_true, y_score)
    )

    precision, recall, thresholds = precision_recall_curve_np(y_true, y_score)
    expected_precision, expected_recall, expected_thresholds = precision_recall_curve(
        y_true,
        y_score,
    )
    np.testing.assert_allclose(precision, expected_precision)
    np.testing.assert_allclose(recall, expected_recall)
    np.testing.assert_allclose(thresholds, expected_thresholds)
