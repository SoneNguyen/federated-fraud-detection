import numpy as np
import pytest

from src.server.aggregation import (
    robust_blended_average_ndarrays,
    stabilize_aggregate_update,
    target_aware_fedavg_weights,
    target_score,
    weighted_average_ndarrays,
)


def test_target_score_caps_metrics_at_target():
    score = target_score(
        auprc=0.80,
        auroc=0.95,
        f1=0.75,
        target_auprc=0.70,
        target_auroc=0.90,
        target_f1=0.70,
    )
    assert score == pytest.approx(1.0)


def test_target_aware_weights_sum_to_one_and_reduce_size_dominance():
    weights = target_aware_fedavg_weights(
        client_metrics=[
            {"val_auprc": 0.40, "val_auroc": 0.70, "val_f1": 0.35},
            {"val_auprc": 0.75, "val_auroc": 0.92, "val_f1": 0.72},
        ],
        client_examples=[100, 400],
        target_auprc=0.70,
        target_auroc=0.90,
        target_f1=0.70,
        fairness_weight=0.15,
        profile="ambitious",
    )

    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > 100 / (100 + 400)
    assert weights[1] < 400 / (100 + 400)


def test_weighted_average_ndarrays_averages_matching_parameters():
    aggregated = weighted_average_ndarrays(
        client_parameters=[
            [np.array([1.0, 3.0], dtype=np.float32)],
            [np.array([3.0, 5.0], dtype=np.float32)],
        ],
        weights=[0.25, 0.75],
    )

    np.testing.assert_allclose(
        aggregated[0],
        np.array([2.5, 4.5], dtype=np.float32),
    )


def test_scalable_weights_prefer_reliable_clients():
    weights = target_aware_fedavg_weights(
        client_metrics=[
            {"val_auprc": 0.30, "val_auroc": 0.70, "val_f1": 0.30},
            {"val_auprc": 0.75, "val_auroc": 0.92, "val_f1": 0.72},
        ],
        client_examples=[400, 400],
        target_auprc=0.70,
        target_auroc=0.90,
        target_f1=0.70,
        fairness_weight=0.15,
        profile="scalable",
    )

    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] > weights[0]


def test_stabilize_aggregate_update_clips_large_delta():
    previous = [np.array([10.0, 0.0], dtype=np.float32)]
    proposed = [np.array([20.0, 0.0], dtype=np.float32)]

    stabilized, meta = stabilize_aggregate_update(
        previous=previous,
        proposed=proposed,
        server_lr=0.5,
        max_update_ratio=0.1,
    )

    np.testing.assert_allclose(stabilized[0], np.array([10.5, 0.0], dtype=np.float32))
    assert meta["server_update_scale"] == pytest.approx(0.1)
    assert meta["server_lr"] == pytest.approx(0.5)


def test_robust_blended_average_dampens_outlier_update():
    client_parameters = [
        [np.array([1.0, 1.0], dtype=np.float32)],
        [np.array([1.1, 0.9], dtype=np.float32)],
        [np.array([0.9, 1.2], dtype=np.float32)],
        [np.array([50.0, -40.0], dtype=np.float32)],
    ]
    weights = [0.25, 0.25, 0.25, 0.25]

    regular = weighted_average_ndarrays(client_parameters, weights)
    robust, meta = robust_blended_average_ndarrays(
        client_parameters,
        weights,
        trim_ratio=0.25,
        median_blend=1.0,
    )

    assert regular[0][0] > 10.0
    np.testing.assert_allclose(robust[0], np.array([1.05, 0.95], dtype=np.float32))
    assert meta["robust_trim_ratio"] == pytest.approx(0.25)
    assert meta["robust_median_blend"] == pytest.approx(1.0)
