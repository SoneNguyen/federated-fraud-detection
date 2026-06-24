import json

import numpy as np
import pandas as pd

from scripts.zero_shot_external_eval import (
    apply_trained_normalization,
    inspect_frame,
    recall_at_precision,
)
from src.data.feature_registry import LABEL


def test_apply_trained_normalization_uses_saved_stats(tmp_path):
    norm_path = tmp_path / "normalization.json"
    norm_path.write_text(
        json.dumps({"tx_amount_usd": {"mean": 2.0, "std": 2.0}}),
        encoding="utf-8",
    )
    frame = pd.DataFrame({"tx_amount_usd": [0.0, 2.0, 4.0], LABEL: [0, 1, 0]})

    out = apply_trained_normalization(frame, norm_path)

    assert out["tx_amount_usd"].tolist() == [-1.0, 0.0, 1.0]
    assert out[LABEL].tolist() == [0, 1, 0]


def test_recall_at_fixed_precision():
    labels = np.array([1, 0, 1, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)

    assert recall_at_precision(labels, scores, target_precision=0.50) == 1.0


def test_inspect_frame_reports_entity_velocity():
    frame = pd.DataFrame(
        {
            "time": [1, 2],
            "amount": [10.0, 20.0],
            "label": [0, 1],
            "account": ["a", "a"],
        }
    )
    mapping = {
        "columns": {
            "transaction_time": "time",
            "amount": "amount",
            "label": "label",
            "account_id": "account",
        }
    }

    report = inspect_frame(frame, mapping)

    assert report["rows"] == 2
    assert report["fraud_ratio"] == 0.5
    assert report["entity_column_for_velocity"] == "account"
    assert report["velocity_reconstructed"] is True
