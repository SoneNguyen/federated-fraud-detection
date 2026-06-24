import pandas as pd

from dataset.custom_transaction_adapter import to_ieee_like_frame
from dataset.load_ieee_cis import engineer
from src.data.feature_registry import FEATURE_ORDER, LABEL


def test_custom_transaction_adapter_matches_feature_contract():
    raw = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "timestamp": [
                "2026-01-01T08:00:00Z",
                "2026-01-01T08:30:00Z",
                "2026-01-01T09:10:00Z",
                "2026-01-02T22:00:00Z",
            ],
            "amount_usd": [10.0, 15.0, 200.0, 500.0],
            "is_fraud": [0, 0, 1, 1],
            "channel": ["payment", "payment", "transfer", "wallet"],
            "customer_id": ["c1", "c1", "c2", "c3"],
            "merchant_id": ["m1", "m1", "m2", "m3"],
            "device_id": ["d1", "d1", "d2", "d3"],
            "device_type": ["mobile", "mobile", "desktop", "mobile"],
            "province": ["HCM", "HCM", "HN", "DN"],
        }
    )
    mapping = {
        "columns": {
            "transaction_id": "transaction_id",
            "transaction_time": "timestamp",
            "amount": "amount_usd",
            "label": "is_fraud",
            "product": "channel",
            "customer_id": "customer_id",
            "merchant_id": "merchant_id",
            "device_id": "device_id",
            "device_type": "device_type",
            "region": "province",
        },
        "product_map": {"payment": "W", "transfer": "C", "wallet": "W"},
    }

    mapped = to_ieee_like_frame(raw, mapping)
    featured = engineer(mapped, mapped)

    assert list(featured[FEATURE_ORDER].columns) == FEATURE_ORDER
    assert featured[LABEL].sum() == 2
    assert len(featured) == len(raw)
