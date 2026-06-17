"""Tests for the Pydantic schemas defined in api.schemas."""

import unittest

from pydantic import ValidationError

from api.schemas import Prediction, PredictionMetadata, Transaction
from src.data.dataset import FEATURE_ORDER


def _payload(**overrides):
    base = {name: 0.0 for name in FEATURE_ORDER}
    base.update(
        {
            "tx_amount_usd": 100.0,
            "tx_count_1h": 1.0,
            "tx_count_24h": 5.0,
            "tx_volume_1h_usd": 50.0,
            "tx_volume_24h_usd": 200.0,
            "geo_velocity_kmh": 10.0,
            "dist2_km": 3.0,
            "days_since_last_tx": 2.0,
            "account_age_days": 30.0,
            "hour_of_day_local": 12.0,
            "day_of_week": 3.0,
            "tx_time_norm": 0.5,
            "week_of_period": 0.25,
            "prod_W": 1.0,
            "card1_norm": 1.0,
            "card2_norm": 1.0,
            "addr1_norm": 10.0,
            "addr2_norm": 2.0,
            "email_domain_match": 1.0,
            "p_email_free": 1.0,
            "card3_norm": 5.0,
            "card4_code": 1.0,
            "orig_currency": "USD",
            "stale_fx_flag": 0,
        }
    )
    base.update(overrides)
    return base


class TestSchemas(unittest.TestCase):
    def test_transaction_validates_successfully(self):
        tx = Transaction(**_payload())

        self.assertEqual(tx.hour_of_day_local, 12)
        self.assertEqual(tx.day_of_week, 3)
        self.assertEqual(tx.orig_currency, "USD")
        self.assertEqual(tx.stale_fx_flag, 0)

    def test_transaction_rejects_invalid_hour(self):
        with self.assertRaises(ValidationError):
            Transaction(**_payload(hour_of_day_local=24))

    def test_transaction_rejects_invalid_day_of_week(self):
        with self.assertRaises(ValidationError):
            Transaction(**_payload(day_of_week=7))

    def test_transaction_rejects_stale_schema_fields(self):
        with self.assertRaises(ValidationError):
            Transaction(**_payload(M4_flag=1.0))

    def test_prediction_model_valid(self):
        metadata = PredictionMetadata(stale_fx_flag=1, orig_currency="EUR")
        pred = Prediction(
            fraud_probability=0.42,
            prediction=1,
            model_version="v1",
            metadata=metadata,
        )

        self.assertEqual(pred.metadata.orig_currency, "EUR")
        self.assertEqual(pred.prediction, 1)
        self.assertEqual(pred.model_version, "v1")


if __name__ == "__main__":
    unittest.main()
