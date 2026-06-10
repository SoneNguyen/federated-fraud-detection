# This module contains unit tests for the Pydantic schemas defined in the api.schemas module.
# The tests verify that the Transaction schema correctly validates valid input and raises ValidationError for invalid input,
# and that the Prediction schema correctly validates valid input and contains the expected values for the metadata.

import unittest
from pydantic import ValidationError

from api.schemas import Transaction, Prediction, PredictionMetadata


class TestSchemas(unittest.TestCase):

    def test_transaction_validates_successfully(self):
        tx = Transaction(
            tx_amount_usd=100.0,
            tx_count_1h=1,
            tx_count_24h=5,
            tx_volume_1h_usd=50.0,
            tx_volume_24h_usd=200.0,
            merchant_cat_dev=0.1,
            geo_velocity_kmh=10.0,
            dist2_km=3.0,
            card6_code=1,
            days_since_last_tx=2.0,
            account_age_days=30,
            hour_of_day_local=12,
            day_of_week=3,
            orig_currency="USD",
            stale_fx_flag=0,
        )

        self.assertEqual(tx.hour_of_day_local, 12)
        self.assertEqual(tx.day_of_week, 3)
        self.assertEqual(tx.orig_currency, "USD")
        self.assertEqual(tx.stale_fx_flag, 0)

    def test_transaction_rejects_invalid_hour(self):
        with self.assertRaises(ValidationError):
            Transaction(
                tx_amount_usd=100.0,
                tx_count_1h=1,
                tx_count_24h=5,
                tx_volume_1h_usd=50.0,
                tx_volume_24h_usd=200.0,
                merchant_cat_dev=0.1,
                geo_velocity_kmh=10.0,
                dist2_km=3.0,
                card6_code=1,
                days_since_last_tx=2.0,
                account_age_days=30,
                hour_of_day_local=24,
                day_of_week=3,
            )

    def test_transaction_rejects_invalid_day_of_week(self):
        with self.assertRaises(ValidationError):
            Transaction(
                tx_amount_usd=100.0,
                tx_count_1h=1,
                tx_count_24h=5,
                tx_volume_1h_usd=50.0,
                tx_volume_24h_usd=200.0,
                merchant_cat_dev=0.1,
                geo_velocity_kmh=10.0,
                dist2_km=3.0,
                card6_code=1,
                days_since_last_tx=2.0,
                account_age_days=30,
                hour_of_day_local=12,
                day_of_week=7,
            )

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
