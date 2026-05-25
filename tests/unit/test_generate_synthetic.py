import unittest
import numpy as np

from data.generate_synthetic import CLIENT_CURRENCIES, gen_rows


class TestGenerateSynthetic(unittest.TestCase):

    def test_gen_rows_outputs_expected_columns_and_lengths(self):
        rng = np.random.default_rng(123)
        currency = CLIENT_CURRENCIES[0]

        rows = gen_rows(10, False, rng, currency, "UTC")

        expected_columns = {
            "tx_amount_usd",
            "tx_count_1h",
            "tx_count_24h",
            "tx_volume_1h_usd",
            "tx_volume_24h_usd",
            "merchant_cat_dev",
            "geo_velocity_kmh",
            "days_since_last_tx",
            "account_age_days",
            "hour_of_day_local",
            "day_of_week",
            "orig_currency",
            "stale_fx_flag",
            "is_fraud",
        }

        self.assertEqual(set(rows.keys()), expected_columns)
        self.assertEqual(len(rows["tx_amount_usd"]), 10)
        self.assertEqual(len(rows["orig_currency"]), 10)
        self.assertEqual(len(rows["is_fraud"]), 10)
        self.assertTrue(all(flag == 0 for flag in rows["is_fraud"]))
        self.assertEqual(set(rows["orig_currency"]), {currency})
        self.assertTrue(all(0 <= h < 24 for h in rows["hour_of_day_local"]))
        self.assertTrue(all(0 <= d < 7 for d in rows["day_of_week"]))

    def test_gen_rows_produces_fraud_flag_as_integer(self):
        rng = np.random.default_rng(456)
        currency = CLIENT_CURRENCIES[2]

        rows = gen_rows(5, True, rng, currency, "UTC")

        self.assertEqual(len(rows["is_fraud"]), 5)
        self.assertTrue(all(flag == 1 for flag in rows["is_fraud"]))
        self.assertTrue(all(c == currency for c in rows["orig_currency"]))
        self.assertEqual(len(rows["tx_count_1h"]), 5)
        self.assertEqual(len(rows["tx_volume_24h_usd"]), 5)


if __name__ == "__main__":
    unittest.main()
