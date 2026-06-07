# This module contains unit tests for the FXConverter class in the data.fx.converter module.
# The tests verify that the to_usd method returns the correct converted amount and stale flag for
# a known currency, that it falls back to a rate of 1.0 for unknown currencies without crashing, 
# and that the FX_CACHE_TTL variable is correctly read from the schema.json file.
import unittest
import json
import time
from data.fx.converter import FXConverter, FX_CACHE_TTL

class TestFXConverter(unittest.TestCase):

    def setUp(self):
        # We pass a controlled 'rates' dictionary here. 
        # This ensures the test always passes even if STATIC_RATES changes later.
        self.mock_rates = {"EUR": 1.082}
        self.fx = FXConverter(rates=self.mock_rates)

    def test_eur_conversion_correct_rate_and_fresh(self):
        """Test 1: to_usd('EUR') returns correct rate and is not stale (0)"""
        usd, stale = self.fx.to_usd(100.0, "EUR")
        
        # Using assertAlmostEqual is best practice for floating-point math
        self.assertAlmostEqual(usd, 108.20, places=2)
        self.assertEqual(stale, 0)

    def test_unknown_currency_fallback(self):
        """Test 2: to_usd('UNKNOWN') falls back to rate 1.0 without crashing"""
        usd, stale = self.fx.to_usd(100.0, "UNKNOWN_CURRENCY")
        
        # If rate falls back to 1.0, 100.0 * 1.0 = 100.0
        self.assertEqual(usd, 100.0)

    def test_fx_cache_ttl_reads_from_schema(self):
        """Test 3: FX_CACHE_TTL is dynamically read from schema.json"""
        # Verify the variable imported from converter.py equals 900
        self.assertEqual(FX_CACHE_TTL, 900)
        
        # To be extra safe, let's verify it actually matches the physical file 
        # just in case someone hardcodes 900 into converter.py later!
        with open("contracts/schema.json") as f:
            schema_data = json.load(f)
            expected_ttl = schema_data["feature_schema"]["fx_cache_ttl_sec"]
            
        self.assertEqual(FX_CACHE_TTL, expected_ttl)

    def test_is_stale_fresh_cache(self):
        """Test 4: is_stale() returns False when cache is fresh"""
        fx = FXConverter(rates=self.mock_rates, cache_ts=time.time())
        self.assertFalse(fx.is_stale())

    def test_is_stale_old_cache(self):
        """Test 5: is_stale() returns True when cache exceeds TTL"""
        old_ts = time.time() - (FX_CACHE_TTL + 100)
        fx = FXConverter(rates=self.mock_rates, cache_ts=old_ts)
        self.assertTrue(fx.is_stale())

    def test_stale_flag_in_to_usd(self):
        """Test 6: to_usd() returns stale flag=1 when cache is old"""
        old_ts = time.time() - (FX_CACHE_TTL + 50)
        fx = FXConverter(rates=self.mock_rates, cache_ts=old_ts)
        usd, stale = fx.to_usd(100.0, "EUR")
        self.assertEqual(stale, 1)

if __name__ == "__main__":
    unittest.main()