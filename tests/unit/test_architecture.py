"""Tests for the active FraudMLP architecture."""
import unittest

from src.model.fraud_mlp import FraudMLP, INPUT_DIM
from src.data.dataset import FEATURE_ORDER


class TestArchitecture(unittest.TestCase):

    def test_fraudmlp_imported(self):
        model = FraudMLP()
        assert model is not None

    def test_input_dim_matches_schema(self):
        assert INPUT_DIM == len(FEATURE_ORDER)


if __name__ == "__main__":
    unittest.main()
