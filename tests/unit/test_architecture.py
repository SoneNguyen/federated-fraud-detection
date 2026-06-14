"""Tests for model/architecture.py re-exports."""
import unittest

from src.model.fraud_mlp import FraudMLP, INPUT_DIM


class TestArchitecture(unittest.TestCase):

    def test_fraudmlp_imported(self):
        """FraudMLP should be available from model.architecture."""
        model = FraudMLP()
        assert model is not None

    def test_input_dim_imported(self):
        """INPUT_DIM should be available from model.architecture."""
        assert INPUT_DIM == 34


if __name__ == "__main__":
    unittest.main()
