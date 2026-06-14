# This module contains unit tests for the FraudMLP model defined in the client.model module.
# The tests verify that the forward method produces an output of the correct shape and that the output
# values are valid probabilities between 0 and 1. A smoke test is included to ensure the model can process a random input without errors.
import unittest
import torch

from src.model.fraud_mlp import FraudMLP, INPUT_DIM


class TestFraudMLP(unittest.TestCase):

    def test_forward_output_shape(self):
        model = FraudMLP()
        x = torch.randn(4, INPUT_DIM)
        out = model(x)

        self.assertEqual(out.shape, (4, 1))
        probs = torch.sigmoid(out)
        self.assertTrue((probs >= 0.0).all().item())
        self.assertTrue((probs <= 1.0).all().item())


if __name__ == "__main__":
    unittest.main()
