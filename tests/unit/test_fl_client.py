# This module contains unit tests for the FraudClient class in the client.fl_client module.
# The tests verify that the get_parameters and set_parameters methods work correctly, that the fit method returns the correct number of examples,
# and that the evaluate method returns a loss value and the correct number of evaluation examples.
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from client.fl_client import FraudClient
from client.model import FraudMLP


class TestFraudClient(unittest.TestCase):

    def setUp(self):
        self.model = FraudMLP()

        first_layer = self.model.net[0]
        assert isinstance(first_layer, nn.Linear), "Expected first layer to be nn.Linear"
        in_features: int = first_layer.in_features  # ← now typed as int, not Module

        x = torch.randn(10, in_features)
        y = torch.randint(0, 2, (10,), dtype=torch.float32)
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=5)
        self.client = FraudClient(self.model, self.loader, self.loader)

    def test_get_and_set_parameters_work(self):
        params = [p.copy() for p in self.client.get_parameters()]
        self.assertTrue(isinstance(params, list))
        self.assertTrue(all(hasattr(p, "shape") for p in params))

        modified = [p + 0.1 for p in params]
        self.client.set_parameters(modified)
        updated = self.client.get_parameters()
        self.assertTrue(
            any(
                not torch.allclose(torch.from_numpy(u), torch.from_numpy(p))
                for u, p in zip(updated, params)
            )
        )

    def test_fit_and_evaluate_return_counts(self):
        params = self.client.get_parameters()
        dataset = self.loader.dataset
        assert isinstance(dataset, TensorDataset)          # ← narrows to Sized subtype
        n = len(dataset)

        updated_params, num_examples, _ = self.client.fit(params, {"lr": 0.01})
        self.assertEqual(num_examples, n)

        loss, eval_examples, _ = self.client.evaluate(updated_params, {})
        self.assertEqual(eval_examples, n)
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()