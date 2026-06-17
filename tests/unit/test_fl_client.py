# This module contains unit tests for the FraudClient class in the client.fl_client module.
# The tests verify that the get_parameters and set_parameters methods work correctly, that the fit method returns the correct number of examples,
# and that the evaluate method returns a loss value and the correct number of evaluation examples.
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.client.client import FocalLoss, FraudClient
from src.model.fraud_mlp import FraudMLP


class TestFraudClient(unittest.TestCase):

    def setUp(self):
        self.model = FraudMLP()

        first_layer = self.model.input_proj
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

        updated_params, num_examples, fit_metrics = self.client.fit(
            params, {"lr": 0.01, "focal_alpha": 0.8}
        )
        self.assertEqual(num_examples, n)
        self.assertEqual(self.client.focal_loss.alpha, 0.8)
        self.assertIn("client_id", fit_metrics)

        loss, eval_examples, _ = self.client.evaluate(updated_params, {})
        self.assertEqual(eval_examples, n)
        self.assertIsInstance(loss, float)

    def test_focal_loss_alpha_weights_positive_and_negative_classes(self):
        logits = torch.tensor([0.0, 0.0])
        labels = torch.tensor([1.0, 0.0])

        high_positive_alpha = FocalLoss(alpha=0.8, gamma=0.0)(logits, labels)
        low_positive_alpha = FocalLoss(alpha=0.2, gamma=0.0)(logits, labels)

        self.assertTrue(torch.allclose(high_positive_alpha, low_positive_alpha))

        positive_only_high = FocalLoss(alpha=0.8, gamma=0.0)(
            logits[:1], labels[:1]
        )
        positive_only_low = FocalLoss(alpha=0.2, gamma=0.0)(
            logits[:1], labels[:1]
        )
        self.assertGreater(positive_only_high.item(), positive_only_low.item())


if __name__ == "__main__":
    unittest.main()
