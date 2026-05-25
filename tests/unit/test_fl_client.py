import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from client.fl_client import FraudClient
from client.model import FraudMLP


class TestFraudClient(unittest.TestCase):

    def setUp(self):
        self.model = FraudMLP()
        x = torch.randn(10, self.model.net[0].in_features)
        y = torch.randint(0, 2, (10,), dtype=torch.float32)
        dataset = TensorDataset(x, y)
        self.loader = DataLoader(dataset, batch_size=5)
        self.client = FraudClient(self.model, self.loader, self.loader)

    def test_get_and_set_parameters_work(self):
        params = [p.copy() for p in self.client.get_parameters()]
        self.assertTrue(isinstance(params, list))
        self.assertTrue(all(hasattr(p, 'shape') for p in params))

        modified = [p + 0.1 for p in params]
        self.client.set_parameters(modified)
        updated = self.client.get_parameters()
        self.assertTrue(
            any(not torch.allclose(torch.from_numpy(u), torch.from_numpy(p))
                for u, p in zip(updated, params))
        )

    def test_fit_and_evaluate_return_counts(self):
        params = self.client.get_parameters()
        updated_params, num_examples, _ = self.client.fit(params, {"lr": 0.01})
        self.assertEqual(num_examples, len(self.loader.dataset))
        loss, eval_examples, _ = self.client.evaluate(updated_params, {})
        self.assertEqual(eval_examples, len(self.loader.dataset))
        self.assertIsInstance(loss, float)


if __name__ == "__main__":
    unittest.main()
