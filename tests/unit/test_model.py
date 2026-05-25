import unittest
import torch

from client.model import FraudMLP, INPUT_DIM


class TestFraudMLP(unittest.TestCase):

    def test_forward_output_shape(self):
        model = FraudMLP()
        x = torch.randn(4, INPUT_DIM)
        out = model(x)

        self.assertEqual(out.shape, (4, 1))
        self.assertTrue((out >= 0.0).all().item())
        self.assertTrue((out <= 1.0).all().item())


if __name__ == "__main__":
    unittest.main()
