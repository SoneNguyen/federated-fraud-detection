# This module contains unit tests for the WeightedFedAvg strategy defined in the server.strategy module.
# The tests verify that the aggregate_fit method correctly aggregates client updates into a single Parameters object,
# that the logged metrics are correct, and that the method handles edge cases such as too few clients appropriately.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from typing import cast

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import FitRes

from src.server.strategy import WeightedFedAvg
from src.server.checkpoint_manager import CheckpointManager


class DummyFitRes:
    def __init__(self, num_examples, parameters):
        self.num_examples = num_examples
        self.parameters = parameters


class DummyClient:
    def __init__(self, cid):
        self.cid = str(cid)


class DummyClientManager:
    def __init__(self, count):
        self.clients = {str(cid): DummyClient(cid) for cid in range(count)}

    def num_available(self):
        return len(self.clients)

    def wait_for(self, _min_num_clients):
        return True

    def sample(self, num_clients, min_num_clients=None):
        return list(self.clients.values())[:num_clients]


class TestWeightedFedAvg(unittest.TestCase):

    @patch("src.server.strategy.mlflow.log_metric")
    def test_aggregate_fit_returns_parameter_object(self, mock_log_metric):
        params = ndarrays_to_parameters([np.array([1.0, 2.0], dtype=np.float32)])

        # Cast satisfies the invariant list type without runtime overhead
        results = cast(
            list[tuple[ClientProxy, FitRes]],
            [
                (None, DummyFitRes(num_examples=2, parameters=params)),
                (None, DummyFitRes(num_examples=6, parameters=params)),
            ],
        )

        strategy = WeightedFedAvg()
        aggregated, info = strategy.aggregate_fit(1, results, [])

        # cast narrows Parameters | None to Parameters for the type checker.
        arrays = parameters_to_ndarrays(cast(Parameters, aggregated))
        self.assertIsNotNone(aggregated)   # kept as the runtime safety check
        self.assertEqual(len(arrays), 1)
        np.testing.assert_allclose(arrays[0], np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(info, {})
        self.assertGreaterEqual(mock_log_metric.call_count, 2)
        mock_log_metric.assert_any_call("clients", 2, step=1)
        mock_log_metric.assert_any_call("total_samples", 8, step=1)
        mock_log_metric.assert_any_call("server_lr", 1.0, step=1)

    @patch("src.server.strategy.mlflow.log_metric")
    def test_aggregate_fit_saves_only_global_round_checkpoint(self, _mock_log_metric):
        params = ndarrays_to_parameters([np.array([1.0, 2.0], dtype=np.float32)])
        results = cast(
            list[tuple[ClientProxy, FitRes]],
            [
                (None, DummyFitRes(num_examples=2, parameters=params)),
                (None, DummyFitRes(num_examples=6, parameters=params)),
            ],
        )

        with TemporaryDirectory() as tmp:
            strategy = WeightedFedAvg(checkpoint_manager=CheckpointManager(tmp))
            strategy.aggregate_fit(1, results, [])
            names = sorted(path.name for path in Path(tmp).iterdir())

        self.assertIn("round_001.pt", names)
        self.assertIn("round_001.json", names)
        self.assertFalse(any(name.startswith("client_") for name in names))
        legacy_target_prefix = "target" + "_met_"
        self.assertFalse(any(name.startswith(legacy_target_prefix) for name in names))
        self.assertFalse(any(name.startswith("best_") for name in names))

    def test_coverage_sampling_rotates_clients(self):
        params = ndarrays_to_parameters([np.array([1.0, 2.0], dtype=np.float32)])
        manager = DummyClientManager(10)
        strategy = WeightedFedAvg(
            fraction_fit=0.3,
            min_fit_clients=3,
            min_available_clients=3,
        )
        strategy.coverage_sampling = True

        first = strategy.configure_fit(1, params, manager)
        second = strategy.configure_fit(2, params, manager)
        third = strategy.configure_fit(3, params, manager)

        self.assertEqual([client.cid for client, _ in first], ["0", "1", "2"])
        self.assertEqual([client.cid for client, _ in second], ["3", "4", "5"])
        self.assertEqual([client.cid for client, _ in third], ["6", "7", "8"])


if __name__ == "__main__":
    unittest.main()
