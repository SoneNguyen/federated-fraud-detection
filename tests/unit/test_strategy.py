import unittest
from unittest.mock import patch
from typing import cast

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import FitRes

from server.strategy import WeightedFedAvg


class DummyFitRes:
    def __init__(self, num_examples, parameters):
        self.num_examples = num_examples
        self.parameters = parameters


class TestWeightedFedAvg(unittest.TestCase):

    @patch("server.strategy.mlflow.log_metric")
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

        # cast narrows Parameters | None → Parameters for the type checker
        arrays = parameters_to_ndarrays(cast(Parameters, aggregated))
        self.assertIsNotNone(aggregated)   # kept as the runtime safety check
        self.assertEqual(len(arrays), 1)
        np.testing.assert_allclose(arrays[0], np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(info, {})
        mock_log_metric.assert_called_once_with("clients", 2, step=1)


if __name__ == "__main__":
    unittest.main()