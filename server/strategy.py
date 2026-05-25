import json
import numpy as np
import mlflow
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

with open("contracts/schema.json") as f:
    _s = json.load(f)
INPUT_DIM = _s["feature_schema"]["total_features"]  # 11

class WeightedFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if len(results) < 2:
            print(f"[Round {server_round}] Too few clients — skip")
            return None, {}

        total = sum(r.num_examples for _, r in results)

        weighted: list[list[np.ndarray]] = []
        for _, fit_res in results:
            w = fit_res.num_examples / total
            params: list[np.ndarray] = parameters_to_ndarrays(fit_res.parameters)
            weighted.append([p * w for p in params])

        agg: list[np.ndarray] = [
            sum((w[i] for w in weighted), np.zeros_like(weighted[0][i]))  # ← fix
            for i in range(len(weighted[0]))
        ]

        mlflow.log_metric("clients", len(results), step=server_round)
        print(f"[Round {server_round}] {len(results)} clients, {total} samples")
        return ndarrays_to_parameters(agg), {}