import flwr as fl, numpy as np, mlflow, json

with open("contracts/schema.json") as f:
    _s = json.load(f)
INPUT_DIM = _s["feature_schema"]["total_features"]  # 11

class WeightedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if len(results) < 2:
            print(f"[Round {server_round}] Too few clients — skip")
            return None, {}
        total = sum(r.num_examples for _, r in results)
        weighted = []
        for _, fit_res in results:
            w = fit_res.num_examples / total
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weighted.append([p * w for p in params])
        agg = [sum(w[i] for w in weighted) for i in range(len(weighted[0]))]
        mlflow.log_metric("clients", len(results), step=server_round)
        print(f"[Round {server_round}] {len(results)} clients, {total} samples")
        return fl.common.ndarrays_to_parameters(agg), {}
