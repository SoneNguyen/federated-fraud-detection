import json
import logging
import numpy as np
import mlflow
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from collections import OrderedDict
from typing import Optional

from client.model import FraudMLP
from server.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

with open("contracts/schema.json") as f:
    _s = json.load(f)
INPUT_DIM = _s["feature_schema"]["total_features"]  # 11


class WeightedFedAvg(FedAvg):
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.ckpt = checkpoint_manager or CheckpointManager("checkpoints")

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} clients failed — continuing with {len(results)}")
        if len(results) < 2:
            logger.warning(f"[Round {server_round}] Too few clients ({len(results)}) — skipping aggregation")
            return None, {}

        total = sum(r.num_examples for _, r in results)
        logger.info(f"[Round {server_round}] Aggregating {len(results)} clients, {total} total samples")

        weighted: list[list[np.ndarray]] = []
        for client_id, (_, fit_res) in enumerate(results):
            w = fit_res.num_examples / total
            params: list[np.ndarray] = parameters_to_ndarrays(fit_res.parameters)
            weighted.append([p * w for p in params])
            logger.debug(f"  Client {client_id}: {fit_res.num_examples} samples, weight={w:.4f}")

        agg: list[np.ndarray] = [
            sum((w[i] for w in weighted), np.zeros_like(weighted[0][i]))
            for i in range(len(weighted[0]))
        ]

        # ── save checkpoint ───────────────────────────────────────────────────
        # Reconstruct a state_dict from the aggregated numpy arrays so
        # torch.load() produces something the model can load directly.
        model = FraudMLP()
        keys = list(model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, agg)}
        )
        name = f"round_{server_round:03d}"
        path = self.ckpt.save(
            name=name,
            state_dict=state_dict,
            metadata={
                "round": server_round,
                "num_clients": len(results),
                "total_samples": total,
            },
        )
        logger.info(f"[Round {server_round}] Checkpoint saved → {path.name}")
        # ─────────────────────────────────────────────────────────────────────

        mlflow.log_metric("clients", len(results), step=server_round)
        mlflow.log_metric("total_samples", total, step=server_round)
        logger.info(f"[Round {server_round}] Aggregation complete")
        return ndarrays_to_parameters(agg), {}