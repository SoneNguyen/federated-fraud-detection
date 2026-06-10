import json
import logging
import numpy as np
import mlflow
import torch
from pathlib import Path
from flwr.common import Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from collections import OrderedDict
from typing import Optional, cast

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
INPUT_DIM = _s["feature_schema"]["total_features"]  # 13


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

    def aggregate_evaluate(self, server_round: int, results, failures) -> tuple[float | None, dict[str, Scalar]]:
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} clients failed evaluation — continuing with {len(results)}")
        if not results:
            logger.warning(f"[Round {server_round}] No evaluation results received")
            return None, {}

        total = sum(r.num_examples for _, r in results)
        weighted_loss = 0.0
        metric_sum: dict[str, float] = {}

        for client_id, (_, eval_res) in enumerate(results):
            w = eval_res.num_examples / total
            weighted_loss += eval_res.loss * eval_res.num_examples
            for metric_name, metric_value in (eval_res.metrics or {}).items():
                if isinstance(metric_value, (int, float)):
                    metric_sum[metric_name] = metric_sum.get(metric_name, 0.0) + float(metric_value) * eval_res.num_examples
            logger.info(f"  Client {client_id}: samples={eval_res.num_examples}, loss={eval_res.loss:.6f}, metrics={eval_res.metrics}")

        avg_loss = weighted_loss / max(total, 1)
        aggregated_metrics = {name: value / total for name, value in metric_sum.items()}
        aggregated_metrics["val_loss"] = float(avg_loss)

        for name, value in aggregated_metrics.items():
            if name != "val_loss":
                mlflow.log_metric(name, value, step=server_round)
            logger.info(f"[Round {server_round}] {name}={value:.6f}")

        self._persist_evaluation_summary(server_round, total, aggregated_metrics)
        return float(avg_loss), cast(dict[str, Scalar], aggregated_metrics)

    def _persist_evaluation_summary(self, server_round: int, total_examples: int, metrics: dict[str, float]) -> None:
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "evaluation_history.json"
        record = {
            "round": server_round,
            "total_examples": total_examples,
            **metrics,
        }
        history = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except Exception:
                history = []
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))
