"""Weighted Federated Averaging Strategy with per-client adaptation."""

import json
import logging
from pathlib import Path

import numpy as np
import mlflow
import torch
from flwr.server import ClientManager
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Optional, cast

from src.model.fraud_mlp import FraudMLP
from src.server.checkpoint_manager import CheckpointManager
from src.server.client_state import client_auprc_history, alpha_for_client, record_auprc
from src.model.fraud_mlp import FraudMLP, is_federated_param

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load schema
config_dir = Path(__file__).parent.parent.parent / "config"
with open(config_dir / "schema.json") as f:
    _s = json.load(f)
INPUT_DIM = _s["feature_schema"]["total_features"]


class WeightedFedAvg(FedAvg):
    """
    Federated Averaging with:
    - AUPRC-weighted aggregation
    - Per-client focal_alpha adaptation
    - Automatic checkpoint saving
    """

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.ckpt = checkpoint_manager or CheckpointManager("checkpoints")
        self.best_auprc = 0.0
        self.patience_counter = 0
        self.best_round = 0

    def configure_fit(self, server_round, parameters, client_manager):
        """Override to inject per-client focal_alpha."""
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        patched = []
        for client_id, (client_proxy, fit_ins) in enumerate(fit_instructions):
            alpha = alpha_for_client(client_id)
            new_config = dict(fit_ins.config)
            new_config["focal_alpha"] = alpha
            logger.info(
                f"[Round {server_round}] Client {client_id}: focal_alpha={alpha:.3f} "
                f"(history={list(client_auprc_history.get(client_id, []))})"
            )
            patched.append((client_proxy, FitIns(fit_ins.parameters, new_config)))
        return patched

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} clients failed")
        if len(results) < 2:
            return None, {}

        # ── update per-client AUPRC history ──────────────────────────────────
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", None)
            if metrics and "val_auprc" in metrics and "client_id" in metrics:
                cid   = int(metrics["client_id"])
                auprc = float(metrics["val_auprc"])
                record_auprc(cid, auprc)
                logger.info(
                    f"[Round {server_round}] Client {cid} AUPRC history: "
                    f"{list(client_auprc_history[cid])}"
                )

        # Pull AUPRC from fit metrics for aggregation weighting
        weights: list[float] = []
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", None)
            auprc = float(metrics.get("val_auprc", 0.0)) if metrics else 0.0
            n = fit_res.num_examples
            # Only reward AUPRC above 0.55 baseline — clients near random get near-zero weight
            effective_auprc = max(auprc - 0.55, 0.02)
            weights.append(effective_auprc * (n ** 0.5))

        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        total_samples = sum(r.num_examples for _, r in results)
        logger.info(f"[Round {server_round}] AUPRC weights: {[f'{w:.3f}' for w in norm_weights]}")

        weighted: list[list[np.ndarray]] = []
        for w, (_, fit_res) in zip(norm_weights, results):
            params = parameters_to_ndarrays(fit_res.parameters)
            weighted.append([p * w for p in params])

        agg: list[np.ndarray] = [
            sum((w[i] for w in weighted), np.zeros_like(weighted[0][i]))
            for i in range(len(weighted[0]))
        ]

        # ── save checkpoint ───────────────────────────────────────────────────
        # Reconstruct a state_dict from the aggregated numpy arrays.
        # Since get_parameters excludes BN running stats, we only populate
        # trainable params; BN stats stay at init (will be rebuilt from data).
        model = FraudMLP()
        full_state = model.state_dict()
        trainable_keys = [
            k for k in full_state.keys()
            if "running_mean" not in k
            and "running_var" not in k
            and "num_batches_tracked" not in k
        ]

        # Populate trainable params from aggregation
        for k, v in zip(trainable_keys, agg):
            full_state[k] = torch.tensor(v)

        # BN buffers remain at initialization values — they'll be rebuilt
        # from the first ~20 batches of training at each client.
        state_dict = full_state
        name = f"round_{server_round:03d}"
        path = self.ckpt.save(
            name=name,
            state_dict=state_dict,
            metadata={
                "round": server_round,
                "num_clients": len(results),
                "total_samples": total_samples,
            },
        )
        logger.info(f"[Round {server_round}] Checkpoint saved → {path.name}")

        mlflow.log_metric("clients", len(results), step=server_round)
        mlflow.log_metric("total_samples", total_samples, step=server_round)
        logger.info(f"[Round {server_round}] Aggregation complete")
        return ndarrays_to_parameters(cast(list[np.ndarray], agg)), {}

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

        global_auprc = aggregated_metrics.get("val_auprc", 0.0)
        if global_auprc > self.best_auprc + 0.001:
            self.best_auprc = global_auprc
            self.best_round = server_round
            self.patience_counter = 0
            # tag the checkpoint
            best_path = Path("outputs/checkpoints") / f"best_round_{server_round:03d}.pt"
            latest = self.ckpt.latest()
            if latest:
                import shutil
                shutil.copy(latest, best_path)
            logger.info(f"[Round {server_round}] New best AUPRC: {global_auprc:.4f} → saved to {best_path.name}")
        else:
            self.patience_counter += 1
            if self.patience_counter >= 10:
                logger.warning(f"[Round {server_round}] No improvement for 10 rounds (best={self.best_auprc:.4f} at round {self.best_round})")

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
