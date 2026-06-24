"""Target-aware federated averaging strategy for fraud detection."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Optional, cast

import mlflow
import numpy as np
import torch
from flwr.common import EvaluateIns, FitIns, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.model.fraud_mlp import FraudMLP, is_federated_param
from src.server.aggregation import (
    robust_blended_average_ndarrays,
    stabilize_aggregate_update,
    target_aware_fedavg_weights,
    target_score,
)
from src.server.checkpoint_manager import CheckpointManager
from src.server.client_state import alpha_for_client, client_auprc_history, record_auprc
from src.server.training_schedule import adapt_client_fit_config

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname).1s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class WeightedFedAvg(FedAvg):
    """FedAvg variant that optimizes against the absolute deployment targets."""

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None, **kwargs):
        self.target_auprc = float(kwargs.pop("target_auprc", 0.70))
        self.target_auroc = float(kwargs.pop("target_auroc", 0.90))
        self.target_f1 = float(kwargs.pop("target_f1", 0.70))
        self.high_target_auprc = float(os.environ.get("HIGH_TARGET_AUPRC", "0.85"))
        self.high_target_auroc = float(os.environ.get("HIGH_TARGET_AUROC", "0.95"))
        self.high_target_f1 = float(os.environ.get("HIGH_TARGET_F1", "0.80"))
        self.client_floor_auprc = float(os.environ.get("CLIENT_FLOOR_AUPRC", "0.80"))
        self.client_floor_auroc = float(os.environ.get("CLIENT_FLOOR_AUROC", "0.93"))
        self.client_floor_f1 = float(os.environ.get("CLIENT_FLOOR_F1", "0.75"))
        self.keep_last_rounds = int(kwargs.pop("keep_last_rounds", 12))
        super().__init__(**kwargs)

        self.ckpt = checkpoint_manager or CheckpointManager("outputs/checkpoints")
        self.results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
        self.best_target_score = 0.0
        self.best_auprc = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        self.best_learning_score = float("-inf")
        self.best_loss = float("inf")
        self.best_loss_round = 0
        self.patience_counter = 0
        self.latest_aggregate_path: Path | None = None
        self.latest_fit_summary: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
        self.selection_loss_weight = float(os.environ.get("SELECTION_LOSS_WEIGHT", "1.5"))
        self.stall_window = int(os.environ.get("STALL_WINDOW", "5"))
        self.fairness_weight = float(os.environ.get("FAIRNESS_AGG_WEIGHT", "0.15"))
        self.previous_aggregate: list[np.ndarray] | None = None
        configured_clients = int(os.environ.get("NUM_CLIENTS", "3"))
        self.configured_clients = configured_clients
        default_server_lr = "0.65" if configured_clients >= 50 else "1.0"
        default_update_ratio = "0.035" if configured_clients >= 50 else "0.10"
        self.server_lr = float(os.environ.get("SERVER_AGG_LR", default_server_lr))
        self.current_server_lr = self.server_lr
        self.current_schedule_meta: dict[str, float | str] = {"adaptive_phase": "warmup"}
        self.max_update_ratio = float(os.environ.get("SERVER_MAX_UPDATE_RATIO", default_update_ratio))
        default_trim = "0.10" if configured_clients >= 50 else "0.0"
        default_blend = "0.25" if configured_clients >= 50 else ("0.10" if configured_clients >= 10 else "0.0")
        self.robust_trim_ratio = float(os.environ.get("ROBUST_TRIM_RATIO", default_trim))
        self.robust_median_blend = float(os.environ.get("ROBUST_MEDIAN_BLEND", default_blend))
        self.adaptive_schedule = os.environ.get("ADAPTIVE_SCHEDULE", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.coverage_sampling = os.environ.get(
            "COVERAGE_SAMPLING",
            "1" if configured_clients >= 50 else "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.fit_sample_counts: dict[str, int] = {}
        self.eval_sample_counts: dict[str, int] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject per-client focal alpha into each client's fit config."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config, self.current_server_lr, self.current_schedule_meta = adapt_client_fit_config(
            config,
            self.history,
            server_round=server_round,
            base_server_lr=self.server_lr,
            best_target_score=self.best_target_score,
            configured_clients=self.configured_clients,
            stall_window=self.stall_window,
            enabled=self.adaptive_schedule,
        )
        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = self._sample_clients(
            client_manager=client_manager,
            sample_size=sample_size,
            min_num_clients=min_num_clients,
            counts=self.fit_sample_counts,
            label="fit",
            server_round=server_round,
        )
        fit_instructions = [(client, fit_ins) for client in clients]

        patched = []
        alphas = []
        for fallback_id, (client_proxy, fit_ins) in enumerate(fit_instructions):
            client_id = self._client_numeric_id(client_proxy, fallback_id)
            alpha = alpha_for_client(client_id)
            new_config = dict(fit_ins.config)
            new_config["focal_alpha"] = alpha
            alphas.append(f"{client_id}:{alpha:.2f}")
            patched.append((client_proxy, FitIns(fit_ins.parameters, new_config)))
        logger.info(
            "R%03d cfg phase=%s lr=%.1e ep=%s slr=%.2f alpha=%s",
            server_round,
            config.get("adaptive_phase", "base"),
            float(config.get("lr", 0.0)),
            int(config.get("local_epochs", 0)),
            self.current_server_lr,
            ",".join(alphas),
        )
        return patched

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = self._sample_clients(
            client_manager=client_manager,
            sample_size=sample_size,
            min_num_clients=min_num_clients,
            counts=self.eval_sample_counts,
            label="eval",
            server_round=server_round,
        )
        return [(client, evaluate_ins) for client in clients]

    def _sample_clients(
        self,
        *,
        client_manager,
        sample_size: int,
        min_num_clients: int,
        counts: dict[str, int],
        label: str,
        server_round: int,
    ):
        if not self.coverage_sampling or not hasattr(client_manager, "clients"):
            clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=min_num_clients,
            )
        else:
            if hasattr(client_manager, "wait_for"):
                client_manager.wait_for(min_num_clients)
            available = list(getattr(client_manager, "clients").values())
            available.sort(key=lambda client: (counts.get(str(client.cid), 0), str(client.cid)))
            clients = available[:sample_size]
        for client in clients:
            cid = str(getattr(client, "cid", "unknown"))
            counts[cid] = counts.get(cid, 0) + 1
        if self.coverage_sampling:
            selected = ",".join(str(getattr(client, "cid", "?")) for client in clients[:8])
            logger.info(
                "R%03d %s_sample size=%s unique=%s first=%s",
                server_round,
                label,
                len(clients),
                len(counts),
                selected,
            )
        return clients

    def _client_numeric_id(self, client_proxy, fallback_id: int) -> int:
        cid = str(getattr(client_proxy, "cid", ""))
        if cid.isdigit():
            return int(cid)
        digits = "".join(ch for ch in cid if ch.isdigit())
        return int(digits) if digits else fallback_id

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            logger.warning("R%03d fit_failed=%s", server_round, len(failures))
        if len(results) < 2:
            return None, {}

        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", None)
            if metrics and "val_auprc" in metrics and "client_id" in metrics:
                cid = int(metrics["client_id"])
                auprc = float(metrics["val_auprc"])
                record_auprc(cid, auprc)

        self.latest_fit_summary = self._aggregate_fit_metrics(results)

        result_metrics = [getattr(fit_res, "metrics", None) or {} for _, fit_res in results]
        result_examples = [int(fit_res.num_examples) for _, fit_res in results]
        norm_weights = target_aware_fedavg_weights(
            client_metrics=result_metrics,
            client_examples=result_examples,
            target_auprc=self.target_auprc,
            target_auroc=self.target_auroc,
            target_f1=self.target_f1,
            fairness_weight=self.fairness_weight,
            profile=os.environ.get("TRAINING_PROFILE", "ambitious"),
        )
        total_samples = sum(r.num_examples for _, r in results)
        weight_text = ",".join(f"{w:.3f}" for w in norm_weights)

        client_parameters: list[list[np.ndarray]] = []
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            client_parameters.append(params)

        proposed_agg, robust_meta = robust_blended_average_ndarrays(
            client_parameters,
            norm_weights,
            trim_ratio=self.robust_trim_ratio,
            median_blend=self.robust_median_blend,
        )
        agg, stability = stabilize_aggregate_update(
            previous=self.previous_aggregate,
            proposed=proposed_agg,
            server_lr=self.current_server_lr,
            max_update_ratio=self.max_update_ratio,
        )
        stability.update(robust_meta)
        stability.update(
            {
                key: value
                for key, value in self.current_schedule_meta.items()
                if isinstance(value, (int, float))
            }
        )
        self.previous_aggregate = [np.array(param, copy=True) for param in agg]

        full_state = self._state_from_ndarrays(agg)

        path = self.ckpt.save(
            name=f"round_{server_round:03d}",
            state_dict=full_state,
            metadata={
                "round": server_round,
                "num_clients": len(results),
                "total_samples": total_samples,
                "aggregation_weights": norm_weights,
                "aggregation_stability": stability,
                "schedule": self.current_schedule_meta,
            },
        )
        self.latest_aggregate_path = path

        removed = self.ckpt.prune_round_checkpoints(self.keep_last_rounds)
        fit_train = self.latest_fit_summary.get("fit_train_loss", 0.0)
        fit_delta = self.latest_fit_summary.get("fit_train_loss_delta", 0.0)
        fit_grad = self.latest_fit_summary.get("fit_grad_norm_mean", 0.0)
        logger.info(
            "R%03d fit clients=%s samples=%s train=%.6f delta=%+.6f "
            "grad=%.3f upd=%.3e scale=%.3f slr=%.2f robust=%.2f phase=%s w=%s ckpt=%s pruned=%s",
            server_round,
            len(results),
            total_samples,
            fit_train,
            fit_delta,
            fit_grad,
            stability["server_update_norm"],
            stability["server_update_scale"],
            stability["server_lr"],
            stability.get("robust_median_blend", 0.0),
            self.current_schedule_meta.get("adaptive_phase", "base"),
            weight_text,
            path.name,
            removed,
        )

        mlflow.log_metric("clients", len(results), step=server_round)
        mlflow.log_metric("total_samples", total_samples, step=server_round)
        for name, value in stability.items():
            mlflow.log_metric(name, value, step=server_round)
        return ndarrays_to_parameters(cast(list[np.ndarray], agg)), {}

    def aggregate_evaluate(self, server_round: int, results, failures) -> tuple[float | None, dict[str, Scalar]]:
        if failures:
            logger.warning("R%03d eval_failed=%s ok=%s", server_round, len(failures), len(results))
        if not results:
            logger.warning("R%03d eval_no_results", server_round)
            return None, {}

        total = sum(r.num_examples for _, r in results)
        weighted_loss = 0.0
        metric_sum: dict[str, float] = {}

        for client_id, (_, eval_res) in enumerate(results):
            weighted_loss += eval_res.loss * eval_res.num_examples
            metrics = eval_res.metrics or {}
            cid = int(metrics.get("client_id", client_id))
            for metric_name, metric_value in (eval_res.metrics or {}).items():
                if metric_name == "client_id":
                    continue
                if isinstance(metric_value, (int, float)):
                    metric_sum[metric_name] = (
                        metric_sum.get(metric_name, 0.0)
                        + float(metric_value) * eval_res.num_examples
                    )
            self._record_client_eval(cid, eval_res.num_examples, eval_res.loss, metrics)

        avg_loss = weighted_loss / max(total, 1)
        aggregated_metrics: dict[str, object] = {
            name: value / total for name, value in metric_sum.items()
        }
        aggregated_metrics["val_loss"] = float(avg_loss)
        aggregated_metrics.update(self._client_eval_summary())
        aggregated_metrics.update(self.latest_fit_summary)

        numeric_metrics = self._numeric_record(aggregated_metrics)
        status = self._target_status(numeric_metrics)
        aggregated_metrics.update(status)
        numeric_metrics = self._numeric_record(aggregated_metrics)
        high_status = self._high_target_status(numeric_metrics)
        aggregated_metrics.update(high_status)
        diagnostics = self._learning_diagnostics(server_round, aggregated_metrics)
        aggregated_metrics.update(diagnostics)

        target_score = float(status["target_score"])
        learning_score = self._metric_float(aggregated_metrics, "learning_score", target_score)
        global_auprc = self._metric_float(aggregated_metrics, "val_auprc")
        global_f1 = self._metric_float(aggregated_metrics, "val_f1")
        val_loss = self._metric_float(aggregated_metrics, "val_loss", float("inf"))

        if target_score > self.best_target_score + 0.001:
            self.best_target_score = target_score

        improved = learning_score > self.best_learning_score + 0.001
        if improved:
            self.best_learning_score = learning_score
            self.best_auprc = max(self.best_auprc, global_auprc)
            self.best_f1 = max(self.best_f1, global_f1)
            self.best_round = server_round
            self.patience_counter = 0
            self._persist_best_summary(server_round, total, aggregated_metrics)
            logger.info(
                "R%03d best learn=%.4f target=%.4f loss=%.6f auprc=%.4f auroc=%.4f f1=%.4f",
                server_round,
                learning_score,
                target_score,
                val_loss,
                global_auprc,
                aggregated_metrics.get("val_auroc", 0.0),
                global_f1,
            )
        else:
            self.patience_counter += 1
            if self.patience_counter >= 10:
                logger.warning(
                    "R%03d stale patience=%s best_round=%s best_learn=%.4f",
                    server_round,
                    self.patience_counter,
                    self.best_round,
                    self.best_learning_score,
                )

        if bool(status["target_met"]):
            if val_loss < self.best_loss - 1e-4:
                self.best_loss = val_loss
                self.best_loss_round = server_round
                logger.info(
                    "R%03d best_loss=%.6f",
                    server_round,
                    val_loss,
                )
        if bool(aggregated_metrics.get("high_target_met", False)):
            logger.info("R%03d high_target=1", server_round)

        for name, value in aggregated_metrics.items():
            if isinstance(value, (int, float, bool)):
                mlflow.log_metric(name, float(value), step=server_round)
        logger.info(
            "R%03d eval state=%s target=%s high=%s floor=%s learn=%.4f band=%.4f loss=%.6f "
            "auprc=%.4f/%.4f auroc=%.4f/%.4f f1=%.4f/%.4f thr=%.4f "
            "slope_loss=%+.5f slope_f1=%+.5f train_delta=%+.6f",
            server_round,
            aggregated_metrics.get("learning_state", "n/a"),
            int(bool(status["target_met"])),
            int(bool(aggregated_metrics.get("high_target_met", False))),
            int(bool(aggregated_metrics.get("client_floor_met", False))),
            learning_score,
            aggregated_metrics.get("high_band_score", 0.0),
            val_loss,
            global_auprc,
            aggregated_metrics.get("min_client_auprc", 0.0),
            aggregated_metrics.get("val_auroc", 0.0),
            aggregated_metrics.get("min_client_auroc", 0.0),
            global_f1,
            aggregated_metrics.get("min_client_f1", 0.0),
            aggregated_metrics.get("val_threshold", 0.0),
            aggregated_metrics.get("loss_slope_5", 0.0),
            aggregated_metrics.get("f1_slope_5", 0.0),
            aggregated_metrics.get("fit_train_loss_delta", 0.0),
        )

        self._persist_evaluation_summary(server_round, total, aggregated_metrics)
        self.history.append(self._numeric_record(aggregated_metrics))
        return float(avg_loss), cast(dict[str, Scalar], aggregated_metrics)

    def _target_score(self, metrics: Mapping[str, float]) -> float:
        return target_score(
            auprc=float(metrics.get("val_auprc", 0.0)),
            auroc=float(metrics.get("val_auroc", 0.0)),
            f1=float(metrics.get("val_f1", 0.0)),
            target_auprc=self.target_auprc,
            target_auroc=self.target_auroc,
            target_f1=self.target_f1,
        )

    def _target_status(self, metrics: Mapping[str, float]) -> dict[str, float | bool]:
        auprc = float(metrics.get("val_auprc", 0.0))
        auroc = float(metrics.get("val_auroc", 0.0))
        f1 = float(metrics.get("val_f1", 0.0))
        return {
            "target_score": self._target_score(metrics),
            "target_met": (
                auprc >= self.target_auprc
                and auroc >= self.target_auroc
                and f1 >= self.target_f1
            ),
            "margin_auprc": auprc - self.target_auprc,
            "margin_auroc": auroc - self.target_auroc,
            "margin_f1": f1 - self.target_f1,
            "gap_auprc": max(self.target_auprc - auprc, 0.0),
            "gap_auroc": max(self.target_auroc - auroc, 0.0),
            "gap_f1": max(self.target_f1 - f1, 0.0),
        }

    def _high_target_status(self, metrics: Mapping[str, float]) -> dict[str, float | bool]:
        auprc = float(metrics.get("val_auprc", 0.0))
        auroc = float(metrics.get("val_auroc", 0.0))
        f1 = float(metrics.get("val_f1", 0.0))
        min_auprc = float(metrics.get("min_client_auprc", auprc))
        min_auroc = float(metrics.get("min_client_auroc", auroc))
        min_f1 = float(metrics.get("min_client_f1", f1))

        high_ratios = [
            min(auprc / self.high_target_auprc, 1.0),
            min(auroc / self.high_target_auroc, 1.0),
            min(f1 / self.high_target_f1, 1.0),
            min(min_auprc / self.client_floor_auprc, 1.0),
            min(min_auroc / self.client_floor_auroc, 1.0),
            min(min_f1 / self.client_floor_f1, 1.0),
        ]
        high_band_score = float(
            0.20 * high_ratios[0]
            + 0.15 * high_ratios[1]
            + 0.20 * high_ratios[2]
            + 0.20 * high_ratios[3]
            + 0.10 * high_ratios[4]
            + 0.15 * high_ratios[5]
        )
        return {
            "high_band_score": high_band_score,
            "high_target_met": (
                auprc >= self.high_target_auprc
                and auroc >= self.high_target_auroc
                and f1 >= self.high_target_f1
            ),
            "client_floor_met": (
                min_auprc >= self.client_floor_auprc
                and min_auroc >= self.client_floor_auroc
                and min_f1 >= self.client_floor_f1
            ),
            "gap_high_auprc": max(self.high_target_auprc - auprc, 0.0),
            "gap_high_auroc": max(self.high_target_auroc - auroc, 0.0),
            "gap_high_f1": max(self.high_target_f1 - f1, 0.0),
            "gap_client_auprc": max(self.client_floor_auprc - min_auprc, 0.0),
            "gap_client_auroc": max(self.client_floor_auroc - min_auroc, 0.0),
            "gap_client_f1": max(self.client_floor_f1 - min_f1, 0.0),
        }

    def _learning_score(self, metrics: Mapping[str, float]) -> float:
        target_score = float(metrics.get("target_score", self._target_score(metrics)))
        if not bool(metrics.get("target_met", False)):
            return target_score

        high_band_score = float(metrics.get("high_band_score", 0.0))
        auprc_ratio = float(metrics.get("val_auprc", 0.0)) / self.target_auprc
        auroc_ratio = float(metrics.get("val_auroc", 0.0)) / self.target_auroc
        f1_ratio = float(metrics.get("val_f1", 0.0)) / self.target_f1
        margin_score = 0.35 * auprc_ratio + 0.20 * auroc_ratio + 0.45 * f1_ratio
        loss_penalty = self.selection_loss_weight * max(float(metrics.get("val_loss", 0.0)), 0.0)
        train_delta_bonus = 0.05 * min(max(float(metrics.get("fit_train_loss_delta", 0.0)), 0.0), 1.0)
        client_floor_bonus = 0.20 * high_band_score
        return float(margin_score + client_floor_bonus - loss_penalty + train_delta_bonus)

    def _learning_diagnostics(
        self,
        server_round: int,
        metrics: Mapping[str, object],
    ) -> dict[str, float | str]:
        numeric_current = self._numeric_record(metrics)
        learning_score = self._learning_score(numeric_current)
        history = [*self.history, numeric_current]
        loss_slope = self._slope(history, "val_loss", self.stall_window)
        f1_slope = self._slope(history, "val_f1", self.stall_window)
        auprc_slope = self._slope(history, "val_auprc", self.stall_window)
        train_loss_slope = self._slope(history, "fit_train_loss", self.stall_window)

        if loss_slope < -5e-4 and (f1_slope >= -1e-3 or auprc_slope >= -1e-3):
            state = "learning"
        elif loss_slope > 5e-4 and (f1_slope < -1e-3 or auprc_slope < -1e-3):
            state = "regressing"
        elif (
            abs(loss_slope) <= 5e-4
            and abs(f1_slope) <= 1e-3
            and abs(auprc_slope) <= 1e-3
        ):
            state = "stalled"
        else:
            state = "mixed"

        val_loss = float(numeric_current.get("val_loss", float("inf")))
        best_loss = min(self.best_loss, val_loss)
        best_loss_round = (
            server_round
            if val_loss <= self.best_loss
            else self.best_loss_round
        )
        rounds_since_best_loss = (
            0 if best_loss_round == 0 else max(server_round - best_loss_round, 0)
        )
        return {
            "learning_score": learning_score,
            "loss_slope_5": loss_slope,
            "f1_slope_5": f1_slope,
            "auprc_slope_5": auprc_slope,
            "train_loss_slope_5": train_loss_slope,
            "learning_state": state,
            "best_val_loss": best_loss,
            "best_val_loss_round": float(best_loss_round),
            "rounds_since_best_loss": float(rounds_since_best_loss),
        }

    def _record_client_eval(
        self,
        cid: int,
        examples: int,
        loss: float,
        metrics: dict[str, Scalar],
    ) -> None:
        if not hasattr(self, "_round_client_eval"):
            self._round_client_eval = {}
        self._round_client_eval[cid] = {
            "examples": float(examples),
            "loss": float(loss),
            "auprc": float(metrics.get("val_auprc", 0.0)),
            "auroc": float(metrics.get("val_auroc", 0.0)),
            "f1": float(metrics.get("val_f1", 0.0)),
        }

    def _client_eval_summary(self) -> dict[str, float]:
        client_eval = getattr(self, "_round_client_eval", {})
        self._round_client_eval = {}
        if not client_eval:
            return {}
        auprcs = np.array([v["auprc"] for v in client_eval.values()], dtype=np.float64)
        aurocs = np.array([v["auroc"] for v in client_eval.values()], dtype=np.float64)
        f1s = np.array([v["f1"] for v in client_eval.values()], dtype=np.float64)
        losses = np.array([v["loss"] for v in client_eval.values()], dtype=np.float64)
        out = {
            "min_client_auprc": float(np.min(auprcs)),
            "max_client_auprc": float(np.max(auprcs)),
            "std_client_auprc": float(np.std(auprcs)),
            "min_client_auroc": float(np.min(aurocs)),
            "max_client_auroc": float(np.max(aurocs)),
            "std_client_auroc": float(np.std(aurocs)),
            "min_client_f1": float(np.min(f1s)),
            "max_client_f1": float(np.max(f1s)),
            "std_client_f1": float(np.std(f1s)),
            "max_client_loss": float(np.max(losses)),
            "std_client_loss": float(np.std(losses)),
        }
        for cid, metrics in sorted(client_eval.items()):
            out[f"client_{cid}_auprc"] = metrics["auprc"]
            out[f"client_{cid}_auroc"] = metrics["auroc"]
            out[f"client_{cid}_f1"] = metrics["f1"]
            out[f"client_{cid}_loss"] = metrics["loss"]
        return out

    def _aggregate_fit_metrics(self, results) -> dict[str, float]:
        metric_names = {
            "train_loss",
            "train_loss_start",
            "train_loss_end",
            "train_loss_delta",
            "grad_norm_mean",
            "fit_lr",
            "fit_lr_final",
            "fit_local_epochs",
            "fit_bce_mix",
            "fit_focal_gamma",
        }
        totals: dict[str, float] = {}
        weights: dict[str, int] = {}
        for _, fit_res in results:
            metrics = getattr(fit_res, "metrics", None) or {}
            for name in metric_names:
                value = metrics.get(name)
                if isinstance(value, (int, float)):
                    out_name = name if name.startswith("fit_") else f"fit_{name}"
                    totals[out_name] = totals.get(out_name, 0.0) + float(value) * fit_res.num_examples
                    weights[out_name] = weights.get(out_name, 0) + fit_res.num_examples
        return {
            name: totals[name] / max(weights[name], 1)
            for name in sorted(totals)
        }

    def _numeric_record(self, metrics: Mapping[str, object]) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float, bool))
        }

    def _metric_float(
        self,
        metrics: Mapping[str, object],
        name: str,
        default: float = 0.0,
    ) -> float:
        value = metrics.get(name, default)
        if isinstance(value, (int, float, bool)):
            return float(value)
        return default

    def _slope(
        self,
        history: list[dict[str, float]],
        key: str,
        window: int,
    ) -> float:
        values = [
            record[key]
            for record in history[-max(window, 2):]
            if key in record and np.isfinite(record[key])
        ]
        if len(values) < 2:
            return 0.0
        return float((values[-1] - values[0]) / (len(values) - 1))

    def _state_from_ndarrays(self, ndarrays: list[np.ndarray]) -> dict[str, torch.Tensor]:
        model = FraudMLP()
        full_state = model.state_dict()
        trainable_keys = [k for k in full_state.keys() if is_federated_param(k)]
        for k, v in zip(trainable_keys, ndarrays):
            full_state[k] = torch.tensor(v)
        return full_state

    def _persist_evaluation_summary(
        self, server_round: int, total_examples: int, metrics: Mapping[str, object]
    ) -> None:
        output_dir = self.results_dir
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

        latest_path = output_dir / "latest_metrics.json"
        latest_path.write_text(json.dumps(record, indent=2))
        self._persist_training_summary(history)

    def _persist_best_summary(
        self, server_round: int, total_examples: int, metrics: Mapping[str, object]
    ) -> None:
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = output_dir / "best_round.json"
        record = {
            "round": server_round,
            "checkpoint": f"round_{server_round:03d}.pt",
            "total_examples": total_examples,
            **metrics,
        }
        best_path.write_text(json.dumps(record, indent=2))

    def _persist_training_summary(self, history: list[dict]) -> None:
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [row for row in history if isinstance(row, dict)]
        ranked = sorted(
            rows,
            key=lambda row: (
                bool(row.get("target_met", False)),
                float(row.get("learning_score", 0.0) or 0.0),
                float(row.get("val_auprc", 0.0) or 0.0),
                float(row.get("val_f1", 0.0) or 0.0),
            ),
            reverse=True,
        )
        latest = rows[-1] if rows else {}
        best = ranked[0] if ranked else {}
        lines = [
            "# Training Summary",
            "",
            "| Item | Round | Checkpoint | Loss | AUPRC | AUROC | F1 | Target | High | Floor |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, row in (("Latest", latest), ("Best", best)):
            if not row:
                continue
            round_no = int(float(row.get("round", 0)))
            lines.append(
                "| {label} | {round_no} | round_{round_no:03d}.pt | {loss:.6f} | {auprc:.4f} | {auroc:.4f} | {f1:.4f} | {target} | {high} | {floor} |".format(
                    label=label,
                    round_no=round_no,
                    loss=float(row.get("val_loss", 0.0) or 0.0),
                    auprc=float(row.get("val_auprc", 0.0) or 0.0),
                    auroc=float(row.get("val_auroc", 0.0) or 0.0),
                    f1=float(row.get("val_f1", 0.0) or 0.0),
                    target=int(bool(row.get("target_met", False))),
                    high=int(bool(row.get("high_target_met", False))),
                    floor=int(bool(row.get("client_floor_met", False))),
                )
            )
        lines.extend(
            [
                "",
                "| Round | Checkpoint | Loss | AUPRC | AUROC | F1 | Learning | State |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in ranked[:10]:
            round_no = int(float(row.get("round", 0)))
            lines.append(
                "| {round_no} | round_{round_no:03d}.pt | {loss:.6f} | {auprc:.4f} | {auroc:.4f} | {f1:.4f} | {learn:.4f} | {state} |".format(
                    round_no=round_no,
                    loss=float(row.get("val_loss", 0.0) or 0.0),
                    auprc=float(row.get("val_auprc", 0.0) or 0.0),
                    auroc=float(row.get("val_auroc", 0.0) or 0.0),
                    f1=float(row.get("val_f1", 0.0) or 0.0),
                    learn=float(row.get("learning_score", 0.0) or 0.0),
                    state=str(row.get("learning_state", "")),
                )
            )
        (output_dir / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
