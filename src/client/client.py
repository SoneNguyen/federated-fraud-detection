"""Federated Learning Client Module"""
from contextlib import nullcontext
from collections.abc import Iterable, Sized
from typing import cast
import logging
import os

from flwr.client import NumPyClient
from flwr.common import Scalar
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler
from src.data.dataset import FraudDataset
from src.model.fraud_mlp import FraudMLP, is_federated_param, federated_params
from src.client.metrics import (
    average_precision_score_np,
    precision_recall_curve_np,
    roc_auc_score_np,
)

CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))

logger = logging.getLogger(f"Client{CLIENT_ID}")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f'[C{CLIENT_ID:1d}] %(asctime)s %(levelname).1s %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)


class FocalLoss(torch.nn.Module):
    """
    Focal loss for class-imbalanced binary classification.

    gamma=2.0 lowers suppression of easy negatives,
    which is important in federated setting where each client sees fewer fraud
    samples per round. gamma=3.0 over-focuses and starves gradients early.

    alpha=0.5 applies less aggressive positive-class weighting,
    allowing natural data distribution to guide training better.
    """
    def __init__(self, alpha: float, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce    = self.bce(logits, target)
        prob   = torch.sigmoid(logits)
        pt     = torch.where(target == 1, prob, 1 - prob)
        alpha_t = torch.where(
            target == 1,
            torch.as_tensor(self.alpha, device=target.device, dtype=target.dtype),
            torch.as_tensor(1.0 - self.alpha, device=target.device, dtype=target.dtype),
        )
        weight = alpha_t * (1 - pt) ** self.gamma
        return (weight * bce).mean()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _make_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def make_weighted_sampler(
    labels: np.ndarray,
    positive_multiplier: float = 5.0,
    positive_cap: float = 0.25,
    seed: int = 42,
) -> WeightedRandomSampler:
    """
    Return a WeightedRandomSampler that oversamples the minority (fraud) class
    to improve gradient signal without over-distorting the training distribution.

    Strategy: oversample positives by at most 2.5x their natural rate,
    capped at 15% (less aggressive than before). This keeps the training
    distribution much closer to the real one, improving calibration.
    """
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    if n_pos == 0:
        # No fraud examples are available, so use uniform sampling.
        return WeightedRandomSampler(
            weights=np.ones(len(labels)).tolist(),
            num_samples=len(labels),
            replacement=True,
        )

    natural_rate = n_pos / len(labels)
    # Target: oversample positives by at most 2.5x their natural rate,
    # capped at 15% (down from 30%).
    target_rate = min(natural_rate * positive_multiplier, positive_cap)

    w_pos = target_rate / n_pos
    w_neg = (1.0 - target_rate) / max(n_neg, 1)
    weights = np.where(labels == 1, w_pos, w_neg).astype(np.float32).tolist()
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def labels_from_dataset(dataset) -> np.ndarray:
    """Return labels without iterating row by row when tensors are available."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = torch.as_tensor(list(dataset.indices), dtype=torch.long)
        if isinstance(base, FraudDataset):
            return base.y[indices].detach().cpu().numpy().astype(int)
        if isinstance(base, TensorDataset) and len(base.tensors) >= 2:
            return base.tensors[1][indices].detach().cpu().numpy().astype(int)

    if isinstance(dataset, FraudDataset):
        return dataset.y.detach().cpu().numpy().astype(int)

    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        return dataset.tensors[1].detach().cpu().numpy().astype(int)

    iterable = cast(Iterable[tuple[torch.Tensor, torch.Tensor]], dataset)
    return np.fromiter((float(y) for _, y in iterable), dtype=np.float32).astype(int)


class FraudClient(NumPyClient):
    """
    Federated learning client for fraud detection.

    Key features:
    - FocalLoss with gamma=2.0 for imbalanced data
    - WeightedRandomSampler for stable oversampling per batch
    - BN stats excluded from federation (client-local)
    - Gradient clipping at 1.0 for stability
    """

    def __init__(
        self,
        model: FraudMLP,
        train_dataset,
        val_loader: DataLoader,
        local_epochs: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
    ):
        self.model        = model
        self.val_loader   = val_loader
        self.local_epochs = local_epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.focal_loss   = FocalLoss(alpha=0.5, gamma=2.0)
        self.use_amp = self.model.device.type == "cuda" and _env_flag("USE_AMP", True)
        self.pin_memory = self.model.device.type == "cuda"
        self.loss_mode = os.environ.get("LOSS_MODE", "hybrid").strip().lower()
        self.bce_mix = float(os.environ.get("BCE_MIX", "0.30"))
        self.local_scheduler = os.environ.get("LOCAL_LR_SCHEDULER", "cosine").strip().lower()
        self.min_lr_ratio = float(os.environ.get("MIN_LR_RATIO", "0.15"))
        self.num_workers = int(os.environ.get(
            "NUM_WORKERS",
            "0" if os.name == "nt" else "2",
        ))
        self.prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "4"))

        # If train_dataset is a DataLoader, extract the underlying dataset
        if isinstance(train_dataset, DataLoader):
            train_dataset = train_dataset.dataset

        self.dataset_size = len(cast(Sized, train_dataset))
        labels = labels_from_dataset(train_dataset)
        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        pos_weight_cap = float(os.environ.get("POS_WEIGHT_CAP", "20.0"))
        pos_weight = min(n_neg / max(n_pos, 1), pos_weight_cap)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.model.device)
        )
        sampler_positive_multiplier = float(os.environ.get("SAMPLER_POS_MULT", "5.0"))
        sampler_positive_cap = float(os.environ.get("SAMPLER_POS_CAP", "0.25"))
        sampler = make_weighted_sampler(
            labels,
            positive_multiplier=sampler_positive_multiplier,
            positive_cap=sampler_positive_cap,
            seed=10_000 + CLIENT_ID,
        )
        loader_kwargs = {
            "batch_size": batch_size,
            "sampler": sampler,
            "drop_last": False,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        self.train_loader = DataLoader(train_dataset, **loader_kwargs)
        logger.info(
            "DATA n=%s fraud=%.2f%% bs=%s workers=%s amp=%s loss=%s "
            "bce=%.2f pos_w=%.2f sampler=%.1fx/%.2f",
            f"{self.dataset_size:,}",
            labels.mean() * 100,
            batch_size,
            self.num_workers,
            int(self.use_amp),
            self.loss_mode,
            self.bce_mix,
            pos_weight,
            sampler_positive_multiplier,
            sampler_positive_cap,
        )

    # Flower interface

    def get_parameters(self, config: dict = {}) -> list[np.ndarray]:
        return federated_params(cast(FraudMLP, self.model))

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        trainable_keys = [k for k in state_dict.keys() if is_federated_param(k)]
        assert len(trainable_keys) == len(parameters), (
            f"Parameter count mismatch: {len(trainable_keys)} keys vs "
            f"{len(parameters)} parameters"
        )
        for k, v in zip(trainable_keys, parameters):
            state_dict[k].copy_(torch.as_tensor(v, device=state_dict[k].device))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr           = float(config.get("lr", self.lr))
        weight_decay = float(config.get("weight_decay", self.weight_decay))
        epochs       = int(config.get("local_epochs", self.local_epochs))
        fedprox_mu   = float(config.get("fedprox_mu", os.environ.get("FEDPROX_MU", "0.001")))
        focal_alpha  = float(config.get("focal_alpha", self.focal_loss.alpha))
        self.loss_mode = str(config.get("loss_mode", os.environ.get("LOSS_MODE", self.loss_mode))).lower()
        self.bce_mix = float(config.get("bce_mix", os.environ.get("BCE_MIX", self.bce_mix)))
        self.focal_loss.gamma = float(config.get("focal_gamma", os.environ.get("FOCAL_GAMMA", self.focal_loss.gamma)))
        self.local_scheduler = str(
            config.get(
                "local_lr_scheduler",
                os.environ.get("LOCAL_LR_SCHEDULER", self.local_scheduler),
            )
        ).lower()
        self.min_lr_ratio = float(config.get("min_lr_ratio", os.environ.get("MIN_LR_RATIO", self.min_lr_ratio)))
        self.focal_loss.alpha = min(max(focal_alpha, 0.01), 0.99)
        global_params = [
            p.detach().clone()
            for p in self.model.parameters()
            if p.requires_grad
        ] if fedprox_mu > 0 else []

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scaler = _make_grad_scaler(self.use_amp)
        scheduler = self._make_local_scheduler(optimizer, lr, epochs)

        logger.info(
            "FIT start ep=%s lr=%.1e wd=%.1e alpha=%.2f gamma=%.2f "
            "loss=%s bce=%.2f sched=%s prox=%.1e",
            epochs,
            lr,
            weight_decay,
            self.focal_loss.alpha,
            self.focal_loss.gamma,
            self.loss_mode,
            self.bce_mix,
            self.local_scheduler,
            fedprox_mu,
        )
        self.model.train()

        epoch_losses: list[float] = []
        grad_norms: list[float] = []
        for epoch in range(1, epochs + 1):
            epoch_loss  = 0.0
            batch_count = 0
            for X, y in self.train_loader:
                X = X.to(self.model.device, non_blocking=self.pin_memory)
                y = y.to(self.model.device, non_blocking=self.pin_memory)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.model.device, self.use_amp):
                    loss = self._training_loss(self.model(X).squeeze(), y)
                if fedprox_mu > 0:
                    prox = torch.zeros((), device=self.model.device)
                    for param, global_param in zip(
                        (p for p in self.model.parameters() if p.requires_grad),
                        global_params,
                    ):
                        prox = prox + torch.sum((param - global_param) ** 2)
                    loss = loss + 0.5 * fedprox_mu * prox
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_norms.append(float(grad_norm.detach().cpu()))
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                epoch_loss  += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(batch_count, 1)
            epoch_losses.append(float(avg_loss))
            logger.info(
                "FIT ep=%s/%s loss=%.6f lr=%.1e",
                epoch,
                epochs,
                avg_loss,
                optimizer.param_groups[0]["lr"],
            )

        # Quick val AUPRC/AUROC for server-side weighting
        train_loss_start = epoch_losses[0] if epoch_losses else 0.0
        train_loss_end = epoch_losses[-1] if epoch_losses else 0.0
        fit_metrics: dict[str, Scalar] = {
            "client_id": CLIENT_ID,
            "train_loss": train_loss_end,
            "train_loss_start": train_loss_start,
            "train_loss_end": train_loss_end,
            "train_loss_delta": train_loss_start - train_loss_end,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "fit_lr": lr,
            "fit_lr_final": float(optimizer.param_groups[0]["lr"]),
            "fit_local_epochs": epochs,
            "fit_bce_mix": self.bce_mix,
            "fit_focal_gamma": self.focal_loss.gamma,
        }
        logger.info(
            "FIT done train=%.6f delta=%+.6f grad=%.3f lr_final=%.1e",
            train_loss_end,
            train_loss_start - train_loss_end,
            float(np.mean(grad_norms)) if grad_norms else 0.0,
            optimizer.param_groups[0]["lr"],
        )
        val_metrics = self._quick_val_metrics()
        if val_metrics is not None:
            fit_metrics["val_auprc"] = val_metrics["auprc"]
            fit_metrics["val_auroc"] = val_metrics["auroc"]
            fit_metrics["val_f1"] = val_metrics["f1"]
            fit_metrics["val_threshold"] = val_metrics["threshold"]
            logger.info(
                "FIT val auprc=%.4f auroc=%.4f f1=%.4f thr=%.4f",
                val_metrics["auprc"],
                val_metrics["auroc"],
                val_metrics["f1"],
                val_metrics["threshold"],
            )

        return (
            self.get_parameters(),
            self.dataset_size,
            fit_metrics,
        )

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss, total_examples = 0.0, 0
        total_focal_loss = 0.0
        total_bce_loss = 0.0
        total_hybrid_loss = 0.0
        all_probs, all_targets     = [], []

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.model.device, non_blocking=self.pin_memory)
                y = y.to(self.model.device, non_blocking=self.pin_memory)
                with _autocast_context(self.model.device, self.use_amp):
                    logits = self.model(X).squeeze()
                    focal_loss = self.focal_loss(logits, y)
                    bce_loss = self.bce_loss(logits, y.float())
                    hybrid_loss = self._mix_loss(focal_loss, bce_loss)
                    loss = self._select_loss(focal_loss, bce_loss, hybrid_loss)
                probs   = torch.sigmoid(logits)
                total_loss     += loss.item() * X.shape[0]
                total_focal_loss += focal_loss.item() * X.shape[0]
                total_bce_loss += bce_loss.item() * X.shape[0]
                total_hybrid_loss += hybrid_loss.item() * X.shape[0]
                total_examples += X.shape[0]
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / max(total_examples, 1)
        avg_focal_loss = total_focal_loss / max(total_examples, 1)
        avg_bce_loss = total_bce_loss / max(total_examples, 1)
        avg_hybrid_loss = total_hybrid_loss / max(total_examples, 1)
        y_true   = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int8)
        y_prob   = np.concatenate(all_probs)   if all_probs   else np.array([], dtype=np.float32)

        if len(np.unique(y_true)) > 1:
            auprc = float(average_precision_score_np(y_true, y_prob))
            auroc = float(roc_auc_score_np(y_true, y_prob))
            prec, rec, thresholds = precision_recall_curve_np(y_true, y_prob)
            f1s    = 2 * prec * rec / (prec + rec + 1e-9)
            best_t = float(thresholds[f1s[:-1].argmax()]) if len(thresholds) else 0.5
            best_f1 = float(f1s.max())
        else:
            auprc, auroc, best_t, best_f1 = float("nan"), float("nan"), 0.5, 0.0

        metrics: dict[str, Scalar] = {
            "client_id": CLIENT_ID,
            "val_loss":      float(avg_loss),
            "val_focal_loss": float(avg_focal_loss),
            "val_bce_loss":   float(avg_bce_loss),
            "val_hybrid_loss": float(avg_hybrid_loss),
            "val_auprc":     auprc,
            "val_auroc":     auroc,
            "val_f1":        best_f1,
            "val_threshold": best_t,
        }
        logger.info(
            "VAL n=%s loss=%.6f focal=%.6f bce=%.6f auprc=%.4f "
            "auroc=%.4f f1=%.4f thr=%.4f",
            total_examples,
            avg_loss,
            avg_focal_loss,
            avg_bce_loss,
            auprc,
            auroc,
            best_f1,
            best_t,
        )
        return float(avg_loss), total_examples, metrics

    # Internal helpers

    def _make_local_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        epochs: int,
    ):
        if self.local_scheduler in {"", "none", "off", "constant"}:
            return None
        steps = max(1, epochs * len(self.train_loader))
        if self.local_scheduler == "cosine":
            eta_min = max(lr * min(max(self.min_lr_ratio, 0.0), 1.0), 1e-7)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=eta_min,
            )
        return None

    def _mix_loss(self, focal: torch.Tensor, bce: torch.Tensor) -> torch.Tensor:
        mix = min(max(float(self.bce_mix), 0.0), 1.0)
        return (1.0 - mix) * focal + mix * bce

    def _select_loss(
        self,
        focal: torch.Tensor,
        bce: torch.Tensor,
        hybrid: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_mode == "focal":
            return focal
        if self.loss_mode == "bce":
            return bce
        return hybrid

    def _training_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(logits, y)
        bce = self.bce_loss(logits, y.float())
        hybrid = self._mix_loss(focal, bce)
        return self._select_loss(focal, bce, hybrid)

    def _quick_val_metrics(self) -> dict[str, float] | None:
        """Run inference on val set, return AUPRC/AUROC if both classes are present."""
        self.model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.model.device, non_blocking=self.pin_memory)
                y = y.to(self.model.device, non_blocking=self.pin_memory)
                with _autocast_context(self.model.device, self.use_amp):
                    probs = torch.sigmoid(self.model(X).squeeze())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        self.model.train()
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)
        if len(np.unique(y_true)) < 2:
            return None
        prec, rec, thresholds = precision_recall_curve_np(y_true, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(f1s[:-1].argmax()) if len(thresholds) else 0
        return {
            "auprc": float(average_precision_score_np(y_true, y_prob)),
            "auroc": float(roc_auc_score_np(y_true, y_prob)),
            "f1": float(f1s.max()),
            "threshold": float(thresholds[best_idx]) if len(thresholds) else 0.5,
        }


__all__ = ["FraudMLP", "is_federated_param", "federated_params"]
