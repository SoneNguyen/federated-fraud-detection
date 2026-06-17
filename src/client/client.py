"""Federated Learning Client Module"""
from collections.abc import Sized
from typing import cast
import logging
import os

from flwr.client import NumPyClient
from flwr.common import Scalar
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.model.fraud_mlp import FraudMLP, is_federated_param, federated_params
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))

logger = logging.getLogger(f"Client{CLIENT_ID}")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f'[C{CLIENT_ID:1d}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)


class FocalLoss(torch.nn.Module):
    """
    Focal loss for class-imbalanced binary classification.

    gamma=2.0 (not 3.0) — lower gamma means less suppression of easy negatives,
    which is important in federated setting where each client sees fewer fraud
    samples per round. gamma=3.0 over-focuses and starves gradients early.

    alpha=0.5 (reduced from 0.75) — less aggressive weighting of positive class,
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
        weight = self.alpha * (1 - pt) ** self.gamma
        return (weight * bce).mean()


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Return a WeightedRandomSampler that oversamples the minority (fraud) class
    to improve gradient signal without over-distorting the training distribution.

    Strategy: oversample positives by at most 2.5× their natural rate,
    capped at 15% (less aggressive than before). This keeps the training
    distribution much closer to the real one, improving calibration.
    """
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    if n_pos == 0:
        # No fraud at all — return uniform sampler
        return WeightedRandomSampler(
            weights=np.ones(len(labels)).tolist(),
            num_samples=len(labels),
            replacement=True,
        )

    natural_rate = n_pos / len(labels)
    # Target: oversample positives by at most 2.5× their natural rate,
    # capped at 15% (down from 30%).
    target_rate = min(natural_rate * 2.5, 0.15)

    w_pos = target_rate / n_pos
    w_neg = (1.0 - target_rate) / max(n_neg, 1)
    weights = np.where(labels == 1, w_pos, w_neg).astype(np.float32).tolist()
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
    )


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
        model: torch.nn.Module,
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

        # If train_dataset is a DataLoader, extract the underlying dataset
        if isinstance(train_dataset, DataLoader):
            train_dataset = train_dataset.dataset

        # Build oversampled train loader from dataset
        labels = np.array([y for _, y in train_dataset])
        self.dataset_size = len(cast(Sized, train_dataset))
        sampler = make_weighted_sampler(labels)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            num_workers=2,
        )
        logger.info(
            f"Train loader: {self.dataset_size:,} samples | "
            f"fraud={labels.mean() * 100:.1f}% | "
            f"batch_size={batch_size} (oversampled)"
        )

    # ── Flower interface ──────────────────────────────────────────────────────

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
            state_dict[k].copy_(torch.tensor(v))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr           = float(config.get("lr", self.lr))
        weight_decay = float(config.get("weight_decay", self.weight_decay))
        epochs       = int(config.get("local_epochs", self.local_epochs))

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        logger.info(f"START TRAINING | epochs={epochs}, lr={lr:.2e}")
        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss  = 0.0
            batch_count = 0
            for X, y in self.train_loader:
                X, y = X.to(self.model.device), y.to(self.model.device)
                optimizer.zero_grad()
                loss = self.focal_loss(self.model(X).squeeze(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss  += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(batch_count, 1)
            logger.info(f"EPOCH {epoch}/{epochs} | loss={avg_loss:.6f} | lr={lr:.2e}")

        logger.info("TRAINING COMPLETE")

        # Quick val AUPRC/AUROC for server-side weighting
        fit_metrics = {}
        val_metrics = self._quick_val_metrics()
        if val_metrics is not None:
            fit_metrics["val_auprc"] = val_metrics["auprc"]
            fit_metrics["val_auroc"] = val_metrics["auroc"]
            logger.info(
                f"FIT AUPRC={val_metrics['auprc']:.4f} | AUROC={val_metrics['auroc']:.4f}"
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
        all_probs, all_targets     = [], []

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y   = X.to(self.model.device), y.to(self.model.device)
                logits  = self.model(X).squeeze()
                loss    = self.focal_loss(logits, y)
                probs   = torch.sigmoid(logits)
                total_loss     += loss.item() * X.shape[0]
                total_examples += X.shape[0]
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / max(total_examples, 1)
        y_true   = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int8)
        y_prob   = np.concatenate(all_probs)   if all_probs   else np.array([], dtype=np.float32)

        if len(np.unique(y_true)) > 1:
            auprc  = float(average_precision_score(y_true, y_prob))
            auroc  = float(roc_auc_score(y_true, y_prob))
            prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
            f1s    = 2 * prec * rec / (prec + rec + 1e-9)
            best_t = float(thresholds[f1s[:-1].argmax()]) if len(thresholds) else 0.5
            best_f1 = float(f1s.max())
        else:
            auprc, auroc, best_t, best_f1 = float("nan"), float("nan"), 0.5, 0.0

        metrics: dict[str, Scalar] = {
            "val_loss":      float(avg_loss),
            "val_auprc":     auprc,
            "val_auroc":     auroc,
            "val_f1":        best_f1,
            "val_threshold": best_t,
        }
        logger.info(
            f"VALIDATION | loss={avg_loss:.6f} AUPRC={auprc:.4f} "
            f"AUROC={auroc:.4f} F1={best_f1:.4f} samples={total_examples}"
        )
        return float(avg_loss), total_examples, metrics

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _quick_val_metrics(self) -> dict[str, float] | None:
        """Run inference on val set, return AUPRC/AUROC if both classes are present."""
        self.model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.model.device), y.to(self.model.device)
                all_probs.append(torch.sigmoid(self.model(X).squeeze()).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        self.model.train()
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)
        if len(np.unique(y_true)) < 2:
            return None
        return {
            "auprc": float(average_precision_score(y_true, y_prob)),
            "auroc": float(roc_auc_score(y_true, y_prob)),
        }


__all__ = ["FraudMLP", "is_federated_param", "federated_params"]
