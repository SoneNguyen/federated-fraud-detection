from collections.abc import Sized
from typing import cast
import logging
import os

from flwr.client import NumPyClient
import numpy as np
import torch
import torch.nn.functional as F

# ─ Per-client logging with easy comparison ─────────────────────────────────
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))

# Create logger with client prefix
logger = logging.getLogger(f"Client{CLIENT_ID}")
logger.setLevel(logging.INFO)

# Console handler with structured format
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'[C{CLIENT_ID:1d}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class FocalLoss(torch.nn.Module):
    """Focal loss for handling class imbalance.
    
    Focuses on hard examples by down-weighting easy ones.
    L = -alpha * (1-pt)^gamma * log(pt)
    where pt is model confidence in true class.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()


class FraudClient(NumPyClient):
    """Federated learning client for fraud detection with focal loss."""

    def __init__(self, model: torch.nn.Module, train_loader, val_loader, local_epochs: int = 5, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0)

    def get_parameters(self, config: dict = {}) -> list[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k].copy_(torch.tensor(v))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        lr = float(config.get("lr", self.lr))
        weight_decay = float(config.get("weight_decay", self.weight_decay))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        epochs = int(config.get("local_epochs", self.local_epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)
        logger.info(f"START TRAINING | epochs={epochs}, lr={lr:.6f}, weight_decay={weight_decay:.6f}, samples={len(cast(Sized, self.train_loader.dataset))}")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.model.device), y.to(self.model.device)
                optimizer.zero_grad()
                pred = self.model(X).squeeze()
                loss = self.focal_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            scheduler.step()
            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            logger.info(f"EPOCH {epoch}/{epochs} | loss={avg_epoch_loss:.6f} | lr={scheduler.get_last_lr()[0]:.6f}")

        logger.info("TRAINING COMPLETE")
        return self.get_parameters(), len(cast(Sized, self.train_loader.dataset)), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_examples = 0.0, 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.model.device), y.to(self.model.device)
                pred = self.model(X).squeeze()
                loss = self.focal_loss(pred, y)
                total_loss += loss.item() * X.shape[0]
                total_examples += X.shape[0]

        avg_loss = total_loss / max(total_examples, 1)
        logger.info(f"VALIDATION | loss={avg_loss:.6f}, samples={total_examples}")
        return float(avg_loss), total_examples, {"val_loss": avg_loss}


__all__ = ["FraudClient"]