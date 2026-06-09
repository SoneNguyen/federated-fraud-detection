from collections.abc import Sized
from typing import cast
import logging

from flwr.client import NumPyClient
import numpy as np
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class FraudClient(NumPyClient):
    # Fraud rate is ~1-3.5% across clients — weight positive class heavily
    # so the model doesn't collapse to always predicting "not fraud"
    FRAUD_WEIGHT = 80.0

    def __init__(self, model: torch.nn.Module, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

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
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=1e-5,
        )
        epochs = int(config.get("local_epochs", 5))
        logger.info(f"Starting local training: {epochs} epochs, LR={optimizer.param_groups[0]['lr']:.6f}")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.model.device), y.to(self.model.device)
                optimizer.zero_grad()
                pred = self.model(X).squeeze()
                # Per-sample weighted BCE — critical for class imbalance
                # fraud samples get 80x more gradient signal than legit ones
                bce = F.binary_cross_entropy(pred, y, reduction="none")
                weights = torch.where(
                    y == 1,
                    torch.full_like(y, self.FRAUD_WEIGHT),
                    torch.ones_like(y),
                )
                loss = (bce * weights).mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                    logger.debug(f"  Epoch {epoch}/{epochs}, Batch {batch_idx + 1}/{len(self.train_loader)}: loss={loss.item():.6f}")
            
            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            logger.info(f"Epoch {epoch}/{epochs} complete: avg_loss={avg_epoch_loss:.6f}")

        logger.info("Local training finished")
        return self.get_parameters(), len(cast(Sized, self.train_loader.dataset)), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_examples = 0.0, 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.model.device), y.to(self.model.device)
                pred = self.model(X).squeeze()
                loss = F.binary_cross_entropy(pred, y, reduction="sum")
                total_loss += loss.item()
                total_examples += X.shape[0]

        avg_loss = total_loss / max(total_examples, 1)
        logger.info(f"Validation: avg_loss={avg_loss:.6f}, n_samples={total_examples}")
        return float(avg_loss), total_examples, {"val_loss": avg_loss}


__all__ = ["FraudClient"]