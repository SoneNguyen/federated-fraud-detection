import os
from collections.abc import Sized
from typing import Any, cast
from opacus import PrivacyEngine
from client.fl_client import FraudClient
import torch


class FraudClientDP(FraudClient):
    """FL client with optional Differential Privacy via Opacus.

    Opacus wraps the model, optimizer, and dataloader together.
    The optimizer must be created here (not inside fit()) so Opacus
    can attach its gradient hooks before any training begins.
    """

    FRAUD_WEIGHT = 80.0

    def __init__(self, model, train_loader, val_loader,
                 max_grad_norm: float = 1.0,
                 noise_mult: float = 0.8):
        super().__init__(model, train_loader, val_loader)

        if os.environ.get("USE_DP", "false").lower() == "true":
            # Create optimizer here so Opacus can wrap it
            self._dp_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-3, weight_decay=1e-5
            )
            engine = PrivacyEngine()
            private_result = cast(
                tuple[torch.nn.Module, torch.optim.Optimizer, Any],
                engine.make_private(
                    module=self.model,
                    optimizer=self._dp_optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=noise_mult,
                    max_grad_norm=max_grad_norm,
                ),
            )

            self.model = cast(torch.nn.Module, private_result[0])
            self._dp_optimizer = cast(torch.optim.Optimizer, private_result[1])
            if len(private_result) == 3:
                self.train_loader = cast(torch.utils.data.DataLoader, private_result[2])
            else:
                self.train_loader = cast(torch.utils.data.DataLoader, private_result[3])

            self._dp_engine = engine
            print(f"[DP] Privacy engine active: noise={noise_mult}, clip={max_grad_norm}")
        else:
            self._dp_engine = None
            self._dp_optimizer = None
            print("[DP] Differential privacy disabled")

    def fit(self, parameters, config):
        """Override fit() to use the pre-built DP optimizer instead of
        creating a new one inside the loop (which would break Opacus hooks)."""
        self.set_parameters(parameters)
        self.model.train()

        if self._dp_optimizer is not None:
            optimizer = self._dp_optimizer
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(config.get("lr", 1e-3)),
                weight_decay=1e-5,
            )

        epochs = int(config.get("local_epochs", 2))
        import torch.nn.functional as F

        for _ in range(epochs):
            for X, y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(X).squeeze()
                bce = F.binary_cross_entropy(pred, y, reduction="none")
                weights = torch.where(
                    y == 1,
                    torch.full_like(y, self.FRAUD_WEIGHT),
                    torch.ones_like(y),
                )
                loss = (bce * weights).mean()
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(cast(Sized, self.train_loader.dataset)), {}

    def get_epsilon(self, delta: float = 1e-5) -> float:
        if self._dp_engine is not None:
            return self._dp_engine.get_epsilon(delta)
        return float("inf")