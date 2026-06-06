from flwr.client import NumPyClient
import numpy as np
import torch
import torch.nn.functional as F


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

        for _ in range(epochs):
            for X, y in self.train_loader:
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

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_examples = 0.0, 0

        with torch.no_grad():
            for X, y in self.val_loader:
                pred = self.model(X).squeeze()
                loss = F.binary_cross_entropy(pred, y, reduction="sum")
                total_loss += loss.item()
                total_examples += X.shape[0]

        avg_loss = total_loss / max(total_examples, 1)
        return float(avg_loss), total_examples, {"val_loss": avg_loss}


__all__ = ["FraudClient"]