from flwr.client import NumPyClient
import numpy as np
import torch


class FraudClient(NumPyClient):
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
            self.model.parameters(), lr=float(config.get("lr", 1e-3))
        )
        loss_fn = torch.nn.BCELoss()

        for X, y in self.train_loader:
            optimizer.zero_grad()
            loss = loss_fn(self.model(X).squeeze(), y)
            loss.backward()
            optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_fn = torch.nn.BCELoss()
        total_loss, total_examples = 0.0, 0

        with torch.no_grad():
            for X, y in self.val_loader:
                loss = loss_fn(self.model(X).squeeze(), y)
                total_loss += loss.item() * X.shape[0]
                total_examples += X.shape[0]

        return float(total_loss / total_examples), len(self.val_loader.dataset), {}


__all__ = ["FraudClient"]