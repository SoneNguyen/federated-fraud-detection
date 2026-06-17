# FraudMLP - residual MLP for federated fraud detection.
# The input dimension is read from src.data.feature_registry at import time.
from typing import Optional

import torch
import torch.nn as nn

from src.data.feature_registry import FEATURE_ORDER

_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

INPUT_DIM = len(FEATURE_ORDER)


class _ResBlock(nn.Module):
    """Two-layer residual block with LayerNorm/ReLU/Linear ordering."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FraudMLP(nn.Module):
    """
    Residual MLP for fraud detection.

    Architecture:
        Input (INPUT_DIM) -> Linear(256)
        ResBlock(256, dropout=0.15)
        ResBlock(256, dropout=0.10)
        ResBlock(256, dropout=0.08)
        LayerNorm/ReLU/Linear(256->128) -> LayerNorm/ReLU/Linear(128->1)
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        hidden_dim = 256
        self.input_proj = nn.Linear(INPUT_DIM, hidden_dim)

        self.res_blocks = nn.Sequential(
            _ResBlock(hidden_dim, dropout=0.15),
            _ResBlock(hidden_dim, dropout=0.10),
            _ResBlock(hidden_dim, dropout=0.08),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()
        self.to(self.device)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.head(x)


def is_federated_param(key: str) -> bool:
    """Return True if this state_dict key should be included in federation.

    BatchNorm running stats are excluded if a future architecture reintroduces
    them, because averaging client-local statistics across heterogeneous fraud
    rates produces stats that match nobody.
    """
    return not any(s in key for s in _BN_BUFFER_SUFFIXES)


def federated_params(model: "FraudMLP") -> list:
    """Return the model's federated parameter arrays."""
    return [
        v.cpu().numpy()
        for k, v in model.state_dict().items()
        if is_federated_param(k)
    ]


if __name__ == "__main__":
    m = FraudMLP()
    x = torch.randn(8, INPUT_DIM).to(m.device)
    out = m(x)
    assert out.shape == (8, 1), f"Bad shape: {out.shape}"
    print(f"FraudMLP OK: device={m.device}, input_dim={INPUT_DIM}, output={out.shape}")
