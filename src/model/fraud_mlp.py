# FraudMLP — residual MLP for federated fraud detection.
# Input dimension is read from schema.json at import time.
# Architecture: 4 residual blocks with skip connections so gradients
# reach early layers even when the minority class is rare per-batch.
import json
from typing import Optional
from pathlib import Path
import torch
import torch.nn as nn

_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

# Load schema from config directory
config_dir = Path(__file__).parent.parent.parent / "config"
with open(config_dir / "schema.json") as f:
    INPUT_DIM = json.load(f)["feature_schema"]["total_features"]


class _ResBlock(nn.Module):
    """
    Two-layer residual block with BN → ReLU → Linear ordering (pre-activation).
    Pre-activation keeps gradients cleaner and works better with focal loss
    on imbalanced data than post-activation.
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)   # skip connection


class FraudMLP(nn.Module):
    """
    Residual MLP for fraud detection.

    Architecture:
        Input (INPUT_DIM) → project to 128
        ResBlock(128, dropout=0.20)
        ResBlock(128, dropout=0.15)
        ResBlock(128, dropout=0.10)
        BN → ReLU → Linear(128→64) → BN → ReLU → Linear(64→1)

    Kept narrower than the old 256-wide stack intentionally:
    - 32–40 features don't need 256 units; excess width encourages memorizing
      the majority class.
    - Residual connections compensate for reduced depth giving worse gradients.
    """
    def __init__(self, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.input_proj = nn.Linear(INPUT_DIM, 128)

        self.res_blocks = nn.Sequential(
            _ResBlock(128, dropout=0.20),
            _ResBlock(128, dropout=0.15),
            _ResBlock(128, dropout=0.10),
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(128, momentum=0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()
        self.to(self.device)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He init — correct for ReLU activations
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.head(x)

# ── Federation helpers ────────────────────────────────────────────────────────

_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def is_federated_param(key: str) -> bool:
    """Return True if this state_dict key should be included in federation.

    BN running stats are client-local — averaging them across clients with
    heterogeneous fraud rates produces stats that match nobody. Each client
    rebuilds its own after receiving fresh weights (momentum=0.1, ~20 batches).
    """
    return not any(s in key for s in _BN_BUFFER_SUFFIXES)


def federated_params(model: "FraudMLP") -> list:
    """Return the model's federated parameter arrays (excludes BN buffers)."""
    return [
        v.cpu().numpy()
        for k, v in model.state_dict().items()
        if is_federated_param(k)
    ]

# Smoke test
if __name__ == "__main__":
    m = FraudMLP()
    x = torch.randn(8, INPUT_DIM).to(m.device)
    out = m(x)
    assert out.shape == (8, 1), f"Bad shape: {out.shape}"
    print(f"FraudMLP OK: device={m.device}, input_dim={INPUT_DIM}, output={out.shape}")
