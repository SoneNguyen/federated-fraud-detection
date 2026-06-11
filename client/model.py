# This module defines the FraudMLP model architecture for fraud detection. 
# It loads the input dimension from a JSON schema and constructs a multi-layer perceptron with batch normalization, dropout, and ReLU activations.
#  The model outputs a single probability value indicating the likelihood of fraud. A smoke test is included to verify the model's output shape.
import json
from typing import Optional
import torch
import torch.nn as nn

with open("contracts/schema.json") as f:
    INPUT_DIM = json.load(f)["feature_schema"]["total_features"]  # 13 (model input features)


class FraudMLP(nn.Module):
    def __init__(self, device: Optional[str] = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Architecture for 13 input features:
        # - Input: 13 features
        # - Hidden1: 256
        # - Hidden2: 128
        # - Hidden3: 64
        # - Output: 1 (fraud probability)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 256), nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(), nn.Dropout(0.25),

            nn.Linear(256, 128), nn.BatchNorm1d(128, momentum=0.01),
            nn.ReLU(), nn.Dropout(0.2),

            nn.Linear(128, 64), nn.BatchNorm1d(64, momentum=0.01),
            nn.ReLU(), nn.Dropout(0.1),

            nn.Linear(64, 1)
        ).to(self.device)
    
    def forward(self, x):
        return self.net(x)


# Smoke test
if __name__ == "__main__":
    m = FraudMLP()
    x = torch.randn(8, INPUT_DIM).to(m.device)
    out = m(x)
    assert out.shape == (8, 1), f"Bad shape: {out.shape}"
    print(f"FraudMLP OK: device={m.device}, input_dim={INPUT_DIM}, output shape={out.shape}")
