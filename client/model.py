# This module defines the FraudMLP model architecture for fraud detection. 
# It loads the input dimension from a JSON schema and constructs a multi-layer perceptron with batch normalization, dropout, and ReLU activations.
#  The model outputs a single probability value indicating the likelihood of fraud. A smoke test is included to verify the model's output shape.
import json, torch, torch.nn as nn

with open("contracts/schema.json") as f:
    INPUT_DIM = json.load(f)["feature_schema"]["total_features"]  # 11

class FraudMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# Smoke test
if __name__ == "__main__":
    m = FraudMLP()
    x = torch.randn(8, INPUT_DIM)
    out = m(x)
    assert out.shape == (8,1), f"Bad shape: {out.shape}"
    print(f"FraudMLP OK: input_dim={INPUT_DIM}, output shape={out.shape}")