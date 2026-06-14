# Federated Learning Fraud Detection System

**A distributed fraud detection system using Federated Learning with PyTorch and Flower (Flwr)**

---

## 📋 Project Overview

This is a production-ready **Federated Learning (FL)** system for credit card fraud detection across distributed clients. The architecture implements:

- **3 Federated Clients**: Each client holds local transaction data without sharing raw records
- **1 Flower Server**: Orchestrates training rounds and aggregates model updates
- **IEEE-CIS Fraud Dataset**: Pre-split into temporal thirds for federated clients
- **Weighted Averaging**: Per-client focal loss adaptation based on historical performance
- **Checkpoint Management**: Automatic model saving and rollback support
- **Imbalanced Learning**: Oversampling strategy (5× natural rate) for sparse fraud samples

---

## 🏗️ Project Structure

```
.
├── src/                         # Source code
│   ├── model/                   # Model architecture
│   │   ├── fraud_mlp.py         # Residual MLP with 37 input features
│   │   └── __init__.py
│   ├── client/                  # Federated client logic
│   │   ├── client.py            # FraudClient, FocalLoss, sampler
│   │   └── __init__.py
│   ├── server/                  # Federated server logic
│   │   ├── strategy.py          # WeightedFedAvg strategy
│   │   ├── checkpoint_manager.py# Checkpoint I/O
│   │   ├── client_state.py      # Per-client AUPRC tracking
│   │   └── __init__.py
│   └── data/                    # Data loading & preprocessing
│       ├── dataset.py           # FraudDataset, make_loaders()
│       └── __init__.py
│
├── scripts/                     # Entry points
│   ├── run_server.py            # Launch Flower server
│   └── run_client.py            # Launch Flower client
│
├── config/                      # Configuration & contracts
│   ├── schema.json              # Feature schema & ordering
│   ├── drift_config.json        # Drift detection thresholds
│   └── normalization_params.json# Federated normalization stats
│
├── outputs/                     # Training artifacts
│   ├── checkpoints/             # Model checkpoints per round
│   └── results/                 # Evaluation metrics history
│
├── data/                        # Dataset storage
│   ├── ieee_cis/                # Raw IEEE-CIS CSVs (user-provided)
│   └── processed/               # Preprocessed Parquet per client
│       ├── client_0/
│       ├── client_1/
│       └── client_2/
│
├── tests/                       # Test suites
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── demo/                    # Demo & smoke tests
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # System design
│   ├── SETUP.md                 # Installation & setup
│   └── API.md                   # API reference
│
├── pyproject.toml               # Project metadata & dependencies
├── README.md                    # This file
└── uv.lock                      # Lockfile (managed by uv)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **uv** package manager (install via `pip install uv`)
- **Raw IEEE-CIS files** under `data/ieee_cis/`:
  - `train_transaction.csv`
  - `train_identity.csv`

### Setup

1. **Clone the repository**
   ```bash
   cd FL_Fraud_Detection
   ```

2. **Install dependencies** (via uv):
   ```bash
   uv sync
   ```

3. **Prepare the dataset**:
   ```bash
   # Place raw IEEE-CIS files in data/ieee_cis/, then:
   uv run python data/load_ieee_cis.py
   ```
   This generates normalized Parquet files in `data/processed/client_{0,1,2}/`.

### Run the System

**Terminal 1: Start the server**
```bash
uv run python scripts/run_server.py
```

**Terminal 2-4: Start three clients** (in parallel)
```bash
# Client 0
CLIENT_ID=0 DATA_PATH=data/processed/client_0/transactions_normalized.parquet uv run python scripts/run_client.py

# Client 1
CLIENT_ID=1 DATA_PATH=data/processed/client_1/transactions_normalized.parquet uv run python scripts/run_client.py

# Client 2
CLIENT_ID=2 DATA_PATH=data/processed/client_2/transactions_normalized.parquet uv run python scripts/run_client.py
```

Or use the batch scripts:
```bash
# PowerShell (Windows)
uv run python run_all_clients.py

# POSIX shell (Linux/Mac)
./run_all_clients.ps1
```

---

## 🎯 Key Features

### 1. **Federated Architecture**
- **No raw data centralization**: Each client trains on local data only
- **Privacy-preserving**: Only model updates (not data) are transmitted
- **3 heterogeneous clients**: Different fraud rates (2%, 4%, 6%)

### 2. **Advanced Loss & Sampling**
- **Focal Loss** (γ=2.0, α=0.75): Handles imbalanced fraud detection
- **Weighted Sampler**: 5× natural rate oversampling (capped at 30%)
- **Per-client adaptation**: focal_alpha adjusted by AUPRC history

### 3. **Batch Normalization Optimization**
- **BN stats excluded from federation**: Kept client-local to prevent distribution mismatch
- **Automatic warmup**: BN stats rebuild naturally from local data
- **Solves threshold miscalibration** on low-fraud clients

### 4. **Robust Aggregation**
- **AUPRC-weighted averaging**: Clients weighted by recent performance
- **Patience-based early stopping**: Stops after 10 rounds with no improvement
- **Automatic checkpoint tagging**: Best models saved separately

### 5. **Monitoring & Logging**
- **MLflow integration**: Automatic metric tracking per round
- **Per-client AUPRC history**: 5-round rolling window
- **Structured logs**: All events timestamped and color-coded

---

## 📊 Performance Metrics

The system optimizes for **AUPRC** (Area Under Precision-Recall Curve):

| Client | Fraud Rate | Fit AUPRC | Val AUPRC | AUROC | F1-Score |
|--------|-----------|----------|----------|-------|----------|
| C0     | 2%        | 0.616    | 0.513    | 0.899 | 0.45     |
| C1     | 4%        | 0.721    | 0.690    | 0.928 | 0.62     |
| C2     | 6%        | 0.734    | 0.708    | 0.939 | 0.63     |
| Avg    | 4%        | 0.690    | 0.637    | 0.922 | 0.57     |

---

## 🔧 Configuration

### Environment Variables

```bash
# Server
NUM_ROUNDS=80                        # Number of training rounds
SERVER_ADDRESS=localhost:8080        # Server bind address

# Client
CLIENT_ID=0                          # Client ID (0, 1, or 2)
DATA_PATH=data/processed/...parquet  # Path to client's local data
SERVER_ADDRESS=localhost:8080        # Server address to connect to
LOCAL_EPOCHS=2                       # Local training epochs per round
DEVICE=cuda                          # Torch device (auto-detect if unset)
```

### Learning Rate Schedule

```
Rounds   LR      Epochs   Rationale
1-5      2e-3    5        Warm-up: aggressive learning
5-20     1e-3    5        Main training: fast convergence
20-35    5e-4    5        Fine-tuning: reduce step size
35-50    1e-4    5        Plateau break: smaller steps
50-70    5e-5    8        Slow burn: longer local training
70+      2e-5    8        Final polish: very conservative
```

---

## 📖 Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design, data flow, training loop
- [SETUP.md](docs/SETUP.md) — Detailed setup and troubleshooting
- [API.md](docs/API.md) — Class/function references

---

## 🧪 Testing

Run tests:
```bash
uv run pytest tests/
```

Run specific test suite:
```bash
# Unit tests
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Demo / smoke tests
uv run pytest tests/demo/ -v
```

---

## 🐛 Troubleshooting

### Port already in use
```bash
# Change server address
SERVER_ADDRESS=localhost:8081 uv run python scripts/run_server.py
```

### Missing data files
```bash
# Re-run data preparation
uv run python data/load_ieee_cis.py
```

### Type errors on Windows
```bash
# Clear Python cache and reinstall
Remove-Item -Path "$PWD\**\__pycache__" -Recurse -Force
uv sync --force-reinstall
```

---

## 📝 Key Papers & References

- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **IEEE-CIS Dataset**: Kaggle Fraud Detection Competition

---

## 📜 License

[Specify your license here, e.g., MIT, Apache 2.0]

---

## 👥 Contributors

- Lead Developer
- Research Team

---

## 📧 Support

For questions or issues, please open a GitHub issue or contact the team.

---

**Last Updated**: June 2026 | **Version**: 1.0
