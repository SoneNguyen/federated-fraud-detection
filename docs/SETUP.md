# Project Setup & Installation Guide

## ✅ Prerequisites

- **Python 3.12+** installed on your system
- **uv** package manager (lightweight Python package manager)
- **IEEE-CIS Kaggle Fraud Detection Dataset**:
  - `train_transaction.csv` (~500MB)
  - `train_identity.csv` (~130MB)

### Install uv (one-time)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (all platforms)
pip install uv
```

Verify installation:
```bash
uv --version  # Should show uv 0.x.x
```

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd FL_Fraud_Detection
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# This will:
# - Read pyproject.toml
# - Create/activate a virtual environment
# - Install all dependencies listed in [project.dependencies]
# - Generate uv.lock for reproducibility
```

### 3. Download IEEE-CIS Dataset

The IEEE-CIS Fraud Detection dataset is available on Kaggle:
1. Visit: https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place them in:
   ```
   FL_Fraud_Detection/
   └── data/
       └── ieee_cis/
           ├── train_transaction.csv
           └── train_identity.csv
   ```

### 4. Prepare Federated Data

Run the data pipeline to split the IEEE-CIS dataset into 3 federated clients:

```bash
uv run python data/load_ieee_cis.py
```

This will:
- Load raw IEEE-CIS transaction & identity data
- Engineer 37 numeric features (log transforms, one-hot encoding, normalization)
- Temporally split into 3 clients (10th, 50th, 90th percentiles)
- Compute global normalization stats
- Save normalized Parquet files per client:

```
data/
└── processed/
    ├── client_0/
    │   └── transactions_normalized.parquet  (← 2% fraud rate)
    ├── client_1/
    │   └── transactions_normalized.parquet  (← 4% fraud rate)
    └── client_2/
        └── transactions_normalized.parquet  (← 6% fraud rate)
```

**Expected runtime**: 5-10 minutes depending on system

---

## 🚀 Running the System

### Option 1: Manual Start (3 Terminals)

**Terminal 1 - Server**:
```bash
cd FL_Fraud_Detection
uv run python scripts/run_server.py
```

**Terminal 2 - Client 0**:
```bash
cd FL_Fraud_Detection
export CLIENT_ID=0
export DATA_PATH=data/processed/client_0/transactions_normalized.parquet
uv run python scripts/run_client.py
```

**Terminal 3 - Client 1**:
```bash
cd FL_Fraud_Detection
export CLIENT_ID=1
export DATA_PATH=data/processed/client_1/transactions_normalized.parquet
uv run python scripts/run_client.py
```

**Terminal 4 - Client 2**:
```bash
cd FL_Fraud_Detection
export CLIENT_ID=2
export DATA_PATH=data/processed/client_2/transactions_normalized.parquet
uv run python scripts/run_client.py
```

### Option 2: Batch Start (Linux/macOS)

```bash
# Requires tmux
./run_all_clients.sh
```

### Option 3: Batch Start (Windows PowerShell)

```powershell
# Automatic: starts server + 3 clients in separate windows
uv run python run_all_clients.py
```

---

## 📊 Monitoring Progress

### Option A: Console Logs

Watch the server terminal — it shows:
```
[Round 1] Client 0: focal_alpha=0.75 (history=[])
[Round 1] Client 1: focal_alpha=0.75 (history=[])
[Round 1] Client 2: focal_alpha=0.75 (history=[])
[Round 1] Aggregation complete
[Round 1] Checkpoint saved → round_001.pt
[Round 1] val_loss=0.142 | val_auprc=0.412 | val_auroc=0.823
```

### Option B: MLflow Dashboard

In a new terminal:
```bash
cd FL_Fraud_Detection
mlflow ui --port 5000
```

Then open http://localhost:5000 in your browser.

### Option C: Results File

Real-time JSON log:
```bash
cat outputs/results/evaluation_history.json
```

---

## 🔧 Configuration

### Environment Variables

Set before running clients/server:

```bash
# CLIENT SETTINGS
export CLIENT_ID=0                                      # 0, 1, or 2
export DATA_PATH=data/processed/client_0/...parquet   # Full path
export SERVER_ADDRESS=localhost:8080                   # Server address
export LOCAL_EPOCHS=2                                  # Epochs per round
export DEVICE=cuda                                     # cuda, cpu, or mps

# SERVER SETTINGS
export NUM_ROUNDS=80                                   # Total training rounds
export SERVER_ADDRESS=localhost:8080                   # Bind address
```

### Checkpoint Paths

Models are saved to:
```
outputs/
└── checkpoints/
    ├── round_001.pt      ← Round 1
    ├── round_002.pt      ← Round 2
    ...
    ├── round_023.pt      ← Round 23 (best)
    ├── best_round_023.pt ← Tagged best
    └── results/
        └── evaluation_history.json
```

To resume from a checkpoint:
```python
from src.server.checkpoint_manager import CheckpointManager
ckpt = CheckpointManager("outputs/checkpoints")
model_path = ckpt.latest()  # Loads most recent
```

---

## 🐛 Troubleshooting

### Error: "Address already in use"
```bash
# Change the port
export SERVER_ADDRESS=localhost:8081
uv run python scripts/run_server.py
```

### Error: "No module named 'torch'"
```bash
# Reinstall dependencies
uv sync --force-reinstall
```

### Error: "data/processed/client_0/...parquet not found"
```bash
# Run data preparation
uv run python data/load_ieee_cis.py

# If that fails, check:
ls -la data/ieee_cis/       # Raw files present?
ls -la data/processed/      # Processed output exists?
```

### Error: Type mismatch on WeightedRandomSampler (Windows)
```bash
# Clear Python cache
Remove-Item -Path "$PWD\**\__pycache__" -Recurse -Force

# Reinstall
uv sync --force-reinstall
```

### GPU Not Detected
```bash
# Check CUDA setup
uv run python -c "import torch; print(torch.cuda.is_available())"

# If False, ensure NVIDIA drivers installed:
nvidia-smi

# Or run on CPU instead:
export DEVICE=cpu
```

### Out of Memory (OOM)
```bash
# Reduce batch size in src/client/client.py
# Change: FraudDataset(..., batch_size=512) → batch_size=256

# Or run on CPU:
export DEVICE=cpu
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] `uv --version` works
- [ ] `data/ieee_cis/` contains raw CSVs
- [ ] `uv run python data/load_ieee_cis.py` completes without error
- [ ] `data/processed/client_{0,1,2}/` contain `.parquet` files
- [ ] `uv run python scripts/run_server.py` starts without error
- [ ] `uv run python scripts/run_client.py` connects to server
- [ ] `outputs/checkpoints/round_001.pt` exists after first round
- [ ] `outputs/results/evaluation_history.json` contains metrics

---

## 🎯 Next Steps

1. **Run Training**: Follow the "Running the System" section above
2. **Monitor Progress**: Use MLflow UI or console logs
3. **Analyze Results**: See [docs/API.md](API.md) for result extraction
4. **Customize**: Modify `config/schema.json` or hyperparameters as needed

---

## 📚 Additional Resources

- **Flower Documentation**: https://flower.ai/docs/
- **PyTorch**: https://pytorch.org/docs/
- **IEEE-CIS Dataset**: https://www.kaggle.com/competitions/ieee-fraud-detection/
- **Federated Learning Overview**: https://arxiv.org/abs/1602.05629

---

## 🆘 Getting Help

1. Check the [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
2. Review logs in `outputs/results/`
3. Inspect checkpoints using:
   ```bash
   uv run python -c "import torch; print(torch.load('outputs/checkpoints/round_001.pt').keys())"
   ```
4. Open a GitHub issue with:
   - Error message & traceback
   - Your system info (`uv --version`, `python --version`, `nvidia-smi`)
   - Steps to reproduce

---

**Enjoy federated learning! 🚀**
