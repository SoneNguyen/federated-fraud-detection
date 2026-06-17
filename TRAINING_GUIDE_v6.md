# Federated Training Guide - v6.0 with 47 Features

## Quick Start

This guide shows how to train the fraud detection model with federated learning using 3 temporal clients.

### Prerequisites
- `uv` package manager installed
- Python 3.12
- All dependencies installed: `uv sync`
- Data prepared: `uv run python data/load_ieee_cis.py` (done)

### Training Setup

You need **4 terminals** to run the training:
- **Terminal 1**: Flower Server (central orchestrator)
- **Terminals 2-4**: 3 Federated Clients (parallel local training)

### Step 1: Start the Server (Terminal 1)

```powershell
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_server
```

Wait for the server to start and display "Waiting for clients..."

### Step 2: Start Client 0 (Terminal 2)

```powershell
$env:CLIENT_ID=0
$env:DATA_PATH="data\processed\client_0\transactions_normalized.parquet"
$env:LOCAL_EPOCHS=2
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

### Step 3: Start Client 1 (Terminal 3)

```powershell
$env:CLIENT_ID=1
$env:DATA_PATH="data\processed\client_1\transactions_normalized.parquet"
$env:LOCAL_EPOCHS=2
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

### Step 4: Start Client 2 (Terminal 4)

```powershell
$env:CLIENT_ID=2
$env:DATA_PATH="data\processed\client_2\transactions_normalized.parquet"
$env:LOCAL_EPOCHS=2
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

## Monitoring Training

- **Server Terminal**: Shows aggregation progress, AUPRC per round, early stopping status
- **Client Terminals**: Show local training loss, batch processing
- **MLflow**: Run `mlflow ui` in a 5th terminal to see detailed metrics at http://localhost:5000

## Key Configuration

- **Total Rounds**: 100 (with early stopping if no improvement for 10 rounds)
- **Local Epochs per Client**: 2 (can increase for slower convergence)
- **Batch Size**: 256 (auto-computed from dataset)
- **Loss Function**: Focal Loss (alpha=0.5, gamma=2.0)
- **Imbalance Handling**: Weighted Random Sampler (2.5x oversample, 15% cap)
- **Aggregation**: AUPRC-weighted Federated Averaging

## Expected Performance

**Previous Run (with 49 features + M4_flag)**:
- AUPRC: ~0.53
- AUROC: ~0.82

**Current Run (with 47 features, improved data quality)**:
- Target: AUPRC >= 0.70, F1 >= 0.70
- Baseline (LightGBM): AUPRC 0.25 (LightGBM worse than federated)

## Outputs

After training completes:
- **Best Model**: `checkpoints/best_model.pt`
- **Metrics**: `results/evaluation_history.json`
- **MLflow Logs**: `mlruns/` directory
- **Results Summary**: Printed to server terminal

## Troubleshooting

### Clients can't connect to server
- Ensure Terminal 1 server is fully started (shows "Waiting for clients")
- Check firewall isn't blocking localhost:8080
- Try `$env:SERVER_ADDRESS="127.0.0.1:8080"` if needed

### Out of memory
- Reduce batch size: lower LOCAL_EPOCHS to 1
- Use GPU: `$env:DEVICE="cuda"`

### Data not found
- Verify paths: `data\processed\client_0\transactions_normalized.parquet` exists
- Run data loader first: `uv run python data/load_ieee_cis.py`

## Feature Engineering (v6.0)

**Removed**:
- M4_flag (constant feature, no variance)
- M6_flag (constant feature, no variance)

**Added (47 total features)**:
- Temporal interactions: amount_x_velocity, amount_per_tx_1h/24h, spending_velocity_1h
- Temporal risk: risky_hour_flag, early_morning_high_value, weekend_high_value
- Identity/Email: both_emails_free, email_mismatch_high_value
- Device/Card: has_device_info, card_device_mismatch, new_account_high_value

All features normalized (zero mean, unit variance) across federated clients.
