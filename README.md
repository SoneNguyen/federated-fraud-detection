# FL Fraud Detection

A federated learning fraud detection project configured for `uv`.

## Overview

This repo implements a local federated learning pipeline for the IEEE-CIS credit card fraud dataset.
The training flow is:

1. prepare the dataset into three federated client partitions
2. start a Flower server
3. start three federated clients, each using its own local data partition
4. train for a fixed number of rounds and save model checkpoints
5. evaluate the trained model on a held-out test split

The Jupyter notebook that was previously included has been removed: the repo now uses the CLI scripts in `server/`, `client/`, `data/`, and `model/`.

## Prerequisites

- Python 3.12
- `uv` package installed
- Raw IEEE-CIS files under `data/ieee_cis/`:
  - `train_transaction.csv`
  - `train_identity.csv`

If `uv` is not installed:

```powershell
pip install uv
```

If the raw dataset files are missing, `uv run python data/load_ieee_cis.py` will fail.

## Setup

Install the project dependencies from `pyproject.toml` and `uv.lock`:

```powershell
uv install
```

This installs all required packages, including `flwr`, `torch`, `pandas`, `numpy`, `mlflow`, and test dependencies.

## Data preparation

Prepare the IEEE-CIS dataset and create normalized per-client parquet files:

```powershell
uv run python data/load_ieee_cis.py
```

This script performs the following steps:

- loads `data/ieee_cis/train_transaction.csv` and `data/ieee_cis/train_identity.csv`
- merges transaction and identity data
- engineers the project feature set defined in `contracts/schema.json`
- splits the processed data into 3 federated clients by `ProductCD`
- computes global normalization parameters and saves them to `contracts/normalization_params.json`
- writes client files to:
  - `data/processed/client_0/transactions_normalized.parquet`
  - `data/processed/client_1/transactions_normalized.parquet`
  - `data/processed/client_2/transactions_normalized.parquet`

If this command completes successfully, the repo is ready for federated training.

## Run federated training

The federated training loop uses `server/fl_server.py` and `client/run_client.py`.

### 1. Start the server

Open a terminal and run:

```powershell
uv run python server/fl_server.py
```

The server:

- starts a Flower server on `0.0.0.0:8080`
- uses the `WeightedFedAvg` strategy in `server/strategy.py`
- loads the latest compatible checkpoint from `checkpoints/` if present
- saves new checkpoints to `checkpoints/round_XXX.pt`

### 2. Start the clients

Open three additional terminals and run one client in each.

Client 0:

```powershell
$env:CLIENT_ID = 0
$env:DATA_PATH = "data/processed/client_0/transactions_normalized.parquet"
uv run python client/run_client.py
```

Client 1:

```powershell
$env:CLIENT_ID = 1
$env:DATA_PATH = "data/processed/client_1/transactions_normalized.parquet"
uv run python client/run_client.py
```

Client 2:

```powershell
$env:CLIENT_ID = 2
$env:DATA_PATH = "data/processed/client_2/transactions_normalized.parquet"
uv run python client/run_client.py
```

Each client:

- loads a local dataset partition from `DATA_PATH`
- builds a `FraudMLP` model
- joins the Flower server at `SERVER_ADDRESS` (default `localhost:8080`)
- performs local training for `LOCAL_EPOCHS` (default `5`)

The client script uses these environment variables:

- `CLIENT_ID` (required)
- `DATA_PATH` (required)
- `SERVER_ADDRESS` (optional; default `localhost:8080`)
- `LOCAL_EPOCHS` (optional; default `5`)
- `DEVICE` (optional; auto-detects GPU if available)

### 2b. Convenience: Run all 3 clients at once

Instead of opening three terminals manually, use one of these convenience scripts to launch all clients simultaneously:

**Option A: PowerShell script (Windows)**

```powershell
.\run_all_clients.ps1
```

**Option B: Python script (cross-platform)**

```powershell
uv run python run_all_clients.py
```

Both scripts:

- verify that data is prepared and available
- start all 3 clients in parallel with proper environment setup
- monitor them until completion
- show cleanup status on Ctrl+C

### 3. Verify training output and checkpoints

During training, the server will print round progress and save checkpoints to `checkpoints/`.

Expected saved files:

- `checkpoints/round_001.pt`
- `checkpoints/round_002.pt`
- ...

These files contain the federated model state dictionary produced by the server aggregation.

### 4. Where the trained model is saved

The trained model is saved as PyTorch checkpoint files in:

- `checkpoints/`

The latest federated model is the newest `round_*.pt` file in that directory.

The evaluation script and API loader read checkpoints from `checkpoints/`.

## Evaluate the trained model

After training completes, run:

```powershell
uv run python model/evaluate.py
```

The evaluation script:

- loads the latest `round_*.pt` checkpoint
- evaluates on the held-out portion of `data/processed/client_0/transactions_normalized.parquet`
- writes `results/evaluation_report.json`
- asserts `AUPRC >= 0.75`

If the assertion fails, review the training logs and verify that all clients connected successfully.

## Optional: Model calibration

To calibrate a saved checkpoint using Platt scaling:

```powershell
uv run python model/calibrate.py --checkpoint checkpoints/round_010.pt --data data/processed/client_0/transactions_normalized.parquet
```

This will write `checkpoints/calibration_params.json`.

## Run tests

Run the full test suite:

```powershell
uv run pytest
```

Run a single test file:

```powershell
uv run pytest tests/unit/test_converter.py
```

## Troubleshooting

### Incompatible checkpoint on server startup

If the server fails with a checkpoint key mismatch (for example `KeyError: 'net.0.bias'`), the existing checkpoint is incompatible with the current model definition.

Resolve it by moving or removing old checkpoints:

```powershell
Remove-Item checkpoints\*.pt
Remove-Item checkpoints\*.json
```

Then restart the server and clients.

### Missing data files

If `data/load_ieee_cis.py` fails because raw files are missing, place the IEEE-CIS source files under `data/ieee_cis/`:

- `data/ieee_cis/train_transaction.csv`
- `data/ieee_cis/train_identity.csv`

### Visual C++ redistributable warning

On Windows, PyTorch may print a warning about the Microsoft Visual C++ Redistributable. Install the redistributable from:

https://aka.ms/vs/17/release/vc_redist.x64.exe

## Notes

- This repo is managed via `pyproject.toml` and `uv.lock`.
- Always run project commands through `uv run ...` so the correct environment is activated.
- The federated server and clients must be started in separate terminals for local end-to-end training.
