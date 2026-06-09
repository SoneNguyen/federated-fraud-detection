# FL Fraud Detection

A federated learning fraud detection project configured for `uv`.

## Prerequisites

- Python 3.12
- `uv` installed in the environment

If `uv` is not installed, install it first:

```powershell
pip install uv
```

## Setup

Install the project dependencies from `pyproject.toml` and `uv.lock`:

```powershell
uv install
```

## Data preparation

Prepare the IEEE-CIS dataset and create normalized per-client parquet files:

```powershell
uv run python data/load_ieee_cis.py
```

This reads raw data from `data/ieee_cis/` and writes processed files to:

- `data/processed/client_0/transactions_normalized.parquet`
- `data/processed/client_1/transactions_normalized.parquet`
- `data/processed/client_2/transactions_normalized.parquet`

## Run federated training

### 1. Start the server

Open a terminal and run:

```powershell
uv run python server/fl_server.py
```

The Flower server listens on `0.0.0.0:8080` and will load the latest checkpoint from `checkpoints/` if available.

### 2. Start the clients

Open three additional terminals and start one client per terminal.

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

Clients will connect to the server and run federated training for the configured number of rounds.

If the server fails at startup with a checkpoint key mismatch (for example `KeyError: 'net.0.bias'`), the existing checkpoint is incompatible with the current model. Remove or move old checkpoint files from `checkpoints/` and restart the server.

### 3. Verify training completion

Checkpoint files are written to the `checkpoints/` folder as training proceeds, e.g.:

- `checkpoints/round_001.pt`
- `checkpoints/round_002.pt`
- ...

## Evaluate the trained model

After training completes, run evaluation using the latest federated checkpoint:

```powershell
uv run python model/evaluate.py
```

This writes results to `results/evaluation_report.json` and checks that `AUPRC >= 0.75`.

## Optional: Calibrate the model

If you want to apply Platt scaling to a saved checkpoint:

```powershell
uv run python model/calibrate.py --checkpoint checkpoints/round_010.pt --data data/processed/client_0/transactions_normalized.parquet
```

Adjust the checkpoint path as needed.

## Run tests

Use `uv` to execute pytest inside the project environment:

```powershell
uv run pytest
```

Or run a single test file:

```powershell
uv run pytest tests/unit/test_converter.py
```

## Notes

- This project is managed via `pyproject.toml` and `uv.lock`
- Use `uv run ...` so commands execute inside the configured project environment
- The federated server and clients should be started in separate terminals for local end-to-end training
