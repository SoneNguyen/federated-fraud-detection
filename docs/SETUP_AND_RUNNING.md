# Setup and Running Guide

This is the operational guide. Use it when someone asks how to install, prepare
data, train, monitor, recover, run the API, run the GUI, or test the project.

## 1. Requirements

Required:

```text
Python 3.12
uv
Node.js and npm
IEEE-CIS raw data files
```

Optional but recommended:

```text
NVIDIA GPU with CUDA
```

The client launcher automatically uses CUDA when available. GPU performance
settings such as AMP, TF32, batch size, and pinned memory are baked into the
training scripts, so normal usage does not require a long list of environment
variables.

## 2. Install

From the repository root:

```powershell
uv sync
```

Install frontend dependencies only if `app/node_modules` is missing:

```powershell
cd app
npm install
cd ..
```

## 3. Raw Data Layout

Place the IEEE-CIS raw files here:

```text
data/ieee_cis/train_transaction.csv
data/ieee_cis/train_identity.csv
```

## 4. Prepare Data

Run:

```powershell
uv run python data/load_ieee_cis.py
```

Expected outputs:

```text
config/normalization_params.json
data/processed/client_0/transactions_normalized.parquet
data/processed/client_1/transactions_normalized.parquet
data/processed/client_2/transactions_normalized.parquet
data/processed/preprocessing_report.json
```

Current processed data:

```text
rows: 590,540
features: 316
clients: 3
label: is_fraud
schema: fraud-history
```

## 5. Train Federated Model

Open four terminals: one server and three clients.

### Terminal 1: Server

Fresh training run:

```powershell
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=1
$env:RESUME_FROM_CHECKPOINT=0
uv run python -m scripts.run_server
```

Resume from a checkpoint:

```powershell
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=0
$env:RESUME_FROM_CHECKPOINT=1
$env:RESUME_CHECKPOINT="target_met_round_024.pt"
uv run python -m scripts.run_server
```

The server listens on:

```text
localhost:8080
```

### Terminal 2: Client 0

```powershell
$env:CLIENT_ID=0
$env:DATA_PATH="data\processed\client_0\transactions_normalized.parquet"
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

### Terminal 3: Client 1

```powershell
$env:CLIENT_ID=1
$env:DATA_PATH="data\processed\client_1\transactions_normalized.parquet"
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

### Terminal 4: Client 2

```powershell
$env:CLIENT_ID=2
$env:DATA_PATH="data\processed\client_2\transactions_normalized.parquet"
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

Client startup markers:

```text
CLIENT phase=model
CLIENT phase=data
CLIENT phase=init
CLIENT phase=connect
```

If a client stops at `phase=connect`, it is ready and waiting for the server.

## 6. Monitor Training

Run:

```powershell
uv run python -m scripts.monitor_training
```

Important files:

```text
results/latest_metrics.json
results/evaluation_history.json
results/best_round.json
outputs/checkpoints/
```

Server log shape:

```text
R055 eval state=learning target=1 high=0 floor=0 learn=1.0254 band=0.8820 loss=0.0489 auprc=0.7480/0.7100 auroc=0.9470/0.9310 f1=0.7270/0.7010
```

Read metrics as:

```text
global / worst-client
```

Fields:

```text
state   learning, mixed, stalled, or regressing
target  core target reached
high    high-band global target reached
floor   worst-client floor reached
loss    validation loss
auprc   global AUPRC / worst-client AUPRC
auroc   global AUROC / worst-client AUROC
f1      global F1 / worst-client F1
```

## 7. Evaluate Checkpoints

Run:

```powershell
uv run python -m scripts.evaluate_target_checkpoints
```

This evaluates saved checkpoints against:

```text
AUPRC >= 0.70
AUROC >= 0.90
F1    >= 0.70
```

## 8. Run Inference API

Start:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Useful endpoints:

```text
GET  /health
GET  /models
POST /models/select
POST /predict-demo
POST /predict
POST /reload
```

The API loads the highest-ranked compatible checkpoint by default.

## 9. Run GUI Demo

Start API first, then:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

GUI demo flow:

```text
1. Confirm API online.
2. Show selected/recommended checkpoint.
3. Show AUPRC/AUROC/F1 chart.
4. Enter transaction amount and currency.
5. Toggle advanced signals if needed.
6. Click Score.
7. Explain probability, threshold, decision, USD conversion, and FX source.
```

## 10. Crash Recovery

If the server crashes:

```powershell
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=0
$env:RESUME_FROM_CHECKPOINT=1
$env:RESUME_CHECKPOINT="target_met_round_024.pt"
uv run python -m scripts.run_server
```

If the API needs to reload the latest ranked model:

```powershell
curl.exe -X POST http://127.0.0.1:8000/reload
```

Rollback file created by drift alert:

```text
outputs/checkpoints/rollback_active.pt
```

## 11. Test

Backend tests:

```powershell
uv run pytest -q
```

Frontend build:

```powershell
cd app
npm run build
```

Expected status:

```text
pytest: all tests pass
vite: build succeeds
```

Vite may warn that the bundled charting dependency is larger than 500 kB. That
is not a functional failure.

## 12. Troubleshooting

Client waits forever at `phase=connect`:

```text
Start the server first or check SERVER_ADDRESS.
```

CUDA memory pressure:

```powershell
$env:BATCH_SIZE=1024
uv run python -m scripts.run_client
```

Need CPU-only client:

```powershell
$env:DEVICE="cpu"
uv run python -m scripts.run_client
```

API says model not loaded:

```text
Check outputs/checkpoints contains compatible .pt files.
Run training or copy a valid checkpoint into outputs/checkpoints.
```

GUI says API offline:

```text
Start uvicorn on http://127.0.0.1:8000.
```

Currency API unavailable:

```text
The backend falls back to static rates and marks fx_source=static.
Prediction still works.
```
