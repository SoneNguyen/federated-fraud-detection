# Operations

## Install

```powershell
uv sync
cd app
npm install
cd ..
```

## Prepare IEEE-CIS

Place raw files at:

```text
dataset/ieee_cis/train_transaction.csv
dataset/ieee_cis/train_identity.csv
```

Prepare the default 3-client split:

```powershell
$env:NUM_CLIENTS=3
uv run python dataset/load_ieee_cis.py
```

Prepare larger splits:

```powershell
$env:NUM_CLIENTS=10
uv run python dataset/load_ieee_cis.py

$env:NUM_CLIENTS=100
uv run python dataset/load_ieee_cis.py
```

The active schema is `fraud-history-scalable` with 328 features. Re-run this
step after pulling schema changes; old processed Parquet files and old
checkpoints are not compatible with the new input dimension.

If Flower reports `Processed dataset schema is stale or incomplete`, rebuild
the processed split with the same client count you are about to train:

```powershell
$env:NUM_CLIENTS=10
uv run python dataset/load_ieee_cis.py
```

## Prepare Vietnam-Style Data

Generate synthetic Vietnam-style transactions:

```powershell
uv run python -m dataset.generate_vietnam_synthetic `
  --rows 250000 `
  --output dataset\vietnam_synthetic\transactions.csv `
  --report dataset\vietnam_synthetic\report.json
```

Convert to federated client partitions:

```powershell
uv run python -m dataset.load_custom_transactions `
  --input dataset\vietnam_synthetic\transactions.csv `
  --mapping config\vietnam_synthetic_mapping.json `
  --output-root dataset\processed_vietnam_synthetic `
  --num-clients 10
```

For real partner data, copy `config/custom_transaction_mapping.example.json`,
map the partner columns, then run:

```powershell
uv run python -m dataset.load_custom_transactions `
  --input dataset\vietnam\transactions.csv `
  --mapping config\custom_transaction_mapping.example.json `
  --output-root dataset\processed_vietnam `
  --num-clients 10
```

## Train With Latest Flower Runtime

For normal local training, use the orchestrator. It starts SuperLink,
SuperNodes, submits the Flower run, streams the run, and stops background
processes when the run finishes. This is the recommended portable path for
other machines because the runtime now self-heals common local failures:

```text
stale Flower DB     archived before startup instead of crashing on missing tables
busy ports          nearby free ports are selected by the orchestrator
missing data        processed IEEE-CIS data is rebuilt when raw files are present
stale schema        startup stops early with a precise rebuild message
client crashes      launcher restarts failed clients up to the configured limit
AppIO port clash    each client picks a nearby free local port
failure report      last repair/failure reason is written to outputs/runtime/last_failure.md
```

The launcher also plans local resources:

```text
active clients       enough clients for Flower, bounded for memory safety
threads per client   logical CPU cores divided across active clients
batch size           larger for 3/10 clients, bounded for 50+ clients
workers              0 on Windows to avoid subprocess overhead
schedule             adaptive when validation stalls or regresses
aggregation          robust blend plus update clipping for large runs
```

```powershell
$env:NUM_CLIENTS=10
$env:MODEL_RUN="10_clients"
$env:FRESH_RUN=1
$env:RESUME_FROM_CHECKPOINT=0
uv run python -m scripts.run_local_flower --num-clients 10 --rounds 100 --model-run 10_clients
```

If another terminal is already using the default Flower ports, the orchestrator
automatically chooses replacement ports and passes them to the server, clients,
and run submission.

Override resource planning only when needed:

```powershell
$env:TORCH_NUM_THREADS=2
$env:BATCH_SIZE=1024
$env:MAX_ACTIVE_CLIENTS=10
uv run python -m scripts.run_local_flower --num-clients 10 --rounds 100 --model-run 10_clients
```

Manual mode uses three terminals.

Terminal 1, infrastructure:

```powershell
$env:NUM_CLIENTS=10
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=1
$env:RESUME_FROM_CHECKPOINT=0
uv run python -m scripts.run_server
```

This starts Flower SuperLink. It is expected to keep running and wait.
It is not frozen; training starts only after SuperNodes connect and a run is
submitted.

By default the local SuperLink uses in-memory runtime state. This avoids
`sqlite3.OperationalError: database is locked` and stale schema errors such as
`no such table: fab` during repeated experiments.
If you previously started SuperLink with a persistent database, stop the old
Flower terminals before starting a new run. Only set `FLOWER_SUPERLINK_DB` when
you intentionally need persistent SuperLink task state. If a persistent database
is incompatible, startup archives it under:

```text
outputs/archive/self_heal/
```

Terminal 2, clients:

```powershell
$env:NUM_CLIENTS=10
uv run python -m scripts.launch_clients --num-clients 10
```

Terminal 3, submit the training run:

```powershell
$env:NUM_CLIENTS=10
uv run python -m scripts.submit_flower_run --num-clients 10 --rounds 100 --model-run 10_clients
```

## Train 100 Virtual Clients

Use this for local-machine 100-client performance comparison:

```powershell
$env:NUM_CLIENTS=100
$env:MODEL_RUN="100_clients_virtual"
uv run python -m scripts.run_virtual_federated --num-clients 100 --rounds 100
```

For 100 clients, the default virtual run samples a broad rotating subset each
round, uses target-aware robust aggregation, and adapts the client learning
schedule from recent validation behavior. This is the preferred comparison path
for scalability quality on one machine.

Resume a virtual run without archiving its existing folder:

```powershell
$env:NUM_CLIENTS=100
$env:MODEL_RUN="100_clients_virtual"
uv run python -m scripts.run_virtual_federated --num-clients 100 --rounds 100 --resume
```

Use the multi-process Flower launcher only for network/process stress testing:

```powershell
$env:NUM_CLIENTS=100
uv run python -m scripts.launch_clients --num-clients 100 --stagger-seconds 0.20
```

If Windows reports page-file or DLL loading errors, reduce active processes:

```powershell
uv run python -m scripts.launch_clients --num-clients 100 --max-active 20
```

## Monitor and Compare

Monitor the selected run:

```powershell
$env:MODEL_RUN="100_clients_virtual"
uv run python -m scripts.monitor_training
```

Compare scalability runs:

```powershell
uv run python -m scripts.compare_scalability_runs --runs 3_clients 10_clients 100_clients_virtual
```

Evaluate checkpoints:

```powershell
$env:NUM_CLIENTS=100
$env:MODEL_RUN="100_clients_virtual"
uv run python -m scripts.evaluate_target_checkpoints
```

## Zero-Shot External Evaluation

Quick smoke test:

```powershell
uv run python -m scripts.zero_shot_external_eval --max-rows 1000 --output results\zero_shot_smoke.json
```

Full capped comparison:

```powershell
uv run python -m scripts.zero_shot_external_eval --output results\zero_shot_external_eval.json
```

Include Vietnam synthetic:

```powershell
uv run python -m scripts.zero_shot_external_eval `
  --datasets vietnam-synthetic paysim ccfraud baf-base `
  --output results\zero_shot_with_vietnam_synthetic.json
```

## API and GUI

API:

```powershell
$env:MODEL_RUN="100_clients_virtual"
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

GUI:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

## Recovery

Resume a run:

```powershell
$env:NUM_CLIENTS=100
$env:MODEL_RUN="100_clients_virtual"
$env:FRESH_RUN=0
$env:RESUME_FROM_CHECKPOINT=1
$env:RESUME_CHECKPOINT="round_050.pt"
uv run python -m scripts.run_server
```

Run folders:

```text
outputs/checkpoints/<run>/
results/<run>/
outputs/runtime/last_failure.md
```

List known runs:

```powershell
uv run python -m scripts.manage_runs
```

Archive old flat runtime files:

```powershell
uv run python -m scripts.manage_runs --archive-flat
```

Run a local drift injection check:

```powershell
uv run python -m scripts.inject_drift
```

## Tests

```powershell
uv run pytest -q
```

Focused backend tests:

```powershell
uv run pytest tests/unit/test_aggregation_algorithm.py tests/unit/test_strategy.py tests/unit/test_failure_detector.py -q
```
