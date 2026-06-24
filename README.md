# FL Fraud Detection

Federated fraud detection for transaction screening using Flower, PyTorch,
FastAPI, and React. The project now focuses on three production-facing concerns:
Vietnam-oriented dataset expansion, scalable 3/10/100-client experiments, and
fault-tolerant training recovery.

Core documents:

```text
docs/SYSTEM_REFERENCE.md   Technical architecture, algorithms, data path, limitations
docs/OPERATIONS.md         Setup, training, evaluation, recovery, API, GUI
docs/IEEE_REPORT.tex       IEEE-style project report source
```

Prepare the baseline dataset:

```powershell
uv sync
uv run python dataset/load_ieee_cis.py
```

Train with the latest Flower runtime locally:

```powershell
$env:NUM_CLIENTS=10
$env:MODEL_RUN="10_clients"
uv run python dataset/load_ieee_cis.py
uv run python -m scripts.run_local_flower --num-clients 10 --rounds 100 --model-run 10_clients
```

Train 100 virtual clients on one machine:

```powershell
$env:NUM_CLIENTS=100
$env:MODEL_RUN="100_clients_virtual"
uv run python -m scripts.run_virtual_federated --num-clients 100 --rounds 100
```

Core server aggregation algorithm:

```text
src/server/aggregation.py
```

Partner or Vietnam transaction dataset adapter:

```powershell
uv run python -m dataset.load_custom_transactions `
  --input dataset\vietnam\transactions.csv `
  --mapping config\custom_transaction_mapping.example.json `
  --output-root dataset\processed_vietnam `
  --num-clients 10
```

Run the inference API:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run the transaction-screening GUI:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```
