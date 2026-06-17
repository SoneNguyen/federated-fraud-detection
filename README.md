# FL Fraud Detection

Federated fraud detection on IEEE-CIS using Flower, PyTorch, FastAPI, and a
React demo GUI.

Use these three docs:

```text
docs/PROJECT_REFERENCE.md      Technical explanation of the whole project
docs/SETUP_AND_RUNNING.md      Install, train, monitor, recover, API, GUI, tests
docs/PRESENTATION_GUIDE.md     30 minute presentation plan and Q&A answers
```

Quick start:

```powershell
uv sync
uv run python data/load_ieee_cis.py
```

Run the demo API:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run the GUI:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```
