# FL Fraud Detection

A federated learning fraud detection project configured for `uv`.

## Setup

Install dependencies with `uv`:

```powershell
uv install
```

## Data preparation

To prepare the IEEE-CIS dataset for federated training, run:

```powershell
uv run python data/load_ieee_cis.py
```

This script reads raw files from `data/ieee_cis/` and writes normalized per-client parquet data to `data/processed/client_{0,1,2}/`.

## Run tests

Use `uv` to execute tests in the current project environment:

```powershell
uv run pytest tests/unit/test_converter.py
```

## Notes

- This project is managed via `pyproject.toml` and `uv.lock`
- `requirements.txt` and `Makefile` are removed in favor of UV-native packaging
- Local environment artifacts are ignored by `.gitignore`
