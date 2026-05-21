# FL Fraud Detection

A federated learning fraud detection project configured for `uv`.

## Setup

Install dependencies with `uv`:

```powershell
uv install
```

## Run tests

Use `uv` to execute tests in the current project environment:

```powershell
uv run pytest tests/unit/test_converter.py
```

## Notes

- This project is managed via `pyproject.toml` and `uv.lock`
- `requirements.txt` and `Makefile` are removed in favor of UV-native packaging
- Local environment artifacts are ignored by `.gitignore`
