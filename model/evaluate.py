# model/evaluate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from client.model import FraudMLP
from client.dataset import FEATURE_ORDER, LABEL

with open("contracts/normalization_params.json") as f:
    NORM = json.load(f)
NUMERIC = list(NORM.keys())


def load_and_prep(parquet_path: str) -> tuple[torch.Tensor, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    # Data is already normalized — just extract X, y
    X = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
    y = np.asarray(df[LABEL].values, dtype=np.int8)
    return X, y


def load_test(parquet_path: str, test_frac: float = 0.15) -> tuple[torch.Tensor, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    test_size = int(n * test_frac)
    split = max(0, n - test_size)
    X = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
    y = np.asarray(df[LABEL].values, dtype=np.int8)
    return X[split:], y[split:]


def eval_model(model: torch.nn.Module, X: torch.Tensor, y: np.ndarray) -> dict[str, Any]:
    result = evaluate(model, X, y)
    with open("contracts/schema.json") as f:
        schema = json.load(f)
    result["schema_version"] = schema["feature_schema"]["version"]
    return result


def evaluate(model: torch.nn.Module, X: torch.Tensor, y: np.ndarray) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probs = model(X).numpy().squeeze()
    auprc = average_precision_score(y, probs)
    auroc = roc_auc_score(y, probs)
    prec, rec, thresholds = precision_recall_curve(y, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_t = float(thresholds[f1s[:-1].argmax()]) if len(thresholds) else 0.5
    best_f1 = float(f1s.max())
    return {
        "AUPRC": round(auprc, 4),
        "AUROC": round(auroc, 4),
        "F1_best": round(best_f1, 4),
        "threshold": round(best_t, 3),
        "fraud_rate": round(float(y.mean()), 4),
        "n_samples": len(y),
    }

def run_evaluation():
    results = {}
    # Use client_0 test set as the common evaluation set
    test_path = "data/processed/client_0/transactions_normalized.parquet"
    df_test = pd.read_parquet(test_path)
    n = len(df_test); split = int(n * 0.85)
    X_test, y_test = load_and_prep(test_path)
    X_test, y_test = X_test[split:], y_test[split:]

    # 1. Federated model (latest checkpoint)
    fl_ckpts = sorted(Path("checkpoints").glob("round_*.pt"))
    if not fl_ckpts:
        raise FileNotFoundError(
            "No federated checkpoint files found in checkpoints/round_*.pt. "
            "Please run training or populate the checkpoints directory before evaluation."
        )
    fl_ckpt = fl_ckpts[-1]
    fl_model = FraudMLP()
    fl_model.load_state_dict(torch.load(fl_ckpt, map_location="cpu"))
    results["federated"] = evaluate(fl_model, X_test, y_test)

    # 2. Local-only baseline (train on client_0 only, no federation)
    # (load from local_only_baseline.pt if exists, else skip)
    local_ckpt = Path("checkpoints/local_only_baseline.pt")
    if local_ckpt.exists():
        local_model = FraudMLP()
        local_model.load_state_dict(torch.load(local_ckpt, map_location="cpu"))
        results["local_only"] = evaluate(local_model, X_test, y_test)

    print("\n=== Evaluation Results ===")
    for name, r in results.items():
        print(f"{name:15s}: AUPRC={r['AUPRC']:.4f} | AUROC={r['AUROC']:.4f} | F1={r['F1_best']:.4f}")
    with open("results/evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    # Assert FL meets minimum bar
    assert results["federated"]["AUPRC"] >= 0.75,         f"FL AUPRC {results['federated']['AUPRC']} below 0.75 target"
    print("\nTarget AUPRC >= 0.75: PASSED")

if __name__ == "__main__":
    run_evaluation()