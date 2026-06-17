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
from src.model.fraud_mlp import FraudMLP
from src.data.dataset import FEATURE_ORDER, LABEL
from src.data.feature_registry import SCHEMA_VERSION

with open("config/normalization_params.json") as f:
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
    result["schema_version"] = SCHEMA_VERSION
    return result


def evaluate(model: torch.nn.Module, X: torch.Tensor, y: np.ndarray) -> dict[str, Any]:
    model.eval()
    # Get device from model if it has one, otherwise use CPU
    if hasattr(model, 'device') and isinstance(model.device, torch.device):
        device = model.device
    else:
        device = torch.device('cpu')
    X = X.to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().squeeze()
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

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Federated model (latest checkpoint)
    checkpoint_dir = Path("outputs/checkpoints")
    fl_ckpts = sorted(checkpoint_dir.glob("round_*.pt"))
    if not fl_ckpts:
        raise FileNotFoundError(
            f"No federated checkpoint files found in {checkpoint_dir}/round_*.pt. "
            "Please run training or populate the checkpoints directory before evaluation."
        )

    fl_model = FraudMLP(device=str(device))
    compatible_ckpt = None
    expected_shapes = {k: v.shape for k, v in fl_model.state_dict().items()}

    for ckpt_path in reversed(fl_ckpts):
        try:
            state = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue

        if not isinstance(state, dict):
            continue

        if set(state.keys()) != set(expected_shapes.keys()):
            continue

        if all(state[k].shape == expected_shapes[k] for k in expected_shapes):
            compatible_ckpt = ckpt_path
            break

    if compatible_ckpt is None:
        raise RuntimeError(
            "No compatible federated checkpoint found for the current model architecture. "
            "Please retrain the model using the current schema and architecture."
        )

    fl_model.load_state_dict(torch.load(compatible_ckpt, map_location=fl_model.device))
    results["federated"] = evaluate(fl_model, X_test, y_test)

    # 2. Local-only baseline (train on client_0 only, no federation)
    # (load from local_only_baseline.pt if exists, else skip)
    local_ckpt = checkpoint_dir / "local_only_baseline.pt"
    if local_ckpt.exists():
        local_model = FraudMLP(device=str(device))
        local_model.load_state_dict(torch.load(local_ckpt, map_location=local_model.device))
        results["local_only"] = evaluate(local_model, X_test, y_test)

    print("\n=== Evaluation Results ===")
    for name, r in results.items():
        print(f"{name:15s}: AUPRC={r['AUPRC']:.4f} | AUROC={r['AUROC']:.4f} | F1={r['F1_best']:.4f}")
    with open("results/evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    # Assert FL meets minimum bar
    assert results["federated"]["AUPRC"] >= 0.70, f"FL AUPRC {results['federated']['AUPRC']} below 0.70 target"
    assert results["federated"]["AUROC"] >= 0.90, f"FL AUROC {results['federated']['AUROC']} below 0.90 target"
    assert results["federated"]["F1_best"] >= 0.70, f"FL F1 {results['federated']['F1_best']} below 0.70 target"
    print("\nAbsolute target passed: AUPRC >= 0.70, AUROC >= 0.90, F1 >= 0.70")

if __name__ == "__main__":
    run_evaluation()
