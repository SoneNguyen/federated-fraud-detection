# model/evaluate.py
import torch, pandas as pd, numpy as np, json, argparse
from pathlib import Path
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_curve, f1_score)
from client.model import FraudMLP
from client.dataset import FEATURE_ORDER, LABEL

with open("contracts/normalization_params.json") as f:
    NORM = json.load(f)
NUMERIC = list(NORM.keys())

def load_and_prep(parquet_path):
    df = pd.read_parquet(parquet_path)
    # Data is already normalized — just extract X, y
    X = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
    y = df[LABEL].values
    return X, y

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        probs = model(X).numpy().squeeze()
    auprc  = average_precision_score(y, probs)
    auroc  = roc_auc_score(y, probs)
    prec, rec, thresholds = precision_recall_curve(y, probs)
    f1s    = 2 * prec * rec / (prec + rec + 1e-9)
    best_t = float(thresholds[f1s[:-1].argmax()]) if len(thresholds) else 0.5
    best_f1 = float(f1s.max())
    return {"AUPRC": round(auprc,4), "AUROC": round(auroc,4),
            "F1_best": round(best_f1,4), "threshold": round(best_t,3),
            "fraud_rate": round(float(y.mean()),4), "n_samples": len(y)}

def run_evaluation():
    results = {}
    # Use client_0 test set as the common evaluation set
    test_path = "data/processed/client_0/transactions_normalized.parquet"
    df_test = pd.read_parquet(test_path)
    n = len(df_test); split = int(n * 0.85)
    X_test, y_test = load_and_prep(test_path)
    X_test, y_test = X_test[split:], y_test[split:]

    # 1. Federated model (latest checkpoint)
    fl_ckpt = sorted(Path("checkpoints").glob("round_*.pt"))[-1]
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