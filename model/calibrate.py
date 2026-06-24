"""Post-training probability calibration using Platt scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.data.dataset import FEATURE_ORDER, LABEL
from src.model.fraud_mlp import FraudMLP


def get_raw_scores(
    model: torch.nn.Module,
    parquet_path: str,
    val_split: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on the validation split and return probabilities and labels."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    split = int(len(df) * (1 - val_split))
    val_df = df.iloc[split:].reset_index(drop=True)

    features = torch.tensor(val_df[FEATURE_ORDER].to_numpy(), dtype=torch.float32)
    labels = val_df[LABEL].to_numpy(dtype=np.float32)

    model.eval()
    with torch.no_grad():
        device = getattr(model, "device", torch.device("cpu"))
        features = features.to(device)
        probs = torch.sigmoid(model(features)).cpu().numpy().squeeze()

    return np.asarray(probs), np.asarray(labels)


def fit_platt_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """Fit a logistic regression on raw scores to calibrate probabilities."""
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(probs.reshape(-1, 1), labels)
    return calibrator


def calibrate(
    checkpoint_path: str,
    data_path: str,
    out_dir: str = "checkpoints",
) -> dict[str, float | int | str]:
    """Fit and save calibration coefficients."""
    model = FraudMLP()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    probs, labels = get_raw_scores(model, data_path)
    calibrator = fit_platt_scaling(probs, labels)

    coef = float(calibrator.coef_[0][0])
    intercept = float(calibrator.intercept_[0])

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    result = {
        "checkpoint": checkpoint_path,
        "data": data_path,
        "platt_coef": coef,
        "platt_intercept": intercept,
        "n_val_samples": int(len(labels)),
        "val_fraud_rate": float(labels.mean()),
    }
    calib_path = out / "calibration_params.json"
    calib_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Calibration saved to {calib_path}")
    print(f"platt_coef={coef:.4f} platt_intercept={intercept:.4f}")
    return result


def apply_calibration(
    raw_prob: float,
    calib_path: str = "checkpoints/calibration_params.json",
) -> float:
    """Apply saved Platt scaling to a raw sigmoid probability."""
    params = json.loads(Path(calib_path).read_text(encoding="utf-8"))
    logit = params["platt_coef"] * raw_prob + params["platt_intercept"]
    return float(1.0 / (1.0 + np.exp(-logit)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()
    calibrate(args.checkpoint, args.data, args.out)


if __name__ == "__main__":
    main()
