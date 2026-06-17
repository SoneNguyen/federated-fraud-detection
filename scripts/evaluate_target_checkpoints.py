"""Evaluate saved checkpoints and a probability ensemble against target metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.data.dataset import FEATURE_ORDER, LABEL, load_validation_frame
from src.model.fraud_mlp import FraudMLP


TARGETS = {"auprc": 0.70, "auroc": 0.90, "f1": 0.70}


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    auprc = float(average_precision_score(y_true, probs))
    auroc = float(roc_auc_score(y_true, probs))
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(f1s[:-1].argmax()) if len(thresholds) else 0
    best_f1 = float(f1s.max())
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    return {
        "AUPRC": auprc,
        "AUROC": auroc,
        "F1_best": best_f1,
        "threshold": best_threshold,
        "target_met": (
            auprc >= TARGETS["auprc"]
            and auroc >= TARGETS["auroc"]
            and best_f1 >= TARGETS["f1"]
        ),
    }


def _load_validation(client_paths: list[Path], val_split: float) -> tuple[torch.Tensor, np.ndarray]:
    frames = [load_validation_frame(str(path), val_split=val_split) for path in client_paths]
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    x = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
    y = df[LABEL].values.astype(np.int8)
    return x, y


def _predict(checkpoint: Path, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model = FraudMLP(device=str(device))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            batch = x[start : start + 4096].to(device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy().squeeze())
    return np.concatenate(probs)


def _candidate_checkpoints(checkpoint_dir: Path) -> list[Path]:
    tagged = sorted(checkpoint_dir.glob("best_target_round_*.pt"))
    target_met = sorted(checkpoint_dir.glob("target_met_round_*.pt"))
    rounds = sorted(checkpoint_dir.glob("round_*.pt"))
    specialists = sorted(checkpoint_dir.glob("client_*_round_*.pt"))
    seen = set()
    candidates = []
    for path in [*target_met, *tagged, *specialists, *rounds]:
        if path.name not in seen:
            seen.add(path.name)
            candidates.append(path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--ensemble-top", type=int, default=5)
    parser.add_argument("--out", default="results/target_evaluation.json")
    args = parser.parse_args()

    client_paths = [
        Path(f"data/processed/client_{cid}/transactions_normalized.parquet")
        for cid in range(3)
    ]
    x, y = _load_validation(client_paths, args.val_split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    candidates = _candidate_checkpoints(checkpoint_dir)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    scored = []
    all_probs = {}
    for checkpoint in candidates:
        probs = _predict(checkpoint, x, device)
        m = _metrics(y, probs)
        scored.append({"checkpoint": checkpoint.name, **m})
        all_probs[checkpoint.name] = probs

    scored.sort(key=lambda item: (item["target_met"], item["F1_best"], item["AUPRC"]), reverse=True)
    top_names = [item["checkpoint"] for item in scored[: args.ensemble_top]]
    ensemble_probs = np.mean([all_probs[name] for name in top_names], axis=0)
    ensemble = {
        "checkpoint": f"ensemble_top_{len(top_names)}",
        "members": top_names,
        **_metrics(y, ensemble_probs),
    }

    result = {
        "targets": TARGETS,
        "n_samples": int(len(y)),
        "fraud_rate": float(y.mean()),
        "best_single": scored[0],
        "ensemble": ensemble,
        "all": scored,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps({"best_single": scored[0], "ensemble": ensemble}, indent=2))


if __name__ == "__main__":
    main()
