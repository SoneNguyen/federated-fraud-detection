"""Evaluate saved checkpoints and a probability ensemble against target metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.data.dataset import FEATURE_ORDER, LABEL, load_validation_frame
from src.model.fraud_mlp import FraudMLP
from scripts.run_paths import checkpoint_dir as default_checkpoint_dir
from scripts.run_paths import results_dir as default_results_dir


def _num_clients() -> int:
    raw = os.environ.get("NUM_CLIENTS", "3")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 3


TARGETS = {"auprc": 0.70, "auroc": 0.90, "f1": 0.70}


def _round_number(path: Path) -> int:
    stem = path.stem
    for token in reversed(stem.split("_")):
        if token.isdigit():
            return int(token)
    return -1


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
    y = df[LABEL].to_numpy(dtype=np.int8)
    return x, y


def _load_compatible_state(checkpoint: Path, device: torch.device) -> dict[str, torch.Tensor] | None:
    model = FraudMLP(device=str(device))
    expected = model.state_dict()
    try:
        state = torch.load(checkpoint, map_location=device)
    except Exception as exc:
        print(f"skip {checkpoint.name}: load failed: {exc}")
        return None
    if not isinstance(state, dict):
        print(f"skip {checkpoint.name}: checkpoint is not a state_dict")
        return None
    if set(state.keys()) != set(expected.keys()):
        print(f"skip {checkpoint.name}: key mismatch")
        return None
    for key, expected_tensor in expected.items():
        loaded = state.get(key)
        if not isinstance(loaded, torch.Tensor) or loaded.shape != expected_tensor.shape:
            shape = getattr(loaded, "shape", None)
            print(f"skip {checkpoint.name}: shape mismatch {key}={shape}, expected={expected_tensor.shape}")
            return None
    return state


def _predict(checkpoint: Path, x: torch.Tensor, device: torch.device) -> np.ndarray | None:
    model = FraudMLP(device=str(device))
    state = _load_compatible_state(checkpoint, device)
    if state is None:
        return None
    model.load_state_dict(state)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            batch = x[start : start + 4096].to(device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy().squeeze())
    return np.concatenate(probs)


def _candidate_checkpoints(checkpoint_dir: Path) -> list[Path]:
    key = lambda p: (_round_number(p), p.stat().st_mtime)
    rounds = sorted(checkpoint_dir.glob("round_*.pt"), key=key, reverse=True)
    seen = set()
    candidates = []
    for path in rounds:
        if path.name not in seen:
            seen.add(path.name)
            candidates.append(path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(default_checkpoint_dir()))
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--ensemble-top", type=int, default=5)
    parser.add_argument("--max-checkpoints", type=int, default=40, help="Newest candidates to evaluate. Use 0 for all.")
    parser.add_argument("--out", default=str(default_results_dir() / "target_evaluation.json"))
    parser.add_argument("--num-clients", type=int, default=_num_clients())
    parser.add_argument("--data-root", default="dataset/processed")
    args = parser.parse_args()

    client_paths = [
        Path(args.data_root) / f"client_{cid}" / "transactions_normalized.parquet"
        for cid in range(args.num_clients)
    ]
    missing = [path for path in client_paths if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing validation data for NUM_CLIENTS={args.num_clients}.\n{preview}{extra}"
        )
    x, y = _load_validation(client_paths, args.val_split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    candidates = _candidate_checkpoints(checkpoint_dir)
    if args.max_checkpoints > 0:
        candidates = candidates[: args.max_checkpoints]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    scored = []
    all_probs = {}
    for idx, checkpoint in enumerate(candidates, start=1):
        print(f"eval {idx}/{len(candidates)} {checkpoint.name}")
        probs = _predict(checkpoint, x, device)
        if probs is None:
            continue
        m = _metrics(y, probs)
        scored.append({"checkpoint": checkpoint.name, **m})
        all_probs[checkpoint.name] = probs
    if not scored:
        raise RuntimeError(
            f"No compatible FraudMLP checkpoints found in {checkpoint_dir}. "
            "Remove stale checkpoint files or run training with the current architecture."
        )

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
