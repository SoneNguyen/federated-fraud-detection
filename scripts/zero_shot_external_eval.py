"""Zero-shot external dataset evaluation.

This script intentionally does not train, fine-tune, refit normalization, or
start federated rounds. It maps held-out fraud datasets into the active feature
contract, applies the IEEE-CIS normalization parameters, and runs inference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from dataset.custom_transaction_adapter import to_ieee_like_frame
from dataset.load_ieee_cis import engineer
from scripts.run_paths import checkpoint_dir as default_checkpoint_dir
from scripts.run_paths import results_dir as default_results_dir
from src.data.feature_registry import FEATURE_ORDER, LABEL
from src.model.fraud_mlp import FraudMLP


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "vietnam-synthetic": {
        "path": "dataset/vietnam_synthetic/transactions.csv",
        "mapping": {
            "dataset_name": "Vietnam Synthetic",
            "source": "Synthetic Vietnam-style banking, wallet, QR, and gateway transactions",
            "time_unit": "auto",
            "amount_to_usd_rate": 0.00004,
            "default_product": "W",
            "fraud_values": [1, "1", True, "true", "fraud", "chargeback", "scam"],
            "product_map": {
                "wallet_payment": "W",
                "qr_transfer": "C",
                "bank_transfer": "C",
                "ecommerce": "W",
                "bill_payment": "S",
                "topup": "S",
                "card_payment": "R",
            },
            "columns": {
                "transaction_id": "transaction_id",
                "transaction_time": "timestamp",
                "amount": "amount_vnd",
                "label": "is_fraud",
                "product": "channel",
                "card_type": "card_type",
                "card_brand": "card_brand",
                "payer_email": "payer_email",
                "receiver_email": "receiver_email",
                "account_id": "customer_id",
                "merchant_id": "merchant_id",
                "device_id": "device_id",
                "device_type": "device_type",
                "region": "province",
                "distance_km": "distance_km",
                "account_age_days": "account_age_days",
                "days_since_last_transaction": "days_since_last_transaction",
                "transactions_1h": "transactions_1h",
                "transactions_24h": "transactions_24h",
                "chargeback_count": "chargeback_count",
            },
        },
    },
    "paysim": {
        "path": "dataset/synthetic/PS_20174392719_1491204439457_log.csv",
        "mapping": {
            "dataset_name": "PaySim",
            "source": "PaySim synthetic mobile-money transactions",
            "time_unit": "auto",
            "default_product": "W",
            "fraud_values": [1, "1", True, "true"],
            "product_map": {
                "payment": "W",
                "transfer": "C",
                "cash_out": "C",
                "cash_in": "W",
                "debit": "R",
            },
            "columns": {
                "transaction_time": "step",
                "amount": "amount",
                "label": "isFraud",
                "product": "type",
                "account_id": "nameOrig",
                "merchant_id": "nameDest",
            },
        },
    },
    "ccfraud": {
        "path": "dataset/creditcard/creditcard.csv",
        "mapping": {
            "dataset_name": "CreditCard",
            "source": "European credit-card fraud dataset",
            "time_unit": "auto",
            "default_product": "W",
            "fraud_values": [1, "1", True, "true"],
            "columns": {
                "transaction_time": "Time",
                "amount": "Amount",
                "label": "Class",
            },
        },
    },
    "baf-base": {
        "path": "dataset/bankaccount/Base.csv",
        "mapping": {
            "dataset_name": "BAF Base",
            "source": "Bank Account Fraud benchmark, base variant",
            "time_unit": "auto",
            "default_product": "W",
            "fraud_values": [1, "1", True, "true"],
            "product_map": {"AA": "W", "AB": "H", "AC": "C", "AD": "S", "AE": "R"},
            "columns": {
                "transaction_time": "month",
                "amount": "proposed_credit_limit",
                "label": "fraud_bool",
                "product": "payment_type",
                "device_type": "device_os",
                "account_age_days": "bank_months_count",
                "transactions_1h": "velocity_6h",
                "transactions_24h": "velocity_24h",
                "chargeback_count": "device_fraud_count",
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference-only zero-shot external dataset evaluation.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to newest target/global checkpoint.")
    parser.add_argument("--normalization", default="config/normalization_params.json")
    parser.add_argument("--output", default=str(default_results_dir() / "zero_shot_external_eval.json"))
    parser.add_argument("--datasets", nargs="+", default=["paysim", "ccfraud", "baf-base"])
    parser.add_argument(
        "--max-rows",
        type=int,
        default=25_000,
        help="Rows per external dataset for the normal run. Use --full to process complete files.",
    )
    parser.add_argument("--full", action="store_true", help="Process complete external files. This can take a long time.")
    parser.add_argument("--chunk-size", type=int, default=5_000, help="Rows per external scoring chunk.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--skip-ieee-test", action="store_true", help="Do not include IEEE-CIS processed test tails.")
    return parser.parse_args()


def load_raw_table(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=max_rows if max_rows > 0 else None)
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        return df.head(max_rows).copy() if max_rows > 0 else df
    raise ValueError(f"Unsupported input format: {path.suffix}")


def iter_raw_chunks(path: Path, max_rows: int, chunk_size: int):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        remaining = max_rows if max_rows > 0 else None
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining).copy()
                remaining -= len(chunk)
            if len(chunk) > 0:
                yield chunk
        return
    yield load_raw_table(path, max_rows)


def inspect_frame(df: pd.DataFrame, mapping: dict[str, Any]) -> dict[str, Any]:
    columns = mapping["columns"]
    label_col = columns["label"]
    amount_col = columns["amount"]
    entity_candidates = [
        columns.get(name)
        for name in ("account_id", "customer_id", "payer_id", "card_id")
        if columns.get(name)
    ]
    entity_col = next((col for col in entity_candidates if col in df.columns), None)
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "fraud_ratio": float((labels > 0).mean()),
        "amount_column": amount_col,
        "time_column": columns["transaction_time"],
        "entity_column_for_velocity": entity_col or "none",
        "velocity_reconstructed": bool(entity_col),
    }


def apply_trained_normalization(featured: pd.DataFrame, normalization_path: Path) -> pd.DataFrame:
    params = json.loads(normalization_path.read_text(encoding="utf-8"))
    normalized = featured.copy()
    for feature, stat in params.items():
        if feature not in normalized.columns:
            continue
        mean = float(stat["mean"])
        std = max(float(stat["std"]), 1e-8)
        normalized[feature] = ((normalized[feature].astype("float64") - mean) / std).astype("float32")
    normalized = normalized.reindex(columns=FEATURE_ORDER + [LABEL], fill_value=0.0)
    normalized[LABEL] = normalized[LABEL].astype("int8")
    return normalized


def load_checkpoint(path: str) -> Path:
    if path:
        return Path(path)
    checkpoint_dir = default_checkpoint_dir()
    patterns = [
        "round_*.pt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(checkpoint_dir.glob(pattern))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def predict_probabilities(model: FraudMLP, frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    X = torch.tensor(frame[FEATURE_ORDER].to_numpy(), dtype=torch.float32)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size].to(model.device)
            prob = torch.sigmoid(model(batch)).detach().cpu().numpy().reshape(-1)
            scores.append(prob)
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def recall_at_precision(labels: np.ndarray, scores: np.ndarray, target_precision: float = 0.50) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = recall[precision >= target_precision]
    if len(valid) == 0:
        return 0.0
    return float(np.max(valid))


def score_dataset(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = int(labels.sum())
    fraud_ratio = float(labels.mean()) if len(labels) else 0.0
    if positives == 0 or positives == len(labels):
        auprc = float("nan")
        recall_p50 = float("nan")
        auprc_lift = float("nan")
    else:
        auprc = float(average_precision_score(labels, scores))
        recall_p50 = recall_at_precision(labels, scores, 0.50)
        auprc_lift = auprc / max(fraud_ratio, 1e-12)
    return {
        "AUPRC": auprc,
        "Random AUPRC": fraud_ratio,
        "AUPRC lift": auprc_lift,
        "Recall@P=50%": recall_p50,
        "rows": int(len(labels)),
        "fraud_ratio": fraud_ratio,
    }


def evaluate_external_dataset(
    name: str,
    spec: dict[str, Any],
    model: FraudMLP,
    normalization_path: Path,
    max_rows: int,
    batch_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing dataset file {path}")
    cap_text = "full file" if max_rows == 0 else f"first {max_rows:,} rows"
    mapping = spec["mapping"]
    print(f"  stream {path} ({cap_text}, chunk={chunk_size:,})", flush=True)

    labels_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    rows_seen = 0
    fraud_seen = 0
    column_count = 0
    first_inspection: dict[str, Any] | None = None

    for chunk_idx, raw in enumerate(iter_raw_chunks(path, max_rows, chunk_size), start=1):
        chunk_started = time.perf_counter()
        inspection = inspect_frame(raw, mapping)
        if first_inspection is None:
            first_inspection = inspection
            print(
                "  inspect cols={columns} entity={entity} amount={amount} time={time}".format(
                    columns=inspection["columns"],
                    entity=inspection["entity_column_for_velocity"],
                    amount=inspection["amount_column"],
                    time=inspection["time_column"],
                ),
                flush=True,
            )
        rows_seen += int(inspection["rows"])
        fraud_seen += int(round(float(inspection["fraud_ratio"]) * int(inspection["rows"])))
        column_count = int(inspection["columns"])

        print(f"  chunk {chunk_idx}: map {len(raw):,} rows", flush=True)
        ieee_like = to_ieee_like_frame(raw, mapping)
        print(f"  chunk {chunk_idx}: engineer schema", flush=True)
        featured = engineer(ieee_like, ieee_like)
        print(f"  chunk {chunk_idx}: normalize + infer", flush=True)
        normalized = apply_trained_normalization(featured, normalization_path)
        labels = normalized[LABEL].to_numpy(dtype=np.int8)
        scores = predict_probabilities(model, normalized, batch_size)
        labels_parts.append(labels)
        score_parts.append(scores)
        print(
            f"  chunk {chunk_idx}: done rows={rows_seen:,} elapsed={time.perf_counter() - chunk_started:.1f}s",
            flush=True,
        )

    if not labels_parts:
        raise ValueError(f"{name}: no rows loaded from {path}")

    labels = np.concatenate(labels_parts)
    scores = np.concatenate(score_parts)
    result = score_dataset(labels, scores)
    elapsed = time.perf_counter() - started
    inspection = dict(first_inspection or {})
    inspection.update(
        {
            "rows": rows_seen,
            "columns": column_count,
            "fraud_ratio": float(labels.mean()) if len(labels) else 0.0,
            "chunk_size": chunk_size,
            "chunks": len(labels_parts),
        }
    )
    result.update(
        {
            "dataset": mapping.get("dataset_name", name),
            "source": mapping.get("source", str(path)),
            "mode": "zero-shot inference only",
            "normalization": str(normalization_path),
            "inspection": inspection,
            "seconds": elapsed,
        }
    )
    print(f"  done {name} in {elapsed:.1f}s", flush=True)
    return result


def evaluate_ieee_test(model: FraudMLP, batch_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    print("Scoring IEEE-CIS processed test tails...", flush=True)
    frames = []
    for path in sorted(Path("dataset/processed").glob("client_*/transactions_normalized.parquet")):
        df = pd.read_parquet(path)
        split = int(len(df) * 0.85)
        frames.append(df.iloc[split:].copy())
    if not frames:
        raise FileNotFoundError("No IEEE-CIS processed client files found under dataset/processed")
    test = pd.concat(frames, ignore_index=True)
    labels = test[LABEL].to_numpy(dtype=np.int8)
    scores = predict_probabilities(model, test, batch_size)
    result = score_dataset(labels, scores)
    result.update(
        {
            "dataset": "IEEE-CIS test",
            "source": "tail 15% from each processed IEEE-CIS client",
            "mode": "in-domain inference only",
            "seconds": time.perf_counter() - started,
        }
    )
    return result


def write_markdown_table(results: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "| Dataset | Mode | Rows | Fraud ratio | AUPRC | Lift vs random | Recall@P=50% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        auprc = row["AUPRC"]
        lift = row["AUPRC lift"]
        recall = row["Recall@P=50%"]
        lines.append(
            "| {dataset} | {mode} | {rows:,} | {fraud:.4f} | {auprc} | {lift} | {recall} |".format(
                dataset=row["dataset"],
                mode=row["mode"],
                rows=row["rows"],
                fraud=row["fraud_ratio"],
                auprc="n/a" if np.isnan(auprc) else f"{auprc:.4f}",
                lift="n/a" if np.isnan(lift) else f"{lift:.2f}x",
                recall="n/a" if np.isnan(recall) else f"{recall:.4f}",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    max_rows = 0 if args.full else args.max_rows
    checkpoint_path = load_checkpoint(args.checkpoint)
    normalization_path = Path(args.normalization)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FraudMLP(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))

    results: list[dict[str, Any]] = []
    if not args.skip_ieee_test:
        results.append(evaluate_ieee_test(model, args.batch_size))

    for name in args.datasets:
        if name not in DATASET_REGISTRY:
            known = ", ".join(sorted(DATASET_REGISTRY))
            raise ValueError(f"Unknown dataset {name!r}. Known: {known}")
        print(f"Zero-shot scoring {name}...", flush=True)
        results.append(
            evaluate_external_dataset(
                name,
                DATASET_REGISTRY[name],
                model,
                normalization_path,
                max_rows,
                args.batch_size,
                args.chunk_size,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "normalization": str(normalization_path),
        "max_rows": max_rows,
        "note": "Inference only. No training, no fine-tuning, no normalization refit.",
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table_path = output_path.with_suffix(".md")
    write_markdown_table(results, table_path)
    print(f"Saved {output_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
