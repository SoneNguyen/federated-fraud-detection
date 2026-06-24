"""Checkpoint discovery and ranking for the inference GUI."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRIC_ALIASES = {
    "auprc": ("val_auprc", "AUPRC", "auprc", "average_precision"),
    "auroc": ("val_auroc", "AUROC", "auroc", "roc_auc"),
    "f1": ("val_f1", "F1_best", "best_f1", "f1", "F1"),
    "loss": ("val_loss", "val_hybrid_loss", "loss"),
    "threshold": ("val_threshold", "threshold", "best_threshold"),
    "min_client_auprc": ("min_client_auprc",),
    "min_client_auroc": ("min_client_auroc",),
    "min_client_f1": ("min_client_f1",),
    "high_band_score": ("high_band_score",),
}

TARGETS = {"auprc": 0.70, "auroc": 0.90, "f1": 0.70}
HIGH_TARGETS = {"auprc": 0.85, "auroc": 0.95, "f1": 0.80}
CLIENT_FLOORS = {"auprc": 0.80, "auroc": 0.93, "f1": 0.75}


def list_model_records(
    checkpoint_dir: Path,
    results_dir: Path,
    *,
    selected_name: str | None = None,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """Return ranked checkpoint records using any metrics currently available."""
    checkpoints = [
        p
        for p in checkpoint_dir.glob("*.pt")
        if p.is_file() and re.fullmatch(r"round_\d+\.pt", p.name)
    ]
    history = _history_by_round(results_dir / "evaluation_history.json")
    latest = _safe_json(results_dir / "latest_metrics.json")
    best = _safe_json(results_dir / "best_round.json")
    eval_by_name = _target_eval_by_name(results_dir / "target_evaluation.json")

    records: list[dict[str, Any]] = []
    for path in checkpoints:
        round_no = _extract_round(path.name)
        sidecar = _safe_json(path.with_suffix(".json"))
        merged: dict[str, Any] = {}
        if round_no is not None:
            merged.update(history.get(round_no, {}))
        if best.get("checkpoint") == path.name:
            merged.update(best)
        if latest.get("checkpoint") == path.name or latest.get("round") == round_no:
            merged.update(latest)
        merged.update(eval_by_name.get(path.name, {}))
        merged.update(sidecar)

        metrics = _normalize_metrics(merged)
        score, score_parts = _score(path.name, metrics)
        records.append(
            {
                "name": path.name,
                "checkpoint": path.name,
                "kind": _kind(path.name),
                "round": round_no,
                "client_id": _extract_client(path.name),
                "metrics": metrics,
                "score": round(score, 6),
                "score_parts": score_parts,
                "threshold": _threshold(metrics),
                "status": _status(metrics),
                "reason": _reason(path.name, metrics),
                "updated_at": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "selected": path.name == selected_name,
                "recommended": False,
            }
        )

    records.sort(key=_sort_key, reverse=True)
    for record in records[:1]:
        record["recommended"] = True
    return records[:limit]


def recommended_checkpoint(checkpoint_dir: Path, results_dir: Path) -> Path | None:
    records = list_model_records(checkpoint_dir, results_dir, limit=1)
    if not records:
        return None
    return checkpoint_dir / records[0]["checkpoint"]


def metrics_for_checkpoint(checkpoint: Path, results_dir: Path) -> dict[str, Any]:
    records = list_model_records(
        checkpoint.parent,
        results_dir,
        selected_name=checkpoint.name,
        limit=500,
    )
    for record in records:
        if record["checkpoint"] == checkpoint.name:
            return record
    return {
        "checkpoint": checkpoint.name,
        "metrics": {},
        "threshold": 0.5,
        "status": "unknown",
        "score": 0.0,
        "score_parts": {},
    }


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _history_by_round(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _safe_json_list(path):
        round_no = _to_int(row.get("round"))
        if round_no is not None:
            out[round_no] = row
    return out


def _target_eval_by_name(path: Path) -> dict[str, dict[str, Any]]:
    data = _safe_json(path)
    candidates = data.get("candidates", data.get("all", []))
    out = {}
    if isinstance(candidates, list):
        for row in candidates:
            if isinstance(row, dict) and isinstance(row.get("checkpoint"), str):
                out[row["checkpoint"]] = row
    for key in ("best_single", "ensemble"):
        row = data.get(key)
        if isinstance(row, dict) and isinstance(row.get("checkpoint"), str):
            out[row["checkpoint"]] = row
    return out


def _normalize_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        value = _first_number(data, aliases)
        if value is not None:
            metrics[canonical] = value

    for key in (
        "target_met",
        "high_target_met",
        "client_floor_met",
        "learning_state",
        "rounds_since_best_loss",
        "best_val_loss",
        "train_loss",
        "train_loss_delta",
    ):
        if key in data:
            metrics[key] = _clean_value(data[key])

    return metrics


def _first_number(data: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        if key not in data:
            continue
        value = _to_float(data[key])
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any) -> int | None:
    out = _to_float(value)
    return int(out) if out is not None else None


def _clean_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    num = _to_float(value)
    return num if num is not None else value


def _score(name: str, metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    metric_scores = []
    for metric, target in HIGH_TARGETS.items():
        value = _to_float(metrics.get(metric))
        if value is not None:
            metric_scores.append(min(value / target, 1.15))
    quality = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0

    floor_scores = []
    for metric, floor in CLIENT_FLOORS.items():
        value = _to_float(metrics.get(f"min_client_{metric}"))
        if value is not None:
            floor_scores.append(min(value / floor, 1.15))
    fairness = sum(floor_scores) / len(floor_scores) if floor_scores else quality * 0.9

    loss = _to_float(metrics.get("loss"))
    loss_bonus = 0.0 if loss is None else min(0.12, 0.08 / max(loss, 0.08))

    tag_bonus = 0.0
    if metrics.get("target_met") is True:
        tag_bonus += 0.04
    if metrics.get("high_target_met") is True:
        tag_bonus += 0.05
    if metrics.get("client_floor_met") is True:
        tag_bonus += 0.05
    if _kind(name) == "global":
        tag_bonus += 0.02
    if _kind(name) == "client":
        tag_bonus -= 0.03

    high_band = _to_float(metrics.get("high_band_score")) or 0.0
    score = quality * 0.5 + fairness * 0.25 + high_band * 0.15 + loss_bonus + tag_bonus
    parts = {
        "quality": round(quality, 6),
        "fairness": round(fairness, 6),
        "loss_bonus": round(loss_bonus, 6),
        "tag_bonus": round(tag_bonus, 6),
        "high_band": round(high_band, 6),
    }
    return score, parts


def _sort_key(record: dict[str, Any]) -> tuple[float, int, float]:
    return (
        float(record.get("score", 0.0)),
        int(record.get("round") or 0),
        _updated_epoch(record.get("updated_at")),
    )


def _updated_epoch(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _threshold(metrics: dict[str, Any]) -> float:
    raw = _to_float(metrics.get("threshold"))
    if raw is None:
        return 0.5
    return max(0.01, min(0.995, raw))


def _status(metrics: dict[str, Any]) -> str:
    if metrics.get("high_target_met") is True:
        return "high-band"
    if metrics.get("client_floor_met") is True:
        return "client-floor"
    if metrics.get("target_met") is True:
        return "target-met"
    if _meets(TARGETS, metrics):
        return "target-met"
    if all(metric in metrics for metric in ("auprc", "auroc", "f1")):
        return "below-target"
    return "metrics-limited"


def _meets(targets: dict[str, float], metrics: dict[str, Any]) -> bool:
    for metric, target in targets.items():
        value = _to_float(metrics.get(metric))
        if value is None or value < target:
            return False
    return True


def _reason(name: str, metrics: dict[str, Any]) -> str:
    if metrics.get("high_target_met") is True:
        return "High-band target reached"
    if metrics.get("client_floor_met") is True:
        return "Worst-client floor reached"
    if metrics.get("target_met") is True or _meets(TARGETS, metrics):
        return "Core target reached"
    if _kind(name) == "client":
        return "Client specialist checkpoint"
    return "Ranked by available metrics"


def _kind(name: str) -> str:
    if name.startswith("client_"):
        return "client"
    if name.startswith("round_"):
        return "global"
    return "other"


def _extract_round(name: str) -> int | None:
    match = re.search(r"round_(\d+)", name)
    return int(match.group(1)) if match else None


def _extract_client(name: str) -> int | None:
    match = re.search(r"client_(\d+)_", name)
    return int(match.group(1)) if match else None
