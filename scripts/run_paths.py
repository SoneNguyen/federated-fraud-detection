"""Resolve per-run checkpoint and result folders."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def run_group_name(default_clients: int = 3) -> str:
    explicit = os.environ.get("MODEL_RUN")
    if explicit and explicit.strip():
        return explicit.strip()
    raw = os.environ.get("NUM_CLIENTS", str(default_clients)).strip()
    try:
        clients = max(int(raw), 1)
    except ValueError:
        return raw
    return f"{clients}_clients"


def checkpoint_dir() -> Path:
    return Path(os.environ.get("CHECKPOINT_DIR", str(Path("outputs/checkpoints") / run_group_name())))


def results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", str(Path("results") / run_group_name())))


def archive_flat_runtime_files() -> list[Path]:
    """Move legacy flat checkpoint/result files away from run-specific folders."""
    moved: list[Path] = []
    archive_root = Path("outputs/archive/flat_runtime")
    groups = (
        (
            Path("outputs/checkpoints"),
            ("round_*.pt", "round_*.json", "rollback_active.*", "active_training_run.json"),
        ),
        (
            Path("results"),
            (
                "latest_metrics.json",
                "evaluation_history.json",
                "best_round.json",
                "training_summary.md",
                "target_evaluation.json",
                "evaluation_report.json",
                "drift_alerts.jsonl",
            ),
        ),
    )
    for root, patterns in groups:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                target_dir = archive_root / root.as_posix().replace("/", "_")
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / path.name
                if target.exists():
                    target = target_dir / f"{path.stem}_{int(path.stat().st_mtime)}{path.suffix}"
                shutil.move(str(path), str(target))
                moved.append(target)
    return moved
