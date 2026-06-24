"""Inspect and organize run-specific training outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_paths import archive_flat_runtime_files


def _round_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _run_names() -> list[str]:
    names = set()
    for root in (Path("outputs/checkpoints"), Path("results")):
        if root.exists():
            names.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(names)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def list_runs() -> None:
    names = _run_names()
    if not names:
        print("No run folders found.")
        return

    print("run              checkpoints  latest_round  best_round  target  checkpoints_dir")
    for name in names:
        checkpoint_dir = Path("outputs/checkpoints") / name
        results_dir = Path("results") / name
        checkpoints = sorted(checkpoint_dir.glob("round_*.pt"), key=_round_number)
        latest_round = _round_number(checkpoints[-1]) if checkpoints else 0
        best = _read_json(results_dir / "best_round.json")
        best_round = int(best.get("round", 0)) if best else 0
        target = best.get("target_met", "n/a") if best else "n/a"
        print(
            f"{name:<16s} {len(checkpoints):>11d} {latest_round:>13d} "
            f"{best_round:>10d} {str(target):>7s}  {checkpoint_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage run-specific output folders.")
    parser.add_argument(
        "--archive-flat",
        action="store_true",
        help="Move legacy flat training files into outputs/archive/flat_runtime.",
    )
    args = parser.parse_args()

    if args.archive_flat:
        moved = archive_flat_runtime_files()
        print(f"Archived flat runtime files: {len(moved)}")
    list_runs()


if __name__ == "__main__":
    main()
