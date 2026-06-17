"""Checkpoint manager for federated learning server."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable, Optional

import torch


class CheckpointManager:
    """Manages checkpoint storage and rollback semantics for server workflows."""

    def __init__(self, checkpoint_dir: Path | str = "outputs\\checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state_dict: dict, metadata: Optional[dict] = None) -> Path:
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(state_dict, checkpoint_path)

        if metadata is not None:
            metadata_path = self.checkpoint_dir / f"{name}.json"
            metadata["saved_at"] = metadata.get("saved_at") or datetime.now(UTC).isoformat()
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

        return checkpoint_path

    def latest(self) -> Optional[Path]:
        return self._latest_checkpoint()

    def prune_round_checkpoints(
        self,
        keep_last: int,
        protected_stems: Iterable[str] = (),
    ) -> int:
        """Remove old round checkpoints while keeping recent and tagged models."""
        if keep_last <= 0:
            return 0

        protected = set(protected_stems)
        numbered_rounds: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("round_*.pt"):
            match = re.fullmatch(r"round_(\d+)", path.stem)
            if match:
                numbered_rounds.append((int(match.group(1)), path))

        numbered_rounds.sort(key=lambda item: item[0])
        stale = numbered_rounds[:-keep_last]
        removed = 0
        for _, path in stale:
            if path.stem in protected:
                continue
            for target in (path, path.with_suffix(".json")):
                if target.exists():
                    target.unlink()
                    removed += 1
        return removed

    def rollback(self) -> Path:
        """Copy and return the latest checkpoint path for rollback."""
        checkpoint = self._latest_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found in {self.checkpoint_dir.resolve()}"
            )

        rollback_target = self.checkpoint_dir / "rollback_active.pt"
        shutil.copy2(checkpoint, rollback_target)

        metadata_path = checkpoint.with_suffix(".json")
        if metadata_path.exists():
            shutil.copy2(metadata_path, self.checkpoint_dir / "rollback_active.json")

        return checkpoint

    def _latest_checkpoint(self) -> Optional[Path]:
        checkpoint_extensions = {".pt", ".ckpt", ".pth"}
        checkpoints = [
            path
            for path in self.checkpoint_dir.iterdir()
            if path.is_file() and path.suffix in checkpoint_extensions
            and not path.stem.startswith("client_")
        ]
        if not checkpoints:
            return None
        round_checkpoints = [
            path for path in checkpoints if re.fullmatch(r"round_\d+", path.stem)
        ]
        if round_checkpoints:
            return max(
                round_checkpoints,
                key=lambda path: int(path.stem.split("_")[1]),
            )
        return max(checkpoints, key=lambda path: path.stat().st_mtime)
