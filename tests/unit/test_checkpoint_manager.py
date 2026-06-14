from pathlib import Path

import pytest

from src.server.checkpoint_manager import CheckpointManager


def test_checkpoint_manager_rollback_without_checkpoints(tmp_path: Path):
    manager = CheckpointManager(checkpoint_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.rollback()


def test_checkpoint_manager_rollback_returns_latest_checkpoint(tmp_path: Path):
    old_checkpoint = tmp_path / "old_checkpoint.ckpt"
    new_checkpoint = tmp_path / "new_checkpoint.ckpt"

    old_checkpoint.write_text("old")
    new_checkpoint.write_text("new")

    manager = CheckpointManager(checkpoint_dir=tmp_path)
    latest = manager.rollback()

    assert latest == new_checkpoint
    assert latest.read_text() == "new"
