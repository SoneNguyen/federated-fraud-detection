from __future__ import annotations

import socket
import sqlite3
from pathlib import Path

from src.system.resilience import (
    choose_superlink_ports,
    ensure_flower_database,
    ensure_flwr_home,
    port_is_free,
)


def test_incompatible_flower_database_is_archived(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "superlink.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE something_else (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    ensured = ensure_flower_database(db_path)

    assert ensured == db_path
    assert not db_path.exists()
    archived = list((tmp_path / "outputs/archive/self_heal").glob("superlink.db.incompatible_flower_db.*"))
    assert len(archived) == 1


def test_locked_incompatible_flower_database_gets_fresh_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "superlink.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE something_else (id INTEGER PRIMARY KEY)")
        conn.commit()

        ensured = ensure_flower_database(db_path)

        assert ensured is not None
        assert ensured != db_path
        assert ensured.name.startswith("superlink.recovered.")
        assert db_path.exists()
    finally:
        conn.close()


def test_compatible_flower_database_is_kept(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "superlink.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE fab (fab_hash TEXT PRIMARY KEY)")

    ensured = ensure_flower_database(db_path)

    assert ensured == db_path
    assert db_path.exists()


def test_choose_superlink_ports_avoids_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]

        chosen = choose_superlink_ports(
            f"127.0.0.1:{busy_port}",
            "127.0.0.1:19093",
            auto_ports=True,
        )

    assert chosen.fleet != f"127.0.0.1:{busy_port}"
    _, chosen_port = chosen.fleet.rsplit(":", 1)
    assert port_is_free("127.0.0.1", int(chosen_port))


def test_ensure_flwr_home_creates_directory(tmp_path: Path) -> None:
    home = ensure_flwr_home(tmp_path / "flwr")

    assert home.exists()
    assert home.is_dir()
