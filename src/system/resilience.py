"""Portable runtime resilience helpers for local and team experiments.

Most cross-device failures in this project are caused by stale runtime files,
occupied ports, incompatible processed data, or missing local commands. This
module keeps those checks inside the application runtime instead of exposing
another user-facing script.
"""

from __future__ import annotations

import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class PortPair:
    fleet: str
    control: str


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def archive_path(path: Path, *, reason: str, archive_root: Path | None = None) -> Path:
    archive_root = archive_root or Path("outputs/archive/self_heal")
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    target = archive_root / f"{path.name}.{reason}.{stamp}"
    counter = 1
    while target.exists():
        target = archive_root / f"{path.name}.{reason}.{stamp}.{counter}"
        counter += 1
    shutil.move(str(path), str(target))
    return target


def fallback_database_path(path: Path, *, reason: str) -> Path:
    """Return a fresh sibling DB path when the requested DB cannot be repaired."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}.{reason}.{stamp}{path.suffix or '.db'}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.{reason}.{stamp}.{counter}{path.suffix or '.db'}"
        )
        counter += 1
    return candidate


def _sqlite_tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path, timeout=1.0)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def ensure_flower_database(path: str | Path | None, *, reset_bad: bool = True) -> Path | None:
    """Validate a persistent Flower DB or archive it if incompatible."""
    if path is None or str(path).strip() == "":
        return None
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return db_path
    if db_path.stat().st_size == 0:
        if reset_bad:
            archived = archive_path(db_path, reason="empty_flower_db")
            print(f"RESILIENCE archived empty Flower DB: {archived}")
        return db_path
    try:
        tables = _sqlite_tables(db_path)
    except sqlite3.Error as exc:
        if not reset_bad:
            raise RuntimeError(f"Flower DB is unreadable: {db_path} ({exc})") from exc
        try:
            archived = archive_path(db_path, reason="bad_flower_db")
            print(f"RESILIENCE archived unreadable Flower DB: {archived}")
            return db_path
        except PermissionError:
            replacement = fallback_database_path(db_path, reason="recovered")
            print(
                "RESILIENCE Flower DB is locked/unreadable; using fresh DB "
                f"{replacement}"
            )
            return replacement
    missing = sorted({"fab"} - tables)
    if missing:
        if not reset_bad:
            raise RuntimeError(
                f"Flower DB is incompatible: {db_path} missing tables {missing}. "
                "Delete it or unset FLOWER_SUPERLINK_DB."
            )
        try:
            archived = archive_path(db_path, reason="incompatible_flower_db")
            print(f"RESILIENCE archived incompatible Flower DB: {archived} missing={missing}")
        except PermissionError:
            replacement = fallback_database_path(db_path, reason="recovered")
            print(
                "RESILIENCE Flower DB is locked/incompatible; using fresh DB "
                f"{replacement} missing={missing}"
            )
            return replacement
    return db_path


def ensure_flwr_home(path: str | Path) -> Path:
    home = Path(path)
    home.mkdir(parents=True, exist_ok=True)
    return home


def parse_host_port(address: str) -> tuple[str, int]:
    if ":" not in address:
        raise ValueError(f"Address must include host:port, got {address!r}")
    host, raw_port = address.rsplit(":", 1)
    return host or "127.0.0.1", int(raw_port)


def format_host_port(host: str, port: int) -> str:
    return f"{host}:{int(port)}"


def port_is_free(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((probe_host, int(port))) != 0


def find_free_port(host: str, start_port: int) -> int:
    for port in range(int(start_port), int(start_port) + 500):
        if port_is_free(host, port):
            return port
    raise RuntimeError(f"No free port found near {host}:{start_port}")


def choose_superlink_ports(
    fleet_address: str,
    control_address: str,
    *,
    auto_ports: bool,
) -> PortPair:
    fleet_host, fleet_port = parse_host_port(fleet_address)
    control_host, control_port = parse_host_port(control_address)
    fleet_free = port_is_free(fleet_host, fleet_port)
    control_free = port_is_free(control_host, control_port)
    if fleet_free and control_free:
        return PortPair(fleet=fleet_address, control=control_address)
    if not auto_ports:
        busy = []
        if not fleet_free:
            busy.append(fleet_address)
        if not control_free:
            busy.append(control_address)
        raise RuntimeError(
            "Flower ports are already in use: "
            + ", ".join(busy)
            + ". Stop old Flower terminals or run the local orchestrator, "
            "which enables AUTO_PORTS by default."
        )
    new_fleet_port = find_free_port(fleet_host, fleet_port + 10)
    new_control_port = find_free_port(control_host, max(control_port + 10, new_fleet_port + 1))
    chosen = PortPair(
        fleet=format_host_port(fleet_host, new_fleet_port),
        control=format_host_port(control_host, new_control_port),
    )
    print(f"RESILIENCE ports busy; using fleet={chosen.fleet} control={chosen.control}")
    return chosen


def require_commands(commands: list[str]) -> None:
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime command(s): "
            + ", ".join(missing)
            + ". Run `uv sync`, then start through `uv run python -m ...`."
        )


def maybe_prepare_processed_data(
    *,
    num_clients: int,
    data_root: Path,
    auto_prepare: bool,
) -> None:
    """Build processed IEEE-CIS data when missing or stale and raw data exists."""
    from src.data.dataset import validate_processed_schema

    first_client = data_root / "client_0" / "transactions_normalized.parquet"
    try:
        validate_processed_schema(first_client)
        return
    except Exception as exc:
        if not auto_prepare:
            raise
        raw_tx = Path("dataset/ieee_cis/train_transaction.csv")
        raw_id = Path("dataset/ieee_cis/train_identity.csv")
        if not raw_tx.exists() or not raw_id.exists():
            raise RuntimeError(
                f"Processed data is not ready at {first_client}, and raw IEEE-CIS "
                "files were not found. Place raw files under dataset/ieee_cis or "
                "run preprocessing manually."
            ) from exc
        print(f"RESILIENCE rebuilding processed data NUM_CLIENTS={num_clients}")
        env = os.environ.copy()
        env["NUM_CLIENTS"] = str(num_clients)
        subprocess.run([sys.executable, "dataset/load_ieee_cis.py"], env=env, check=True)
        validate_processed_schema(first_client)


def write_failure_report(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Runtime Failure Report",
                "",
                f"time_utc: {datetime.now(UTC).isoformat()}",
                "",
                message,
                "",
            ]
        ),
        encoding="utf-8",
    )
