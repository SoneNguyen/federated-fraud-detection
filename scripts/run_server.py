"""Start the Flower SuperLink infrastructure process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.system.resilience import (
    choose_superlink_ports,
    ensure_flower_database,
    env_flag,
    require_commands,
    write_failure_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Flower SuperLink.")
    parser.add_argument(
        "--fleet-api-address",
        default=os.environ.get("FLOWER_FLEET_API_ADDRESS", "0.0.0.0:9092"),
    )
    parser.add_argument(
        "--control-api-address",
        default=os.environ.get("FLOWER_CONTROL_API_ADDRESS", "0.0.0.0:9093"),
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("FLOWER_SUPERLINK_DB", ""),
        help=(
            "Optional SuperLink database path. Leave empty for in-memory local "
            "state, which avoids SQLite lock issues during repeated experiments."
        ),
    )
    parser.add_argument("--secure", action="store_true", help="Do not pass --insecure.")
    parser.add_argument(
        "--auto-ports",
        action="store_true",
        default=env_flag("AUTO_PORTS", False),
        help="Pick nearby free ports if the requested Flower ports are busy.",
    )
    parser.add_argument(
        "--no-runtime-reset",
        action="store_true",
        help="Fail instead of archiving incompatible Flower runtime databases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        require_commands(["flower-superlink"])
        ports = choose_superlink_ports(
            args.fleet_api_address,
            args.control_api_address,
            auto_ports=bool(args.auto_ports),
        )
    except Exception as exc:
        write_failure_report(Path("outputs/runtime/last_failure.md"), str(exc))
        raise

    cmd = [
        "flower-superlink",
        "--fleet-api-address",
        ports.fleet,
        "--control-api-address",
        ports.control,
    ]
    database_text = str(args.database).strip()
    database_label = "memory"
    if database_text:
        database = ensure_flower_database(
            database_text,
            reset_bad=not bool(args.no_runtime_reset),
        )
        assert database is not None
        cmd.extend(["--database", str(database)])
        database_label = str(database)
    if not args.secure:
        cmd.append("--insecure")

    print(
        "SUPERLINK start "
        f"fleet={ports.fleet} control={ports.control} db={database_label}"
    )
    print(
        "SUPERLINK ready_when_logs_show_apis. Keep this terminal open, then start "
        "SuperNodes and submit a run with scripts.submit_flower_run."
    )
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
