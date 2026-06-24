"""Start the Flower SuperLink infrastructure process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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
        default=os.environ.get("FLOWER_SUPERLINK_DB", "outputs/runtime/flwr/superlink.db"),
    )
    parser.add_argument("--secure", action="store_true", help="Do not pass --insecure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "flower-superlink",
        "--fleet-api-address",
        args.fleet_api_address,
        "--control-api-address",
        args.control_api_address,
        "--database",
        str(database),
    ]
    if not args.secure:
        cmd.append("--insecure")

    print(
        "SUPERLINK start "
        f"fleet={args.fleet_api_address} control={args.control_api_address} db={database}"
    )
    print(
        "SUPERLINK ready_when_logs_show_apis. Keep this terminal open, then start "
        "SuperNodes and submit a run with scripts.submit_flower_run."
    )
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
