"""Submit the Flower app to a local SuperLink."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the Flower app run.")
    parser.add_argument("--connection", default=os.environ.get("FLOWER_CONNECTION", "local-deployment"))
    parser.add_argument("--control-address", default=os.environ.get("FLOWER_CONTROL_ADDRESS", "127.0.0.1:9093"))
    parser.add_argument("--num-clients", type=int, default=int(os.environ.get("NUM_CLIENTS", "3")))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("NUM_ROUNDS", "100")))
    parser.add_argument("--model-run", default=os.environ.get("MODEL_RUN", ""))
    parser.add_argument("--stream", action="store_true", default=True)
    return parser.parse_args()


def _write_local_flower_config(flwr_home: Path, connection: str, control_address: str) -> None:
    flwr_home.mkdir(parents=True, exist_ok=True)
    config_path = flwr_home / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[superlink]",
                f'default = "{connection}"',
                "",
                f"[superlink.{connection}]",
                f'address = "{control_address}"',
                "insecure = true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    flwr_home = Path(os.environ.get("FLWR_HOME", "outputs/runtime/flwr"))
    _write_local_flower_config(flwr_home, args.connection, args.control_address)

    run_config = [
        f"num-clients={args.num_clients}",
        f"num-server-rounds={args.rounds}",
    ]
    if args.model_run:
        run_config.append(f'model-run="{args.model_run}"')

    env = os.environ.copy()
    env["FLWR_HOME"] = str(flwr_home)
    cmd = [
        "flwr",
        "run",
        ".",
        args.connection,
        "--run-config",
        " ".join(run_config),
    ]
    if args.stream:
        cmd.append("--stream")

    print(
        "FLOWER run "
        f"connection={args.connection} control={args.control_address} "
        f"clients={args.num_clients} rounds={args.rounds} flwr_home={flwr_home}"
    )
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
