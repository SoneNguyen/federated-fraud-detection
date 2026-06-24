"""Start SuperLink, SuperNodes, and submit one local Flower run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.resource_profile import plan_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Flower training end to end.")
    parser.add_argument("--num-clients", type=int, default=int(os.environ.get("NUM_CLIENTS", "10")))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("NUM_ROUNDS", "100")))
    parser.add_argument("--model-run", default=os.environ.get("MODEL_RUN", ""))
    parser.add_argument("--data-root", default=os.environ.get("PROCESSED_DATA_ROOT", "dataset/processed"))
    parser.add_argument("--max-active", type=int, default=int(os.environ.get("MAX_ACTIVE_CLIENTS", "0")))
    parser.add_argument("--startup-seconds", type=float, default=8.0)
    return parser.parse_args()


def _terminate_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.terminate()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env["NUM_CLIENTS"] = str(args.num_clients)
    env["NUM_ROUNDS"] = str(args.rounds)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.model_run:
        env["MODEL_RUN"] = args.model_run
    profile = plan_resources(num_clients=args.num_clients, requested_max_active=args.max_active)
    env.setdefault("MAX_ACTIVE_CLIENTS", str(profile.max_active))

    data_root = Path(args.data_root)
    first_client = data_root / "client_0" / "transactions_normalized.parquet"
    if not first_client.exists():
        raise FileNotFoundError(
            f"Processed data not found at {first_client}. Run dataset/load_ieee_cis.py first."
        )

    server = subprocess.Popen([sys.executable, "-m", "scripts.run_server"], env=env)
    clients = None
    try:
        time.sleep(args.startup_seconds)
        client_cmd = [
            sys.executable,
            "-m",
            "scripts.launch_clients",
            "--num-clients",
            str(args.num_clients),
            "--data-root",
            args.data_root,
            "--max-active",
            str(profile.max_active),
        ]
        clients = subprocess.Popen(client_cmd, env=env)

        time.sleep(args.startup_seconds)
        submit_cmd = [
            sys.executable,
            "-m",
            "scripts.submit_flower_run",
            "--num-clients",
            str(args.num_clients),
            "--rounds",
            str(args.rounds),
        ]
        if args.model_run:
            submit_cmd.extend(["--model-run", args.model_run])
        code = subprocess.call(submit_cmd, env=env)
        raise SystemExit(code)
    finally:
        if clients is not None:
            _terminate_tree(clients)
        _terminate_tree(server)


if __name__ == "__main__":
    main()
