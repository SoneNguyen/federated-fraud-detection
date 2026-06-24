"""Start one Flower SuperNode for the configured client partition."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start one Flower SuperNode.")
    parser.add_argument("--client-id", type=int, default=int(os.environ.get("CLIENT_ID", "0")))
    parser.add_argument(
        "--data-path",
        default=os.environ.get("DATA_PATH", ""),
        help="Client parquet path. Defaults to dataset/processed/client_<id>/transactions_normalized.parquet.",
    )
    parser.add_argument(
        "--superlink",
        default=os.environ.get("FLOWER_SUPERLINK", "127.0.0.1:9092"),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--local-epochs", type=int, default=int(os.environ.get("LOCAL_EPOCHS", "2")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "512")))
    parser.add_argument(
        "--clientappio-port",
        type=int,
        default=0,
        help="0 picks 9094 + client id for local multi-node runs.",
    )
    parser.add_argument("--secure", action="store_true", help="Do not pass --insecure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path) if args.data_path else (
        Path("dataset/processed") / f"client_{args.client_id}" / "transactions_normalized.parquet"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Client data not found: {data_path}")

    appio_port = args.clientappio_port or (9094 + args.client_id)
    node_config = (
        f"client-id={args.client_id} "
        f"partition-id={args.client_id} "
        f'data-path="{data_path.as_posix()}" '
        f'device="{args.device}" '
        f"local-epochs={args.local_epochs} "
        f"batch-size={args.batch_size}"
    )
    cmd = [
        "flower-supernode",
        "--superlink",
        args.superlink,
        "--clientappio-api-address",
        f"127.0.0.1:{appio_port}",
        "--node-config",
        node_config,
    ]
    if not args.secure:
        cmd.append("--insecure")

    print(
        "SUPERNODE start "
        f"client={args.client_id} superlink={args.superlink} appio=127.0.0.1:{appio_port} "
        f"data={data_path}"
    )
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
