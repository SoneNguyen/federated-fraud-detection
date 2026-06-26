"""Launch many Flower clients from processed client parquet folders."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.resource_profile import apply_client_resource_env, plan_resources
from src.data.dataset import validate_processed_schema
from src.system.resilience import maybe_prepare_processed_data, write_failure_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start multiple federated clients for scalability experiments."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=int(os.environ.get("NUM_CLIENTS", "3")),
        help="Number of client_N folders to launch.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PROCESSED_DATA_ROOT", "dataset/processed"),
        help="Root containing client_N/transactions_normalized.parquet folders.",
    )
    parser.add_argument(
        "--superlink",
        default=os.environ.get("FLOWER_SUPERLINK", "127.0.0.1:9092"),
        help="Flower SuperLink Fleet API address.",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=None,
        help="Delay between client starts to avoid a connection burst.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Client device. Defaults to CPU for >=25 clients to avoid GPU contention.",
    )
    parser.add_argument(
        "--max-active",
        type=int,
        default=0,
        help="Maximum live client processes. 0 uses a safe local default.",
    )
    parser.add_argument(
        "--restart-limit",
        type=int,
        default=int(os.environ.get("CLIENT_RESTART_LIMIT", "2")),
        help="Restart a failed client at most this many times.",
    )
    parser.add_argument(
        "--monitor-seconds",
        type=float,
        default=float(os.environ.get("CLIENT_MONITOR_SECONDS", "2.0")),
        help="Delay between process health checks.",
    )
    parser.add_argument(
        "--no-auto-prepare-data",
        action="store_true",
        help="Fail instead of rebuilding processed IEEE-CIS data when raw files are available.",
    )
    return parser.parse_args()


def client_data_path(data_root: Path, client_id: int) -> Path:
    return data_root / f"client_{client_id}" / "transactions_normalized.parquet"


def main() -> None:
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("--num-clients must be >= 1")
    if args.max_active < 0:
        raise ValueError("--max-active must be >= 0")
    if args.restart_limit < 0:
        raise ValueError("--restart-limit must be >= 0")

    try:
        data_root = Path(args.data_root)
        maybe_prepare_processed_data(
            num_clients=args.num_clients,
            data_root=data_root,
            auto_prepare=not args.no_auto_prepare_data,
        )
        missing = [
            client_data_path(data_root, cid)
            for cid in range(args.num_clients)
            if not client_data_path(data_root, cid).exists()
        ]
        if missing:
            joined = "\n".join(str(path) for path in missing[:10])
            extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
            raise FileNotFoundError(
                "Missing processed client data. Re-run preprocessing with matching "
                f"NUM_CLIENTS={args.num_clients}.\n{joined}{extra}"
            )
        for cid in range(args.num_clients):
            validate_processed_schema(client_data_path(data_root, cid))
    except Exception as exc:
        write_failure_report(Path("outputs/runtime/last_failure.md"), str(exc))
        raise

    profile = plan_resources(
        num_clients=args.num_clients,
        requested_max_active=args.max_active,
        requested_device=args.device,
        requested_stagger=args.stagger_seconds,
    )
    pending = list(range(args.num_clients))
    processes: dict[int, subprocess.Popen] = {}
    restart_counts = {cid: 0 for cid in range(args.num_clients)}

    print(
        f"CLIENTS launch count={args.num_clients} "
        f"active={profile.max_active} cores={profile.logical_cores} "
        f"threads={profile.torch_threads}/client bs={profile.batch_size} "
        f"workers={profile.num_workers} device={profile.device} "
        f"restarts={args.restart_limit} stagger={profile.stagger_seconds:.2f}s "
        f"superlink={args.superlink} data_root={data_root}"
    )

    def start_one(cid: int) -> None:
        env = os.environ.copy()
        env["CLIENT_ID"] = str(cid)
        env["DATA_PATH"] = str(client_data_path(data_root, cid))
        env["FLOWER_SUPERLINK"] = args.superlink
        apply_client_resource_env(env, profile)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.run_client",
                "--client-id",
                str(cid),
                "--data-path",
                str(client_data_path(data_root, cid)),
                "--superlink",
                args.superlink,
                "--device",
                profile.device,
                "--batch-size",
                str(profile.batch_size),
            ],
            env=env,
        )
        processes[cid] = proc
        print(f"CLIENTS started id={cid} pid={proc.pid}")

    try:
        while pending or processes:
            while pending and len(processes) < profile.max_active:
                start_one(pending.pop(0))
                time.sleep(max(profile.stagger_seconds, 0.0))

            exited: list[tuple[int, int]] = []
            for cid, proc in list(processes.items()):
                code = proc.poll()
                if code is not None:
                    exited.append((cid, code))
                    del processes[cid]

            for cid, code in exited:
                print(f"CLIENTS exited id={cid} code={code} live={len(processes)} pending={len(pending)}")
                if code != 0 and restart_counts[cid] < args.restart_limit:
                    restart_counts[cid] += 1
                    pending.append(cid)
                    print(f"CLIENTS restart id={cid} attempt={restart_counts[cid]}")

            if not pending and not processes:
                break
            time.sleep(max(args.monitor_seconds, 0.5))
    except KeyboardInterrupt:
        print("\nCLIENTS stopping")
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2.0)
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
