#!/usr/bin/env python3
"""
Run all 3 federated clients simultaneously.
Usage: uv run python run_all_clients.py
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import signal

PROJECT_ROOT = Path(__file__).parent
DATA_BASE_PATH = PROJECT_ROOT / "data" / "processed"

def main():
    print("=" * 50)
    print("Federated Learning - Launching 3 Clients")
    print("=" * 50)
    print()
    
    # Verify data exists
    if not DATA_BASE_PATH.exists():
        print(f"Error: Data directory not found at {DATA_BASE_PATH}")
        print("Please run: uv run python data/load_ieee_cis.py")
        sys.exit(1)
    
    processes = []
    
    for client_id in range(3):
        data_path = DATA_BASE_PATH / f"client_{client_id}" / "transactions_normalized.parquet"
        
        # Verify client data exists
        if not data_path.exists():
            print(f"Error: Client {client_id} data not found at {data_path}")
            sys.exit(1)
        
        print(f"[Client {client_id}] Starting...")
        print(f"  Data: {data_path}")
        print()
        
        # Set environment variables
        env = os.environ.copy()
        env["CLIENT_ID"] = str(client_id)
        env["DATA_PATH"] = str(data_path)
        env["SERVER_ADDRESS"] = "localhost:8080"
        env["LOCAL_EPOCHS"] = "10"
        
        # Start client process
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "client.run_client"],
                env=env,
                cwd=PROJECT_ROOT,
            )
            processes.append((client_id, proc))
        except Exception as e:
            print(f"[Client {client_id}] Failed to start: {e}")
            sys.exit(1)
    
    print("=" * 50)
    print("All 3 clients launched")
    print("=" * 50)
    print()
    print("Press Ctrl+C to stop all clients")
    print()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nShutting down clients...")
        for client_id, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"[Client {client_id}] Terminated")
            except Exception as e:
                proc.kill()
                print(f"[Client {client_id}] Force killed: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Monitor all processes
    try:
        while all(proc.poll() is None for _, proc in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    # Print final status
    print()
    print("=" * 50)
    print("Client Status:")
    print("=" * 50)
    
    for client_id, proc in processes:
        return_code = proc.poll()
        if return_code is None:
            status = "Running"
        elif return_code == 0:
            status = "[OK] Completed"
        else:
            status = f"[FAIL] Exit code {return_code}"
        print(f"[Client {client_id}] {status}")

if __name__ == "__main__":
    main()
