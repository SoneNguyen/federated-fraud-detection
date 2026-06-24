"""Single-process virtual federated training for large client-count experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.resource_profile import plan_resources
from scripts.run_paths import checkpoint_dir as default_checkpoint_dir
from scripts.run_paths import results_dir as default_results_dir
from src.server.runtime import lr_schedule
from src.client.client import FraudClient
from src.data.dataset import loader_kwargs, split_dataset
from src.model.fraud_mlp import FraudMLP, federated_params, is_federated_param
from src.server.aggregation import (
    robust_blended_average_ndarrays,
    stabilize_aggregate_update,
    target_aware_fedavg_weights,
    target_score,
)
from src.server.checkpoint_manager import CheckpointManager
from src.server.training_schedule import adapt_client_fit_config


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _archive_run_dir(path: Path, label: str) -> Path:
    archive = (
        Path("outputs/archive")
        / f"{label}_{path.name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(archive))
    return archive


def _client_path(data_root: Path, cid: int) -> Path:
    return data_root / f"client_{cid}" / "transactions_normalized.parquet"


def _select_clients(round_no: int, num_clients: int, sample_size: int) -> list[int]:
    start = ((round_no - 1) * sample_size) % num_clients
    return [(start + offset) % num_clients for offset in range(sample_size)]


def _state_from_arrays(arrays: list[np.ndarray]) -> dict[str, torch.Tensor]:
    model = FraudMLP(device="cpu")
    state = model.state_dict()
    keys = [key for key in state if is_federated_param(key)]
    for key, array in zip(keys, arrays):
        state[key] = torch.tensor(array)
    return state


def _make_client(cid: int, data_path: Path, global_arrays: list[np.ndarray], args: argparse.Namespace) -> FraudClient:
    model = FraudMLP(device=args.device)
    state = _state_from_arrays(global_arrays)
    model.load_state_dict(state, strict=True)
    train_dataset, val_dataset = split_dataset(str(data_path), val_split=args.val_split)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs(
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False,
        ),
    )
    os.environ["CLIENT_ID"] = str(cid)
    return FraudClient(
        model=model,
        train_dataset=train_dataset,
        val_loader=val_loader,
        local_epochs=args.local_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )


def _aggregate_eval(eval_results: list[tuple[int, float, int, dict[str, Any]]]) -> dict[str, Any]:
    total = sum(examples for _, _, examples, _ in eval_results)
    out: dict[str, Any] = {
        "total_examples": total,
        "val_loss": sum(loss * examples for _, loss, examples, _ in eval_results) / max(total, 1),
    }
    metric_names = {
        name
        for _, _, _, metrics in eval_results
        for name, value in metrics.items()
        if name != "client_id" and isinstance(value, (int, float, bool))
    }
    for name in metric_names:
        out[name] = (
            sum(float(metrics.get(name, 0.0)) * examples for _, _, examples, metrics in eval_results)
            / max(total, 1)
        )

    for metric in ("auprc", "auroc", "f1"):
        values = np.array(
            [float(metrics.get(f"val_{metric}", 0.0)) for _, _, _, metrics in eval_results],
            dtype=np.float64,
        )
        out[f"min_client_{metric}"] = float(np.min(values)) if len(values) else 0.0
        out[f"max_client_{metric}"] = float(np.max(values)) if len(values) else 0.0
        out[f"std_client_{metric}"] = float(np.std(values)) if len(values) else 0.0
    return out


def _status(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    auprc = float(metrics.get("val_auprc", 0.0))
    auroc = float(metrics.get("val_auroc", 0.0))
    f1 = float(metrics.get("val_f1", 0.0))
    score = target_score(
        auprc=auprc,
        auroc=auroc,
        f1=f1,
        target_auprc=args.target_auprc,
        target_auroc=args.target_auroc,
        target_f1=args.target_f1,
    )
    return {
        "target_score": score,
        "target_met": auprc >= args.target_auprc and auroc >= args.target_auroc and f1 >= args.target_f1,
        "margin_auprc": auprc - args.target_auprc,
        "margin_auroc": auroc - args.target_auroc,
        "margin_f1": f1 - args.target_f1,
    }


def _write_summary(results_dir: Path, history: list[dict[str, Any]]) -> None:
    ranked = sorted(
        history,
        key=lambda row: (
            bool(row.get("target_met", False)),
            float(row.get("target_score", 0.0)),
            float(row.get("val_auprc", 0.0)),
            float(row.get("val_f1", 0.0)),
        ),
        reverse=True,
    )
    latest = history[-1] if history else {}
    best = ranked[0] if ranked else {}
    lines = [
        "# Virtual Federated Training Summary",
        "",
        "| Item | Round | Checkpoint | Loss | AUPRC | AUROC | F1 | Target |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Latest", latest), ("Best", best)):
        if not row:
            continue
        round_no = int(row.get("round", 0))
        lines.append(
            "| {label} | {round_no} | round_{round_no:03d}.pt | {loss:.6f} | {auprc:.4f} | {auroc:.4f} | {f1:.4f} | {target} |".format(
                label=label,
                round_no=round_no,
                loss=float(row.get("val_loss", 0.0)),
                auprc=float(row.get("val_auprc", 0.0)),
                auroc=float(row.get("val_auroc", 0.0)),
                f1=float(row.get("val_f1", 0.0)),
                target=int(bool(row.get("target_met", False))),
            )
        )
    (results_dir / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run virtual FL without spawning one OS process per client.")
    parser.add_argument("--num-clients", type=int, default=int(os.environ.get("NUM_CLIENTS", "100")))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("NUM_ROUNDS", "100")))
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--data-root", default=os.environ.get("PROCESSED_DATA_ROOT", "dataset/processed"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--server-lr", type=float, default=0.65)
    parser.add_argument("--max-update-ratio", type=float, default=0.035)
    parser.add_argument("--robust-trim-ratio", type=float, default=0.10)
    parser.add_argument("--robust-median-blend", type=float, default=0.25)
    parser.add_argument("--keep-last-rounds", type=int, default=int(os.environ.get("KEEP_LAST_ROUNDS", "10000")))
    parser.add_argument("--fresh", action="store_true", default=_env_bool("FRESH_RUN", True))
    parser.add_argument("--resume", dest="fresh", action="store_false")
    parser.add_argument("--target-auprc", type=float, default=0.70)
    parser.add_argument("--target-auroc", type=float, default=0.90)
    parser.add_argument("--target-f1", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["NUM_CLIENTS"] = str(args.num_clients)
    os.environ.setdefault("TRAINING_PROFILE", "scalable")
    profile = plan_resources(
        num_clients=args.num_clients,
        requested_max_active=1,
        requested_device=args.device,
    )
    torch.set_num_threads(profile.torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(profile.torch_threads, 4)))
    except RuntimeError:
        pass

    data_root = Path(args.data_root)
    missing = [_client_path(data_root, cid) for cid in range(args.num_clients) if not _client_path(data_root, cid).exists()]
    if missing:
        raise FileNotFoundError(f"Missing client data: {missing[0]}")

    checkpoint_dir = default_checkpoint_dir()
    results_dir = default_results_dir()
    if args.fresh:
        for path in (checkpoint_dir, results_dir):
            if path.exists() and any(path.iterdir()):
                archive = _archive_run_dir(path, path.parent.name)
                print(f"VFL archive {path} -> {archive}")
            path.mkdir(parents=True, exist_ok=True)

    ckpt = CheckpointManager(checkpoint_dir)
    sample_size = args.sample_size or min(50, max(30, int(np.ceil(args.num_clients * 0.30))))
    global_arrays = federated_params(FraudMLP(device=args.device))
    previous: list[np.ndarray] | None = None
    history: list[dict[str, Any]] = []
    best_target_score = 0.0

    (checkpoint_dir / "active_training_run.json").write_text(
        json.dumps(
            {
                "training_run_id": f"virtual_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "mode": "virtual_federated",
                "num_clients": args.num_clients,
                "sample_size": sample_size,
                "checkpoint_dir": str(checkpoint_dir),
                "results_dir": str(results_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for round_no in range(1, args.rounds + 1):
        cfg = lr_schedule(round_no)
        cfg, adaptive_server_lr, schedule_meta = adapt_client_fit_config(
            cfg,
            history,
            server_round=round_no,
            base_server_lr=args.server_lr,
            best_target_score=best_target_score,
            configured_clients=args.num_clients,
            enabled=True,
        )
        args.lr = float(cfg["lr"])
        args.local_epochs = int(cfg["local_epochs"])
        selected = _select_clients(round_no, args.num_clients, sample_size)
        print(
            f"VFL R{round_no:03d} clients={selected[:8]}... sample={len(selected)} "
            f"phase={cfg.get('adaptive_phase', 'base')} lr={args.lr:.1e} "
            f"ep={args.local_epochs} slr={adaptive_server_lr:.2f} threads={profile.torch_threads}"
        )

        fit_arrays: list[list[np.ndarray]] = []
        fit_metrics: list[dict[str, Any]] = []
        fit_examples: list[int] = []
        eval_results: list[tuple[int, float, int, dict[str, Any]]] = []
        for cid in selected:
            client = _make_client(cid, _client_path(data_root, cid), global_arrays, args)
            params, examples, metrics = client.fit(global_arrays, cfg)
            metrics = dict(metrics)
            metrics["client_id"] = cid
            fit_arrays.append(params)
            fit_examples.append(examples)
            fit_metrics.append(metrics)
            loss, eval_examples, eval_metrics = client.evaluate(params, {})
            eval_metrics = dict(eval_metrics)
            eval_metrics["client_id"] = cid
            eval_results.append((cid, loss, eval_examples, eval_metrics))

        weights = target_aware_fedavg_weights(
            client_metrics=fit_metrics,
            client_examples=fit_examples,
            target_auprc=args.target_auprc,
            target_auroc=args.target_auroc,
            target_f1=args.target_f1,
            fairness_weight=0.15,
            profile="scalable",
        )
        proposed, robust_meta = robust_blended_average_ndarrays(
            fit_arrays,
            weights,
            trim_ratio=args.robust_trim_ratio,
            median_blend=args.robust_median_blend,
        )
        global_arrays, stability = stabilize_aggregate_update(
            previous=previous,
            proposed=proposed,
            server_lr=adaptive_server_lr,
            max_update_ratio=args.max_update_ratio,
        )
        stability.update(robust_meta)
        stability.update({k: v for k, v in schedule_meta.items() if isinstance(v, (int, float))})
        previous = [np.array(array, copy=True) for array in global_arrays]
        state = _state_from_arrays(global_arrays)
        ckpt.save(
            f"round_{round_no:03d}",
            state,
            {
                "round": round_no,
                "mode": "virtual_federated",
                "num_clients": len(selected),
                "selected_clients": selected,
                "aggregation_weights": weights,
                "aggregation_stability": stability,
                "schedule": schedule_meta,
            },
        )
        ckpt.prune_round_checkpoints(args.keep_last_rounds)

        record = {
            "round": round_no,
            "selected_clients": selected,
            **_aggregate_eval(eval_results),
        }
        record.update(_status(record, args))
        best_target_score = max(best_target_score, float(record.get("target_score", 0.0)))
        history.append(record)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "evaluation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (results_dir / "latest_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        best = max(history, key=lambda row: (bool(row.get("target_met", False)), float(row.get("target_score", 0.0))))
        (results_dir / "best_round.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        _write_summary(results_dir, history)
        print(
            "VFL R{round:03d} loss={loss:.5f} auprc={auprc:.4f} auroc={auroc:.4f} f1={f1:.4f} target={target}".format(
                round=round_no,
                loss=float(record.get("val_loss", 0.0)),
                auprc=float(record.get("val_auprc", 0.0)),
                auroc=float(record.get("val_auroc", 0.0)),
                f1=float(record.get("val_f1", 0.0)),
                target=int(bool(record.get("target_met", False))),
            )
        )


if __name__ == "__main__":
    main()
