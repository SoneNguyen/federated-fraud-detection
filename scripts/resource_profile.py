"""Local resource planning for Flower client processes."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceProfile:
    logical_cores: int
    max_active: int
    torch_threads: int
    batch_size: int
    num_workers: int
    device: str
    stagger_seconds: float


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return max(value, min_value)


def _default_device(num_clients: int) -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    return "cpu" if num_clients >= 25 else "auto"


def _default_max_active(num_clients: int, logical_cores: int) -> int:
    if os.environ.get("MAX_ACTIVE_CLIENTS"):
        return min(_env_int("MAX_ACTIVE_CLIENTS", num_clients), num_clients)
    if num_clients >= 50:
        return min(num_clients, max(20, min(32, logical_cores)))
    return num_clients


def _default_batch_size(num_clients: int, device: str) -> int:
    if os.environ.get("BATCH_SIZE"):
        return _env_int("BATCH_SIZE", 512)
    if device == "cuda":
        return 2048
    if num_clients >= 50:
        return 512
    return 1024


def _default_num_workers(device: str, torch_threads: int) -> int:
    if os.environ.get("NUM_WORKERS"):
        return max(0, int(os.environ["NUM_WORKERS"]))
    if os.name == "nt":
        return 0
    if device == "cuda":
        return min(4, max(1, torch_threads))
    return 0


def _default_stagger(num_clients: int) -> float:
    if os.environ.get("CLIENT_STAGGER_SECONDS"):
        return float(os.environ["CLIENT_STAGGER_SECONDS"])
    if num_clients >= 50:
        return 0.15
    if num_clients >= 10:
        return 0.06
    return 0.15


def plan_resources(
    *,
    num_clients: int,
    requested_max_active: int = 0,
    requested_device: str | None = None,
    requested_stagger: float | None = None,
) -> ResourceProfile:
    logical_cores = max(os.cpu_count() or 1, 1)
    device = requested_device or _default_device(num_clients)
    max_active = (
        min(requested_max_active, num_clients)
        if requested_max_active > 0
        else _default_max_active(num_clients, logical_cores)
    )
    torch_threads = _env_int(
        "TORCH_NUM_THREADS",
        max(1, logical_cores // max(max_active, 1)),
    )
    batch_size = _default_batch_size(num_clients, device)
    num_workers = _default_num_workers(device, torch_threads)
    stagger_seconds = _default_stagger(num_clients) if requested_stagger is None else requested_stagger
    return ResourceProfile(
        logical_cores=logical_cores,
        max_active=max_active,
        torch_threads=torch_threads,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        stagger_seconds=stagger_seconds,
    )


def apply_client_resource_env(env: dict[str, str], profile: ResourceProfile) -> None:
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["DEVICE"] = profile.device
    env["BATCH_SIZE"] = str(profile.batch_size)
    env["NUM_WORKERS"] = str(profile.num_workers)
    env["TORCH_NUM_THREADS"] = str(profile.torch_threads)
    env["OMP_NUM_THREADS"] = str(profile.torch_threads)
    env["MKL_NUM_THREADS"] = str(profile.torch_threads)
    env["OPENBLAS_NUM_THREADS"] = str(profile.torch_threads)
    env["NUMEXPR_NUM_THREADS"] = str(profile.torch_threads)
    env.setdefault("MATMUL_PRECISION", "high")
