"""Adaptive client training schedule shared by Flower and virtual FL runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, bool)):
        return float(value)
    return default


def _slope(history: list[Mapping[str, object]], key: str, window: int) -> float:
    values = [
        _as_float(record[key])
        for record in history[-max(window, 2):]
        if key in record and np.isfinite(_as_float(record[key], float("nan")))
    ]
    if len(values) < 2:
        return 0.0
    return float((values[-1] - values[0]) / (len(values) - 1))


def adapt_client_fit_config(
    base_config: Mapping[str, Any],
    history: list[Mapping[str, object]],
    *,
    server_round: int,
    base_server_lr: float,
    best_target_score: float,
    configured_clients: int,
    stall_window: int = 5,
    enabled: bool = True,
) -> tuple[dict[str, Any], float, dict[str, float | str]]:
    """Return an adaptive fit config and server learning rate.

    Round-based schedules are useful early, but 100-client splits create noisy
    validation signals. This controller keeps the base schedule, then adjusts
    learning rate, local epochs, loss mix, and server update size from recent
    validation behavior.
    """
    cfg = dict(base_config)
    if not enabled or not history:
        cfg["adaptive_phase"] = "warmup"
        return cfg, float(base_server_lr), {
            "adaptive_lr_scale": 1.0,
            "adaptive_server_lr": float(base_server_lr),
            "adaptive_phase": "warmup",
        }

    latest = history[-1]
    state = str(latest.get("learning_state", "mixed"))
    target_score = _as_float(latest.get("target_score"), 0.0)
    target_met = bool(latest.get("target_met", False))
    high_target_met = bool(latest.get("high_target_met", False))
    score_slope = _slope(history, "target_score", stall_window)
    loss_slope = _slope(history, "val_loss", stall_window)
    f1_slope = _slope(history, "val_f1", stall_window)
    auprc_slope = _slope(history, "val_auprc", stall_window)

    lr_scale = 1.0
    server_lr_scale = 1.0
    epoch_delta = 0
    bce_delta = 0.0
    gamma_delta = 0.0
    fedprox_scale = 1.0
    phase = "base"

    score_drop = best_target_score > 0 and target_score < best_target_score - 0.015
    regressing = state == "regressing" or (score_drop and server_round > stall_window)
    stalled = state == "stalled" or (
        abs(score_slope) < 0.001 and abs(f1_slope) < 0.001 and abs(auprc_slope) < 0.001
    )

    if regressing:
        phase = "recovery"
        lr_scale = 0.45
        server_lr_scale = 0.70
        epoch_delta = -1
        bce_delta = 0.15
        gamma_delta = -0.25
        fedprox_scale = 1.5
    elif stalled and not target_met:
        phase = "plateau"
        lr_scale = 0.70
        server_lr_scale = 0.85
        epoch_delta = 0 if configured_clients >= 50 else 1
        bce_delta = 0.10
        gamma_delta = -0.15
        fedprox_scale = 1.25
    elif target_met and not high_target_met:
        phase = "refine"
        lr_scale = 0.65
        server_lr_scale = 0.80
        epoch_delta = 1 if configured_clients < 50 else 0
        bce_delta = 0.05
        gamma_delta = -0.10
    elif state == "learning" and loss_slope <= 0:
        phase = "learning"
        lr_scale = 1.05

    lr = max(float(cfg.get("lr", 1e-3)) * lr_scale, 1e-6)
    epochs = max(1, int(cfg.get("local_epochs", 1)) + epoch_delta)
    max_epochs = 3 if configured_clients >= 50 else 5
    bce_mix = min(max(float(cfg.get("bce_mix", 0.30)) + bce_delta, 0.05), 0.60)
    focal_gamma = min(max(float(cfg.get("focal_gamma", 1.75)) + gamma_delta, 1.0), 3.0)
    fedprox_mu = min(max(float(cfg.get("fedprox_mu", 0.001)) * fedprox_scale, 0.0), 0.01)
    server_lr = min(max(float(base_server_lr) * server_lr_scale, 0.05), 1.0)

    cfg.update(
        {
            "lr": lr,
            "local_epochs": min(epochs, max_epochs),
            "bce_mix": bce_mix,
            "focal_gamma": focal_gamma,
            "fedprox_mu": fedprox_mu,
            "adaptive_phase": phase,
            "adaptive_lr_scale": lr_scale,
            "adaptive_server_lr": server_lr,
        }
    )
    return cfg, server_lr, {
        "adaptive_lr_scale": lr_scale,
        "adaptive_server_lr": server_lr,
        "adaptive_phase": phase,
        "adaptive_score_slope": score_slope,
        "adaptive_loss_slope": loss_slope,
    }
