"""Server-side aggregation algorithms for federated fraud training."""

from __future__ import annotations

from math import sqrt

import numpy as np


def target_score(
    *,
    auprc: float,
    auroc: float,
    f1: float,
    target_auprc: float,
    target_auroc: float,
    target_f1: float,
) -> float:
    """Return capped progress toward the deployment metrics target."""
    ratios = [
        min(float(auprc) / target_auprc, 1.0),
        min(float(auroc) / target_auroc, 1.0),
        min(float(f1) / target_f1, 1.0),
    ]
    return float(0.35 * ratios[0] + 0.20 * ratios[1] + 0.45 * ratios[2])


def target_aware_fedavg_weights(
    *,
    client_metrics: list[dict],
    client_examples: list[int],
    target_auprc: float,
    target_auroc: float,
    target_f1: float,
    fairness_weight: float,
    profile: str = "ambitious",
) -> list[float]:
    """Return normalized target-aware FedAvg weights for responding clients.

    The algorithm keeps FedAvg's sample-count weighting, but uses sqrt(n) to
    reduce dominance by one large client and adds a quality multiplier for
    clients that are below the absolute target.
    """
    if len(client_metrics) != len(client_examples):
        raise ValueError("client_metrics and client_examples must have the same length")

    profile_name = profile.strip().lower()
    ambitious = profile_name == "ambitious"
    scalable = profile_name in {"scalable", "scale", "large"}
    raw_weights: list[float] = []
    for metrics, num_examples in zip(client_metrics, client_examples):
        score = target_score(
            auprc=float(metrics.get("val_auprc", 0.0)),
            auroc=float(metrics.get("val_auroc", 0.0)),
            f1=float(metrics.get("val_f1", 0.0)),
            target_auprc=target_auprc,
            target_auroc=target_auroc,
            target_f1=target_f1,
        )
        capped = min(max(score, 0.0), 1.0)
        if scalable:
            quality = 0.70 + 0.30 * capped
        elif ambitious:
            quality = 1.0 + fairness_weight * (1.0 - capped)
        else:
            quality = 0.80 + 0.20 * capped
        raw_weights.append(quality * sqrt(max(int(num_examples), 0)))

    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(raw_weights) for _ in raw_weights] if raw_weights else []
    return [float(weight / total) for weight in raw_weights]


def stabilize_aggregate_update(
    *,
    previous: list[np.ndarray] | None,
    proposed: list[np.ndarray],
    server_lr: float,
    max_update_ratio: float,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Damp and clip the server update for large/noisy client populations."""
    if previous is None or len(previous) != len(proposed):
        return proposed, {
            "server_update_norm": 0.0,
            "server_param_norm": 0.0,
            "server_update_scale": 1.0,
            "server_lr": 1.0,
        }

    lr = min(max(float(server_lr), 0.0), 1.0)
    ratio = max(float(max_update_ratio), 0.0)
    deltas = [new - old for old, new in zip(previous, proposed)]
    update_norm = float(np.sqrt(sum(float(np.sum(delta.astype(np.float64) ** 2)) for delta in deltas)))
    param_norm = float(np.sqrt(sum(float(np.sum(old.astype(np.float64) ** 2)) for old in previous)))
    max_norm = ratio * max(param_norm, 1.0)
    scale = 1.0
    if ratio > 0.0 and update_norm > max_norm:
        scale = max_norm / max(update_norm, 1e-12)

    stabilized = [
        old + (delta * scale * lr)
        for old, delta in zip(previous, deltas)
    ]
    return stabilized, {
        "server_update_norm": update_norm,
        "server_param_norm": param_norm,
        "server_update_scale": scale,
        "server_lr": lr,
    }


def weighted_average_ndarrays(
    client_parameters: list[list[np.ndarray]],
    weights: list[float],
) -> list[np.ndarray]:
    """Average matching client parameter arrays using normalized weights."""
    if len(client_parameters) != len(weights):
        raise ValueError("client_parameters and weights must have the same length")
    if not client_parameters:
        return []

    return [
        sum(
            (client[i] * weight for client, weight in zip(client_parameters, weights)),
            np.zeros_like(client_parameters[0][i]),
        )
        for i in range(len(client_parameters[0]))
    ]


def robust_blended_average_ndarrays(
    client_parameters: list[list[np.ndarray]],
    weights: list[float],
    *,
    trim_ratio: float = 0.0,
    median_blend: float = 0.0,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Blend weighted FedAvg with a robust coordinate statistic.

    The weighted average remains the main optimizer. For larger federations,
    a trimmed mean or coordinate median reduces the impact of outlier updates
    caused by non-IID partitions, stale clients, or partial training failures.
    """
    weighted = weighted_average_ndarrays(client_parameters, weights)
    n_clients = len(client_parameters)
    trim = min(max(float(trim_ratio), 0.0), 0.49)
    blend = min(max(float(median_blend), 0.0), 1.0)
    if n_clients < 3 or (trim <= 0.0 and blend <= 0.0):
        return weighted, {
            "robust_clients": float(n_clients),
            "robust_trim_ratio": 0.0,
            "robust_median_blend": 0.0,
        }

    robust_params: list[np.ndarray] = []
    for tensor_idx in range(len(client_parameters[0])):
        stack = np.stack(
            [client[tensor_idx].astype(np.float64, copy=False) for client in client_parameters],
            axis=0,
        )
        trim_count = int(n_clients * trim)
        if trim_count > 0 and (2 * trim_count) < n_clients:
            sorted_stack = np.sort(stack, axis=0)
            robust = sorted_stack[trim_count : n_clients - trim_count].mean(axis=0)
        else:
            robust = np.median(stack, axis=0)
        robust_params.append(robust.astype(weighted[tensor_idx].dtype, copy=False))

    blended = [
        ((1.0 - blend) * avg + blend * robust).astype(avg.dtype, copy=False)
        for avg, robust in zip(weighted, robust_params)
    ]
    return blended, {
        "robust_clients": float(n_clients),
        "robust_trim_ratio": trim,
        "robust_median_blend": blend,
    }
