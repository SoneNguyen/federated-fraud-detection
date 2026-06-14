"""Shared mutable state for per-client AUPRC tracking.

Lives here (not in fl_server.py or strategy.py) to avoid circular imports.
"""
from collections import deque

# Keys are CLIENT_IDs (int); values are rolling 5-round AUPRC deques.
client_auprc_history: dict[int, deque] = {}


def record_auprc(client_id: int, auprc: float, maxlen: int = 5) -> None:
    """Record AUPRC for a client."""
    if client_id not in client_auprc_history:
        client_auprc_history[client_id] = deque(maxlen=maxlen)
    client_auprc_history[client_id].append(auprc)


def alpha_for_client(client_id: int) -> float:
    """
    Compute focal_alpha for a client based on its recent AUPRC trend.

    Logic:
    - If the client has no history yet, return the neutral default (0.75).
    - If the client's recent mean AUPRC is below 0.50 (clearly struggling),
      raise alpha toward 0.85 to increase positive-class weight.
    - If between 0.50–0.58 (below the good-client band), nudge to 0.80.
    - Otherwise leave at 0.75 — don't over-correct healthy clients.
    """
    history = client_auprc_history.get(client_id)
    if not history:
        return 0.75
    mean_auprc = sum(history) / len(history)
    if mean_auprc < 0.50:
        return 0.85
    if mean_auprc < 0.58:
        return 0.80
    return 0.75
