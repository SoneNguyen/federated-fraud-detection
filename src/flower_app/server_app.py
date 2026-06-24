"""Flower ServerApp for target-aware federated fraud training."""

from __future__ import annotations

from flwr.app import Context
from flwr.serverapp import ServerApp

from src.server.runtime import build_server_components


def server_fn(context: Context):
    return build_server_components(context.run_config)


app = ServerApp(server_fn=server_fn)

