"""Flower ClientApp for fraud detection clients."""

from __future__ import annotations

import os
from typing import Any

from flwr.app import Context
from flwr.clientapp import ClientApp

from src.client.factory import build_fraud_client


def _value(config: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    value = config.get(key)
    if value is not None:
        return value
    return os.environ.get(env_name, default)


def client_fn(context: Context):
    node_config = dict(context.node_config)
    client_id = int(
        _value(
            node_config,
            "client-id",
            "CLIENT_ID",
            node_config.get("partition-id", 0),
        )
    )
    data_path = str(
        _value(
            node_config,
            "data-path",
            "DATA_PATH",
            f"dataset/processed/client_{client_id}/transactions_normalized.parquet",
        )
    )
    device = str(_value(node_config, "device", "DEVICE", "auto"))
    local_epochs = int(_value(node_config, "local-epochs", "LOCAL_EPOCHS", 2))
    batch_size = int(_value(node_config, "batch-size", "BATCH_SIZE", 512))

    return build_fraud_client(
        client_id=client_id,
        data_path=data_path,
        device_str=device,
        local_epochs=local_epochs,
        batch_size=batch_size,
    ).to_client()


app = ClientApp(client_fn=client_fn)

