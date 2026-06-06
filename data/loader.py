"""Shared data loading utilities.

Other modules should call make_loaders() from client.dataset directly.
This module re-exports it for convenience and adds a helper for loading
a raw dataframe with metadata columns intact (used by the drift monitor).
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_raw(client_id: int, base: str = "data/raw") -> pd.DataFrame:
    """Load the raw (pre-normalization) parquet for a given client."""
    path = Path(base) / f"client_{client_id}" / "transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run data/generate_synthetic.py first."
        )
    return pd.read_parquet(path)


def load_processed(client_id: int, base: str = "data/processed") -> pd.DataFrame:
    """Load the normalized parquet for a given client (includes metadata columns)."""
    path = Path(base) / f"client_{client_id}" / "transactions_normalized.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run data/normalize.py first."
        )
    return pd.read_parquet(path)


def load_reference_window(
    client_id: int = 0,
    n: int = 10_000,
    base: str = "data/processed",
) -> pd.DataFrame:
    """Return the first n rows of a client's processed data as a drift reference window."""
    df = load_processed(client_id, base)
    return df.head(n).reset_index(drop=True)