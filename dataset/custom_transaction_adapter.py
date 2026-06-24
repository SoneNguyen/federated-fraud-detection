"""Map local transaction datasets into the active fraud feature pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


REQUIRED_FIELDS = ("transaction_time", "amount", "label")


def load_mapping(path: Path | str) -> dict[str, Any]:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = mapping.get("columns", {})
    missing = [name for name in REQUIRED_FIELDS if not columns.get(name)]
    if missing:
        raise ValueError(f"Mapping is missing required columns: {missing}")
    return mapping


def load_table(path: Path | str) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def to_ieee_like_frame(source: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    """Return columns expected by dataset.load_ieee_cis.engineer.

    The adapter intentionally uses deterministic defaults for fields that are
    unavailable in a partner dataset. The resulting frame is schema-compatible,
    while the preprocessing report still records the real dataset name/source.
    """
    columns = mapping.get("columns", {})
    out = pd.DataFrame(index=source.index)

    out["TransactionID"] = _mapped_series(source, columns, "transaction_id")
    if out["TransactionID"].isna().all():
        out["TransactionID"] = np.arange(1, len(source) + 1)

    out["TransactionDT"] = _transaction_dt(
        _required_series(source, columns, "transaction_time"),
        unit=str(mapping.get("time_unit", "auto")),
    )

    amount = pd.to_numeric(_required_series(source, columns, "amount"), errors="coerce").fillna(0)
    amount_to_usd_rate = float(mapping.get("amount_to_usd_rate", 1.0))
    out["TransactionAmt"] = (amount * amount_to_usd_rate).clip(0, 1e9).astype("float64")

    out["isFraud"] = _label_series(
        _required_series(source, columns, "label"),
        fraud_values=mapping.get("fraud_values", [1, "1", True, "true", "fraud"]),
    )

    out["ProductCD"] = _product_series(
        _mapped_series(source, columns, "product"),
        product_map=mapping.get("product_map", {}),
        default=str(mapping.get("default_product", "W")),
    )
    out["card6"] = _card_type_series(_mapped_series(source, columns, "card_type"))
    out["card4"] = _card_brand_series(_mapped_series(source, columns, "card_brand"))

    out["P_emaildomain"] = _email_domain(_mapped_series(source, columns, "payer_email"))
    out["R_emaildomain"] = _email_domain(_mapped_series(source, columns, "receiver_email"))

    account_key = _first_available(
        source,
        columns,
        ("card_id", "account_id", "customer_id", "payer_id"),
        default="unknown_account",
    )
    merchant_key = _first_available(
        source,
        columns,
        ("merchant_id", "receiver_id", "merchant_name"),
        default="unknown_merchant",
    )
    region_key = _first_available(
        source,
        columns,
        ("region", "province", "city", "addr1"),
        default="unknown_region",
    )
    device_key = _first_available(
        source,
        columns,
        ("device_id", "device_info", "ip_address"),
        default="unknown_device",
    )

    out["card1"] = _stable_bucket(account_key, modulo=20_000)
    out["card2"] = _stable_bucket(merchant_key, modulo=1_000)
    out["card3"] = _stable_bucket(region_key, modulo=200)
    out["card5"] = _stable_bucket(_card_brand_series(_mapped_series(source, columns, "card_brand")), modulo=500)
    out["addr1"] = _stable_bucket(region_key, modulo=1_000)
    out["addr2"] = _stable_bucket(region_key, modulo=100)
    out["DeviceInfo"] = device_key.astype(str)
    out["DeviceType"] = _device_type_series(_mapped_series(source, columns, "device_type"))

    out["C1"] = _numeric_or_window_count(source, columns, "transactions_1h", out, account_key, 3600)
    out["C2"] = _numeric_or_window_count(source, columns, "transactions_24h", out, account_key, 86400)
    out["C5"] = _numeric_or_default(source, columns, "chargeback_count", 0)
    out = _with_default_columns(out, [f"C{i}" for i in range(3, 15)], 0.0)

    out["D1"] = _numeric_or_days_since_previous(
        source,
        columns,
        "days_since_last_transaction",
        out,
        account_key,
    )
    out["D3"] = _numeric_or_account_age(source, columns, "account_age_days", out, account_key)
    out = _with_default_columns(out, [f"D{i}" for i in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], 0.0)

    distance_km = _numeric_or_default(source, columns, "distance_km", 0)
    out["dist1"] = (distance_km / 1.60934).astype("float64")
    out["dist2"] = distance_km.astype("float64")

    out = _with_default_columns(out, [f"id_{i:02d}" for i in range(1, 39)], np.nan)
    out = _with_default_columns(out, [f"V{i}" for i in range(1, 340)], 0.0)

    return out.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


def _with_default_columns(frame: pd.DataFrame, names: list[str], value: float) -> pd.DataFrame:
    missing = [name for name in names if name not in frame]
    if not missing:
        return frame
    defaults = pd.DataFrame(value, index=frame.index, columns=missing)
    return pd.concat([frame, defaults], axis=1)


def _mapped_series(source: pd.DataFrame, columns: dict[str, str], name: str) -> pd.Series:
    col = columns.get(name)
    if col and col in source:
        return source[col]
    return pd.Series(np.nan, index=source.index)


def _required_series(source: pd.DataFrame, columns: dict[str, str], name: str) -> pd.Series:
    col = columns.get(name)
    if not col or col not in source:
        raise ValueError(f"Required mapped column {name!r} not found: {col!r}")
    return source[col]


def _first_available(
    source: pd.DataFrame,
    columns: dict[str, str],
    names: tuple[str, ...],
    *,
    default: str,
) -> pd.Series:
    for name in names:
        series = _mapped_series(source, columns, name)
        if not series.isna().all():
            return series.fillna(default).astype(str)
    return pd.Series(default, index=source.index, dtype="object")


def _transaction_dt(values: pd.Series, *, unit: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if unit != "auto":
        dt = pd.to_datetime(cast(Any, values), unit=cast(Any, unit), utc=True, errors="coerce")
    elif numeric.notna().mean() > 0.95:
        max_value = numeric.max()
        if max_value > 1e12:
            dt = pd.to_datetime(cast(Any, numeric), unit="ms", utc=True, errors="coerce")
        elif max_value > 1e9:
            dt = pd.to_datetime(cast(Any, numeric), unit="s", utc=True, errors="coerce")
        else:
            seconds = (numeric - numeric.min()).fillna(0).clip(lower=0)
            return seconds.astype("float64")
    else:
        dt = pd.to_datetime(cast(Any, values), utc=True, errors="coerce")

    seconds = (dt - dt.min()).dt.total_seconds()
    return seconds.fillna(0).clip(lower=0).astype("float64")


def _label_series(values: pd.Series, fraud_values: list[Any]) -> pd.Series:
    fraud_set = {str(value).strip().lower() for value in fraud_values}
    normalized = values.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    labels = normalized.isin(fraud_set) | (numeric.fillna(0) > 0)
    return labels.astype("int8")


def _product_series(values: pd.Series, *, product_map: dict[str, str], default: str) -> pd.Series:
    allowed = {"W", "H", "C", "S", "R"}
    normalized_map = {str(k).lower(): str(v).upper() for k, v in product_map.items()}
    products = values.fillna(default).astype(str).str.strip().str.lower().map(normalized_map)
    products = products.fillna(default.upper())
    products = products.where(products.isin(allowed), default.upper())
    return products.astype(str)


def _card_type_series(values: pd.Series) -> pd.Series:
    normalized = values.fillna("debit").astype(str).str.lower().str.strip()
    mapped = normalized.replace(
        {
            "charge": "charge card",
            "charge_card": "charge card",
            "mixed": "debit or credit",
            "debit_credit": "debit or credit",
            "debit or credit": "debit or credit",
        }
    )
    return mapped.where(mapped.isin({"debit", "credit", "charge card", "debit or credit"}), "debit")


def _card_brand_series(values: pd.Series) -> pd.Series:
    normalized = values.fillna("unknown").astype(str).str.lower().str.strip()
    return normalized.replace({"master": "mastercard", "amex": "american express"})


def _device_type_series(values: pd.Series) -> pd.Series:
    normalized = values.fillna("unknown").astype(str).str.lower().str.strip()
    desktop = normalized.str.contains("desktop|pc|laptop", regex=True, na=False)
    mobile = normalized.str.contains("mobile|phone|android|ios", regex=True, na=False)
    return pd.Series(np.where(desktop, "desktop", np.where(mobile, "mobile", "unknown")), index=values.index)


def _email_domain(values: pd.Series) -> pd.Series:
    text = values.fillna("unknown").astype(str).str.lower().str.strip()
    return text.str.split("@").str[-1].replace("", "unknown")


def _stable_bucket(values: pd.Series, *, modulo: int) -> pd.Series:
    text = values.fillna("__missing__").astype(str).str.lower().str.strip()
    hashed = text.map(
        lambda value: int(
            hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest(),
            16,
        )
    )
    return (hashed.astype("uint64") % modulo).astype("float32")


def _numeric_or_default(
    source: pd.DataFrame,
    columns: dict[str, str],
    name: str,
    default: float,
) -> pd.Series:
    series = _mapped_series(source, columns, name)
    if series.isna().all():
        return pd.Series(default, index=source.index, dtype="float64")
    return pd.to_numeric(series, errors="coerce").fillna(default).astype("float64")


def _numeric_or_window_count(
    source: pd.DataFrame,
    columns: dict[str, str],
    name: str,
    out: pd.DataFrame,
    entity: pd.Series,
    window_seconds: int,
) -> pd.Series:
    series = _mapped_series(source, columns, name)
    if not series.isna().all():
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0).astype("float64")
    return _previous_window_count(out["TransactionDT"], entity, window_seconds)


def _previous_window_count(times: pd.Series, entity: pd.Series, window_seconds: int) -> pd.Series:
    result = pd.Series(0.0, index=times.index, dtype="float64")
    frame = pd.DataFrame({"time": times, "entity": entity.astype(str)})
    for _, group in frame.sort_values("time").groupby("entity", sort=False):
        idx = group.index.to_numpy()
        t = group["time"].to_numpy(dtype="float64")
        left = 0
        counts = np.zeros(len(group), dtype="float64")
        for pos, current in enumerate(t):
            while left < pos and current - t[left] > window_seconds:
                left += 1
            counts[pos] = pos - left
        result.loc[idx] = counts
    return result


def _numeric_or_days_since_previous(
    source: pd.DataFrame,
    columns: dict[str, str],
    name: str,
    out: pd.DataFrame,
    entity: pd.Series,
) -> pd.Series:
    series = _mapped_series(source, columns, name)
    if not series.isna().all():
        return pd.to_numeric(series, errors="coerce").fillna(1).clip(lower=0.01).astype("float64")

    result = pd.Series(1.0, index=out.index, dtype="float64")
    frame = pd.DataFrame({"time": out["TransactionDT"], "entity": entity.astype(str)})
    for _, group in frame.sort_values("time").groupby("entity", sort=False):
        diff = group["time"].diff().fillna(86400.0) / 86400.0
        result.loc[group.index] = diff.clip(lower=0.01)
    return result


def _numeric_or_account_age(
    source: pd.DataFrame,
    columns: dict[str, str],
    name: str,
    out: pd.DataFrame,
    entity: pd.Series,
) -> pd.Series:
    series = _mapped_series(source, columns, name)
    if not series.isna().all():
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0).astype("float64")

    result = pd.Series(0.0, index=out.index, dtype="float64")
    frame = pd.DataFrame({"time": out["TransactionDT"], "entity": entity.astype(str)})
    for _, group in frame.sort_values("time").groupby("entity", sort=False):
        age = (group["time"] - group["time"].min()) / 86400.0
        result.loc[group.index] = age.clip(lower=0)
    return result
