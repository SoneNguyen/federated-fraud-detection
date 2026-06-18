"""Prepare IEEE-CIS data for federated fraud training.

The fraud-history pipeline is deliberately more tabular-model friendly than the old
47-feature contract: it keeps the previous engineered features, adds C/D/id
signals, selected V columns, unsupervised frequency encodings, and temporal
entity-history features. Rows are sorted by TransactionDT before temporal
client splitting.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.feature_registry import (
    BINARY_FEATURES,
    D_BASE_COLUMNS,
    FEATURE_ORDER,
    ID_NUMERIC_COLUMNS,
    LABEL,
    SCHEMA_VERSION,
    SELECTED_V_COLUMNS,
)

__all__ = ["main"]

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RAW_DIR = Path("data/ieee_cis")
RAW_PROC = Path("data/processed")


def _log1p_col(df: pd.DataFrame, col: str, clip_hi: float) -> pd.Series:
    if col not in df:
        return pd.Series(0.0, index=df.index, dtype="float32")
    values = df[col].fillna(0).clip(0, clip_hi).astype("float64")
    return pd.Series(np.log1p(values.to_numpy()), index=df.index, dtype="float32")


def _missing_flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(1.0, index=df.index, dtype="float32")
    return df[col].isna().astype("float32")


def _freq(values: pd.Series) -> pd.Series:
    normalized = values.fillna("__missing__").astype(str).str.lower().str.strip()
    counts = normalized.value_counts(dropna=False)
    return normalized.map(counts).fillna(0).astype("float32")


def _key(values: pd.Series) -> pd.Series:
    normalized = values.fillna("__missing__").astype(str).str.lower().str.strip()
    return normalized.mask(normalized == "", "__missing__")


def _pair_key(left: pd.Series, right: pd.Series) -> pd.Series:
    return _key(left) + "|" + _key(right)


def _add_history_features(
    out: pd.DataFrame,
    dt: pd.Series,
    raw_amount: pd.Series,
    fraud_label: pd.Series,
    global_prior: pd.Series,
    key: pd.Series,
    prefix: str,
) -> None:
    """Add strictly backward-looking transaction history for one entity key."""
    entity = _key(key)
    counts = entity.groupby(entity, sort=False).cumcount().astype("float64")
    prev_dt = dt.groupby(entity, sort=False).shift(1)
    since_prev = (dt - prev_dt).fillna(0).clip(0, 180 * 86400)

    amount_cumsum = raw_amount.groupby(entity, sort=False).cumsum()
    prev_amount_sum = amount_cumsum - raw_amount
    mean_prev = prev_amount_sum / counts.replace(0, np.nan)
    mean_prev_filled = mean_prev.fillna(0).clip(0, 1e9)

    has_history = counts > 0
    amount_ratio = pd.Series(0.0, index=raw_amount.index, dtype="float64")
    amount_ratio.loc[has_history] = (
        np.log1p(raw_amount.loc[has_history])
        - np.log1p(mean_prev_filled.loc[has_history] + 0.01)
    )

    prev_fraud_count = (
        fraud_label.groupby(entity, sort=False).cumsum() - fraud_label
    ).clip(0, 1e7)
    fraud_rate = (
        (prev_fraud_count + 32.0 * global_prior) / (counts + 32.0)
    ).clip(0, 1)

    fraud_dt = dt.where(fraud_label > 0)
    last_fraud_seen = fraud_dt.groupby(entity, sort=False).ffill()
    prev_fraud_dt = last_fraud_seen.groupby(entity, sort=False).shift(1)
    since_prev_fraud = (dt - prev_fraud_dt).fillna(0).clip(0, 365 * 86400)

    out[f"hist_{prefix}_count_log"] = np.log1p(counts.clip(0, 1e7)).astype("float32")
    out[f"hist_{prefix}_since_prev_log"] = np.log1p(since_prev).astype("float32")
    out[f"hist_{prefix}_amount_mean_log"] = np.log1p(mean_prev_filled).astype("float32")
    out[f"hist_{prefix}_amount_ratio"] = amount_ratio.clip(-8, 8).astype("float32")
    out[f"hist_{prefix}_fraud_count_log"] = np.log1p(prev_fraud_count).astype("float32")
    out[f"hist_{prefix}_fraud_rate"] = fraud_rate.astype("float32")
    out[f"hist_{prefix}_since_prev_fraud_log"] = np.log1p(since_prev_fraud).astype("float32")


def schema_hash() -> str:
    payload = json.dumps(
        {"version": SCHEMA_VERSION, "feature_order": FEATURE_ORDER},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def should_skip_norm(feature: str) -> bool:
    return feature in BINARY_FEATURES


def validate_client_frame(cid: int, df: pd.DataFrame, norm_features: list[str]) -> dict:
    missing = [c for c in FEATURE_ORDER + [LABEL] if c not in df.columns]
    if missing:
        raise AssertionError(f"Client {cid}: missing columns {missing}")
    if list(df[FEATURE_ORDER].columns) != FEATURE_ORDER:
        raise AssertionError(f"Client {cid}: feature order mismatch")
    if df[FEATURE_ORDER].isna().any().any():
        raise AssertionError(f"Client {cid}: NaN values found")
    if not np.isfinite(df[FEATURE_ORDER].to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"Client {cid}: non-finite values found")
    labels = df[LABEL].astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise AssertionError(f"Client {cid}: labels must be binary")
    return {
        "client_id": cid,
        "rows": int(len(df)),
        "fraud_rate": float(labels.mean()),
        "positives": int(labels.sum()),
        "features": len(FEATURE_ORDER),
        "normalized_features": len(norm_features),
    }


def engineer(df: pd.DataFrame, tx_ref: pd.DataFrame) -> pd.DataFrame:
    """Transform raw merged IEEE-CIS columns into the fraud-history feature vector."""
    out = pd.DataFrame(index=df.index)

    raw_amount = df["TransactionAmt"].fillna(0).clip(0, 1e9).astype("float64")
    raw_count_1h = df["C1"].fillna(0).clip(0, 500).astype("float64")
    raw_count_24h = df["C2"].fillna(0).clip(0, 5000).astype("float64")
    fraud_label = df["isFraud"].astype("float64")
    global_prior = fraud_label.expanding().mean().shift(1).fillna(fraud_label.mean())

    out["tx_amount_usd"] = np.log1p(raw_amount).astype("float32")
    out["tx_count_1h"] = np.log1p(raw_count_1h).astype("float32")
    out["tx_count_24h"] = np.log1p(raw_count_24h).astype("float32")
    out["tx_volume_1h_usd"] = np.log1p((raw_amount * raw_count_1h).clip(0, 5e8)).astype("float32")
    out["tx_volume_24h_usd"] = np.log1p((raw_amount * raw_count_24h).clip(0, 5e9)).astype("float32")

    dist = df["dist1"].fillna(0).clip(0, 10000).astype("float64")
    days = df["D1"].fillna(1).clip(0.01, 365).astype("float64")
    velocity = ((dist * 1.60934) / (days * 24.0)).clip(0, 2000)
    out["geo_velocity_kmh"] = np.log1p(velocity).astype("float32")
    out["dist2_km"] = np.log1p(df["dist2"].fillna(0).clip(0, 10000).astype("float64") * 1.60934).astype("float32")

    amount_x_velocity = np.log1p(raw_amount) * np.log1p(velocity)
    out["amount_x_velocity"] = (
        amount_x_velocity / max(1.0 + amount_x_velocity.std(), 1.0)
    ).clip(-10, 10).astype("float32")
    out["amount_per_tx_1h"] = (
        np.log1p(raw_amount) - np.log1p(raw_count_1h + 0.1)
    ).clip(-5, 10).astype("float32")
    out["amount_per_tx_24h"] = (
        np.log1p(raw_amount) - np.log1p(raw_count_24h + 0.1)
    ).clip(-5, 10).astype("float32")
    out["spending_velocity_1h"] = (
        np.log1p(raw_count_1h) * 0.1 + np.log1p(raw_amount) * 0.9
    ).clip(0, 10).astype("float32")

    card6_map = {"debit": 0, "credit": 1, "charge card": 2, "debit or credit": 3}
    card6_int = df["card6"].map(card6_map).fillna(4).astype(int)
    for i, name in enumerate(["debit", "credit", "charge_card", "debit_or_credit"]):
        out[f"card6_{name}"] = (card6_int == i).astype("float32")

    out["days_since_last_tx"] = _log1p_col(df, "D1", 365)
    out["account_age_days"] = _log1p_col(df, "D3", 10000)

    dt = df["TransactionDT"].astype("float64")
    out["hour_of_day_local"] = ((dt // 3600) % 24).astype("float32")
    out["day_of_week"] = ((dt // 86400) % 7).astype("float32")
    out["tx_time_norm"] = ((dt % 86400) / 86400.0).astype("float32")
    out["week_of_period"] = ((dt % 604800) / 604800.0).astype("float32")
    hour = (dt // 3600) % 24
    unusual_hour = ((hour < 6) | (hour > 23)).astype("float64")
    out["risky_hour_flag"] = unusual_hour.astype("float32")
    out["early_morning_high_value"] = (unusual_hour * (np.log1p(raw_amount) > 5)).astype("float32")
    dow = (dt // 86400) % 7
    is_weekend = ((dow >= 5) | (dow <= 0)).astype("float64")
    out["weekend_high_value"] = (is_weekend * (np.log1p(raw_amount) > 4)).astype("float32")

    for cat in ["W", "H", "C", "S", "R"]:
        out[f"prod_{cat}"] = (tx_ref["ProductCD"] == cat).astype("float32")

    out["card1_norm"] = _log1p_col(df, "card1", 20000)
    out["card2_norm"] = _log1p_col(df, "card2", 1000)
    out["card3_norm"] = df["card3"].fillna(0).clip(0, 200).astype("float32")
    out["card5_norm"] = _log1p_col(df, "card5", 500)
    out["addr1_norm"] = _log1p_col(df, "addr1", 1000)
    out["addr2_norm"] = df["addr2"].fillna(0).clip(0, 100).astype("float32")
    card4_map = {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}
    out["card4_code"] = df["card4"].str.lower().map(card4_map).fillna(4).astype("float32")

    out["c5_chargeback"] = _log1p_col(df, "C5", 100)
    for i in range(3, 15):
        out[f"C{i}_log"] = _log1p_col(df, f"C{i}", 5000)
    for i in D_BASE_COLUMNS:
        col = f"D{i}"
        out[f"{col}_log"] = _log1p_col(df, col, 10000)
        out[f"{col}_missing"] = _missing_flag(df, col)

    free_mail = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "icloud.com", "live.com", "protonmail.com",
    }
    p_dom = df["P_emaildomain"].fillna("unknown").str.lower().str.strip()
    r_dom = df["R_emaildomain"].fillna("unknown").str.lower().str.strip()
    out["email_domain_match"] = (p_dom == r_dom).astype("float32")
    out["p_email_free"] = p_dom.isin(free_mail).astype("float32")
    out["r_email_free"] = r_dom.isin(free_mail).astype("float32")
    out["both_emails_free"] = (out["p_email_free"] * out["r_email_free"]).astype("float32")
    email_mismatch = (1 - out["email_domain_match"]).astype("float64")
    out["email_mismatch_high_value"] = (
        email_mismatch * (np.log1p(raw_amount) > 4.5)
    ).astype("float32")

    has_device = df["DeviceInfo"].notna().astype("float32")
    out["has_device_info"] = has_device
    card_entropy = (df["card1"].fillna(-1) != df["card1"].shift().fillna(-2)).astype("float32")
    out["card_device_mismatch"] = (has_device * card_entropy).astype("float32")
    acct_age = np.log1p(df["D3"].fillna(0).clip(0, 10000).astype("float64"))
    out["new_account_high_value"] = ((acct_age < 2) & (np.log1p(raw_amount) > 5)).astype("float32")

    out["card1_freq"] = np.log1p(_freq(df["card1"])).astype("float32")
    out["card2_freq"] = np.log1p(_freq(df["card2"])).astype("float32")
    out["card5_freq"] = np.log1p(_freq(df["card5"])).astype("float32")
    out["addr1_freq"] = np.log1p(_freq(df["addr1"])).astype("float32")
    out["p_email_freq"] = np.log1p(_freq(df["P_emaildomain"])).astype("float32")
    out["r_email_freq"] = np.log1p(_freq(df["R_emaildomain"])).astype("float32")
    out["device_info_freq"] = np.log1p(_freq(df["DeviceInfo"])).astype("float32")

    _add_history_features(out, dt, raw_amount, fraud_label, global_prior, df["card1"], "card1")
    _add_history_features(out, dt, raw_amount, fraud_label, global_prior, df["card2"], "card2")
    _add_history_features(out, dt, raw_amount, fraud_label, global_prior, df["addr1"], "addr1")
    _add_history_features(out, dt, raw_amount, fraud_label, global_prior, p_dom, "p_email")
    _add_history_features(out, dt, raw_amount, fraud_label, global_prior, r_dom, "r_email")
    _add_history_features(out, dt, raw_amount, fraud_label, global_prior, df["DeviceInfo"], "device_info")
    _add_history_features(
        out,
        dt,
        raw_amount,
        fraud_label,
        global_prior,
        _pair_key(df["card1"], df["card2"]),
        "card_pair",
    )
    _add_history_features(
        out,
        dt,
        raw_amount,
        fraud_label,
        global_prior,
        _pair_key(p_dom, r_dom),
        "email_pair",
    )

    for i in ID_NUMERIC_COLUMNS:
        col = f"id_{i:02d}"
        out[f"{col}_norm"] = df[col].fillna(0).clip(-100000, 100000).astype("float32") if col in df else 0.0
        out[f"{col}_missing"] = _missing_flag(df, col)

    device_type = df["DeviceType"].fillna("unknown").str.lower().str.strip()
    out["device_type_desktop"] = (device_type == "desktop").astype("float32")
    out["device_type_mobile"] = (device_type == "mobile").astype("float32")
    for col in ["id_12", "id_15", "id_16", "id_28", "id_29"]:
        out[f"{col}_found"] = df[col].notna().astype("float32") if col in df else 0.0

    for i in SELECTED_V_COLUMNS:
        col = f"V{i}"
        if col not in out:
            out[col] = df[col].fillna(0).clip(-1000, 1000).astype("float32") if col in df else 0.0

    out[LABEL] = df["isFraud"].astype("int8")
    out = out.reindex(columns=FEATURE_ORDER + [LABEL], fill_value=0.0)
    out[LABEL] = out[LABEL].astype("int8")
    if out[FEATURE_ORDER].isnull().sum().sum() != 0:
        raise AssertionError("Nulls found before normalization")
    return out


def main() -> None:
    print("Loading IEEE-CIS data...")
    tx = pd.read_csv(RAW_DIR / "train_transaction.csv")
    identity = pd.read_csv(RAW_DIR / "train_identity.csv")
    df = tx.merge(identity, on="TransactionID", how="left")
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    tx = tx.set_index("TransactionID").loc[df["TransactionID"]].reset_index()
    print(f"Merged: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")

    print("\nEngineering fraud-history features...")
    featured = engineer(df, tx)
    print(f"Feature matrix: {featured.shape}")
    print(f"Fraud rate: {featured[LABEL].mean() * 100:.2f}%")

    n = len(featured)
    cut1, cut2 = n // 3, 2 * n // 3
    clients_raw = {
        0: featured.iloc[:cut1].reset_index(drop=True),
        1: featured.iloc[cut1:cut2].reset_index(drop=True),
        2: featured.iloc[cut2:].reset_index(drop=True),
    }
    for cid, c in clients_raw.items():
        print(f"  Client {cid}: {len(c):,} rows | fraud={c[LABEL].mean() * 100:.2f}%")

    norm_features = [f for f in FEATURE_ORDER if not should_skip_norm(f)]
    print(f"\nComputing federated normalization params for {len(norm_features)} features...")
    all_stats = []
    for c in clients_raw.values():
        stats = {
            col: {
                "n": len(c),
                "sum": float(c[col].astype("float64").sum()),
                "sum_sq": float((c[col].astype("float64") ** 2).sum()),
            }
            for col in norm_features
        }
        all_stats.append(stats)

    global_params: dict = {}
    for col in norm_features:
        n_total = sum(s[col]["n"] for s in all_stats)
        total = sum(s[col]["sum"] for s in all_stats)
        total_sq = sum(s[col]["sum_sq"] for s in all_stats)
        mean = total / n_total
        variance = max(total_sq / n_total - mean**2, 0.0)
        std = max(np.sqrt(variance), 1e-8)
        global_params[col] = {"mean": round(mean, 8), "std": round(std, 8)}

    Path("config").mkdir(exist_ok=True)
    Path("config/normalization_params.json").write_text(json.dumps(global_params, indent=4))
    print("Saved config/normalization_params.json")

    RAW_PROC.mkdir(parents=True, exist_ok=True)
    client_reports = []
    print("\nNormalizing and saving processed data...")
    for cid, c in clients_raw.items():
        c = c.copy()
        for col in norm_features:
            c[col] = (
                (c[col].astype("float64") - global_params[col]["mean"])
                / global_params[col]["std"]
            ).astype("float32")
        out_dir = RAW_PROC / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "transactions_normalized.parquet"
        c.to_parquet(out_path, index=False)
        client_reports.append(validate_client_frame(cid, c, norm_features))
        print(f"  Client {cid}: saved {len(c):,} rows to {out_path}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "label": LABEL,
        "total_rows": int(len(featured)),
        "total_features": len(FEATURE_ORDER),
        "normalized_features": norm_features,
        "native_scale_features": sorted(BINARY_FEATURES),
        "clients": client_reports,
    }
    report_path = RAW_PROC / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {report_path}")
    print("\n[COMPLETE] IEEE-CIS data pipeline complete.")


if __name__ == "__main__":
    main()
