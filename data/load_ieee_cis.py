"""
Load IEEE-CIS Fraud Detection dataset, engineer schema v5 features,
split into 3 federated clients by TransactionDT (temporal thirds),
and write normalized parquet files to data/processed/client_{0,1,2}/.

Key changes vs previous version:
- fillna(0) is applied BEFORE normalization only for the numeric value.
- Currency note: pipeline assumes all amounts are already in USD.

Usage:
    uv run python data/load_ieee_cis.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = ["main"]

# ── Load schema ───────────────────────────────────────────────────────────────
with open("config/schema.json") as f:
    _s = json.load(f)
FEATURE_ORDER = _s["feature_schema"]["feature_order"]
NUMERIC       = [f["name"] for f in _s["feature_schema"]["numeric_features"]]
LABEL         = _s["feature_schema"]["label"]["name"]   # "is_fraud"

RAW_DIR  = Path("data/ieee_cis")
RAW_PROC = Path("data/processed")


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, tx_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw merged IEEE-CIS columns into the model feature vector.

    Enhanced with interaction features and identity/device grouping signals:
    - Amount × velocity: high-spend rapid movement pattern
    - Amount × count interactions: spending bursts
    - Temporal risk patterns: unusual times + high-value txns
    - Device/identity consistency: same device/IP but different card
    - Email domain reputation: free vs enterprise

    All intermediate arithmetic is done in float64; final cast to float32
    for storage efficiency. Missing-value indicators are float32 {0, 1}.
    """
    out = pd.DataFrame(index=df.index)

    # ── Core numeric features (float64 intermediates) ─────────────────────────
    raw_amount    = df["TransactionAmt"].fillna(0).clip(0, 1e9).astype("float64")
    raw_count_1h  = df["C1"].fillna(0).clip(0, 500).astype("float64")
    raw_count_24h = df["C2"].fillna(0).clip(0, 5000).astype("float64")

    out["tx_amount_usd"]  = np.log1p(raw_amount).astype("float32")
    out["tx_count_1h"]    = np.log1p(raw_count_1h).astype("float32")
    out["tx_count_24h"]   = np.log1p(raw_count_24h).astype("float32")

    raw_volume_1h  = (raw_amount * raw_count_1h).clip(0, 5e8)
    raw_volume_24h = (raw_amount * raw_count_24h).clip(0, 5e9)
    out["tx_volume_1h_usd"]  = np.log1p(raw_volume_1h).astype("float32")
    out["tx_volume_24h_usd"] = np.log1p(raw_volume_24h).astype("float32")

    # Geo velocity — dist1 (miles) / (D1 days × 24h) → km/h, log-scaled.
    dist  = df["dist1"].fillna(0).clip(0, 10000).astype("float64")
    days  = df["D1"].fillna(1).clip(0.01, 365).astype("float64")
    out["geo_velocity_kmh"] = np.log1p(
        ((dist * 1.60934) / (days * 24.0)).clip(0, 2000)
    ).astype("float32")

    dist2 = df["dist2"].fillna(0).clip(0, 10000).astype("float64")
    out["dist2_km"] = np.log1p(dist2 * 1.60934).astype("float32")
    
    # ── NEW: Amount × velocity interaction (high-risk combo) ──────────────────
    amount_x_velocity = np.log1p(raw_amount) * np.log1p(
        ((dist * 1.60934) / (days * 24.0)).clip(0, 2000)
    )
    out["amount_x_velocity"] = (amount_x_velocity / (1 + amount_x_velocity.std())).clip(-10, 10).astype("float32")
    
    # ── NEW: Amount × count interactions ───────────────────────────────────────
    out["amount_per_tx_1h"] = (np.log1p(raw_amount) - np.log1p(raw_count_1h + 0.1)).clip(-5, 10).astype("float32")
    out["amount_per_tx_24h"] = (np.log1p(raw_amount) - np.log1p(raw_count_24h + 0.1)).clip(-5, 10).astype("float32")
    
    # ── NEW: Spending burst indicator (high count but consistent amount) ───────
    out["spending_velocity_1h"] = (np.log1p(raw_count_1h) * 0.1 + np.log1p(raw_amount) * 0.9).clip(0, 10).astype("float32")

    # Card type — one-hot encoding (not ordinal, which was false assumption).
    # Encode k-1=4 indicators to avoid perfect multicollinearity; unknown is reference.
    card6_map = {"debit": 0, "credit": 1, "charge card": 2, "debit or credit": 3}
    card6_int = df["card6"].map(card6_map).fillna(4).astype(int)
    for i, name in enumerate(["debit", "credit", "charge_card", "debit_or_credit"]):
        out[f"card6_{name}"] = (card6_int == i).astype("float32")

    d1 = df["D1"].fillna(0).clip(0, 365).astype("float64")
    d3 = df["D3"].fillna(0).clip(0, 10000).astype("float64")
    out["days_since_last_tx"] = np.log1p(d1).astype("float32")
    out["account_age_days"]   = np.log1p(d3).astype("float32")

    dt = df["TransactionDT"].astype("float64")
    out["hour_of_day_local"] = ((dt // 3600) % 24).astype("float32")
    out["day_of_week"]       = ((dt // 86400) % 7).astype("float32")
    out["tx_time_norm"]      = ((dt % 86400) / 86400.0).astype("float32")
    out["week_of_period"]    = ((dt % 604800) / 604800.0).astype("float32")
    
    # ── NEW: Temporal risk patterns ─────────────────────────────────────────────
    # High-value transactions at unusual hours (red flag)
    hour = (dt // 3600) % 24
    unusual_hour = ((hour < 6) | (hour > 23)).astype("float64")
    out["risky_hour_flag"] = unusual_hour.astype("float32")
    
    # Early morning high-value transactions (especially risky)
    out["early_morning_high_value"] = (unusual_hour * (np.log1p(raw_amount) > 5)).astype("float32")
    
    # Weekend activity pattern
    dow = (dt // 86400) % 7
    is_weekend = ((dow >= 5) | (dow <= 0)).astype("float64")
    out["weekend_high_value"] = (is_weekend * (np.log1p(raw_amount) > 4)).astype("float32")

    # ProductCD one-hot
    for cat in ["W", "H", "C", "S", "R"]:
        out[f"prod_{cat}"] = (tx_ref["ProductCD"] == cat).astype("float32")

    card1 = df["card1"].fillna(0).clip(0, 20000).astype("float64")
    card2 = df["card2"].fillna(0).clip(0, 1000).astype("float64")
    out["card1_norm"] = np.log1p(card1).astype("float32")
    out["card2_norm"] = np.log1p(card2).astype("float32")

    addr1 = df["addr1"].fillna(0).clip(0, 1000).astype("float64")
    out["addr1_norm"] = np.log1p(addr1).astype("float32")
    out["addr2_norm"] = df["addr2"].fillna(0).clip(0, 100).astype("float32")

    # V-features — fill missing values with 0 after clipping.
    out["V258"] = df["V258"].fillna(0).clip(-100, 100).astype("float32")
    out["V257"] = df["V257"].fillna(0).clip(-100, 100).astype("float32")
    out["V201"] = df["V201"].fillna(0).clip(-100, 100).astype("float32")

    c5 = df["C5"].fillna(0).clip(0, 100).astype("float64")
    out["c5_chargeback"] = np.log1p(c5).astype("float32")

    _free_mail = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "icloud.com", "live.com", "protonmail.com",
    }
    p_dom = df["P_emaildomain"].fillna("unknown").str.lower().str.strip()
    r_dom = df["R_emaildomain"].fillna("unknown").str.lower().str.strip()
    out["email_domain_match"] = (p_dom == r_dom).astype("float32")
    out["p_email_free"]       = p_dom.isin(_free_mail).astype("float32")
    out["r_email_free"]       = r_dom.isin(_free_mail).astype("float32")
    
    # ── NEW: Email domain consistency patterns ───────────────────────────────
    # Both free domain (suspicious combo)
    out["both_emails_free"] = (out["p_email_free"] * out["r_email_free"]).astype("float32")
    
    # Email mismatch with high transaction amount (device theft indicator)
    email_mismatch = (1 - out["email_domain_match"]).astype("float64")
    out["email_mismatch_high_value"] = (
        email_mismatch * (np.log1p(raw_amount) > 4.5)
    ).astype("float32")

    out["card3_norm"] = df["card3"].fillna(0).clip(0, 200).astype("float32")
    card4_map = {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}
    out["card4_code"] = (
        df["card4"].str.lower().map(card4_map).fillna(4).astype("float32")
    )
    
    # ── NEW: Device/Identity consistency features ────────────────────────────
    # Device ID (D2 if available, otherwise use IP)
    has_device = df["DeviceInfo"].notna().astype("float32")
    out["has_device_info"] = has_device
    
    # Card-device consistency: multiple cards on same device is suspicious
    card_entropy = (df["card1"].fillna(-1) != df["card1"].shift().fillna(-2)).astype("float32")
    out["card_device_mismatch"] = (has_device * card_entropy).astype("float32")
    
    # Account age combined with high transaction (new account fraud)
    acct_age = np.log1p(d3)
    new_account_high_value = (acct_age < 2) & (np.log1p(raw_amount) > 5)
    out["new_account_high_value"] = new_account_high_value.astype("float32")

    # Label — stored as int8 to match schema
    out[LABEL] = df["isFraud"].astype("int8")

    # Reorder to match schema and ensure all expected columns are present.
    full_cols = FEATURE_ORDER + [LABEL]
    out = out.reindex(columns=full_cols, fill_value=0.0)
    out[LABEL] = out[LABEL].astype("int8")

    missing_before = out[FEATURE_ORDER].isnull().sum().sum()
    assert missing_before == 0, f"Nulls found before normalization: {missing_before}"
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading IEEE-CIS data...")
    tx  = pd.read_csv(RAW_DIR / "train_transaction.csv")
    id_ = pd.read_csv(RAW_DIR / "train_identity.csv")
    df  = tx.merge(id_, on="TransactionID", how="left")
    print(f"Merged: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")

    print("\nEngineering features...")
    featured = engineer(df, tx)
    print(f"Feature matrix: {featured.shape}")
    print(f"Fraud rate: {featured[LABEL].mean() * 100:.2f}%")

    # ── Temporal split into 3 federated clients ───────────────────────────────
    print("\nSplitting into 3 federated clients (temporal thirds)...")
    n = len(featured)
    cut1, cut2 = n // 3, 2 * n // 3

    clients_raw = {
        0: featured.iloc[:cut1].reset_index(drop=True),
        1: featured.iloc[cut1:cut2].reset_index(drop=True),
        2: featured.iloc[cut2:].reset_index(drop=True),
    }

    for cid, c in clients_raw.items():
        print(f"  Client {cid}: {len(c):,} rows | fraud={c[LABEL].mean() * 100:.2f}%")

    # ── Federated normalization (numeric features only, NOT indicators) ────────
    # Binary/indicator features (prod_*, *_missing, M4/M6 flags, email flags)
    # are excluded from normalization — they're already in [0,1].
    SKIP_NORM = {
        f for f in NUMERIC
        if any(f.startswith(p) for p in ("prod_", "M4", "M6", "email_", "p_email", "r_email"))
        or f.endswith("_missing")
    }
    NORM_FEATURES = [f for f in NUMERIC if f not in SKIP_NORM]

    print(f"\nComputing federated normalization params for {len(NORM_FEATURES)} features...")
    print(f"  (Skipping {len(SKIP_NORM)} binary/indicator features)")

    all_stats = []
    for c in clients_raw.values():
        stats = {
            col: {
                "n":      len(c),
                "sum":    float(c[col].astype("float64").sum()),
                "sum_sq": float((c[col].astype("float64") ** 2).sum()),
            }
            for col in NORM_FEATURES
        }
        all_stats.append(stats)

    global_params: dict = {}
    for col in NORM_FEATURES:
        N  = sum(s[col]["n"]      for s in all_stats)
        S  = sum(s[col]["sum"]    for s in all_stats)
        SQ = sum(s[col]["sum_sq"] for s in all_stats)
        mean     = S / N
        variance = max(SQ / N - mean ** 2, 0.0)
        std      = max(np.sqrt(variance), 1e-8)
        global_params[col] = {"mean": round(mean, 8), "std": round(std, 8)}
        print(f"  {col:25s}: mean={mean:14.6f}, std={std:14.6f}")

    with open("config/normalization_params.json", "w") as f:
        json.dump(global_params, f, indent=4)
    print("\nSaved config/normalization_params.json")

    # ── Apply normalization and save ──────────────────────────────────────────
    print("\nNormalizing and saving processed data...")

    for cid, c in clients_raw.items():
        c = c.copy()
        for col in NORM_FEATURES:
            c[col] = (
                (c[col].astype("float64") - global_params[col]["mean"])
                / global_params[col]["std"]
            ).astype("float32")
        # Binary/indicator features are left as-is (already [0,1])

        out_dir  = RAW_PROC / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "transactions_normalized.parquet"
        c.to_parquet(out_path, index=False)

        assert c[NORM_FEATURES].notna().all().all(), \
            f"Normalization produced NaN for client {cid}"
        assert np.isfinite(c[NORM_FEATURES].values).all(), \
            f"Normalization produced non-finite values for client {cid}"
        print(f"  Client {cid}: saved {len(c):,} rows to {out_path}")

    print("\n[COMPLETE] IEEE-CIS data pipeline complete.")


if __name__ == "__main__":
    main()