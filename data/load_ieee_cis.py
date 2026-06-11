"""
Load IEEE-CIS Fraud Detection dataset, engineer schema v5 features,
split into 3 federated clients by TransactionDT (temporal thirds),
and write normalized parquet files to data/processed/client_{0,1,2}/.

Currency conversion is handled by the serving layer (external FX API)
before calling the model — this pipeline assumes all amounts are already
in USD and stores no FX metadata.

Usage:
    uv run python data/load_ieee_cis.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = ["main"]

# ── Load schema ───────────────────────────────────────────────────────────────
with open("contracts/schema.json") as f:
    _s = json.load(f)
FEATURE_ORDER = _s["feature_schema"]["feature_order"]
NUMERIC       = [f["name"] for f in _s["feature_schema"]["numeric_features"]]
LABEL         = _s["feature_schema"]["label"]["name"]

RAW_DIR  = Path("data/ieee_cis")
RAW_PROC = Path("data/processed")


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, tx_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw merged IEEE-CIS columns into the model feature vector.

    All intermediate arithmetic is done in float64 to avoid accumulation
    of float32 rounding errors before the final cast to float32 for storage.
    The serving layer is responsible for converting amounts to USD before
    calling the model — no FX metadata is written here.
    """
    out = pd.DataFrame(index=df.index)

    # Work in float64 throughout to maximise precision before the final cast.
    raw_amount    = df["TransactionAmt"].fillna(0).clip(0, 1e9).astype("float64")
    raw_count_1h  = df["C1"].fillna(0).clip(0, 500).astype("float64")
    raw_count_24h = df["C2"].fillna(0).clip(0, 5000).astype("float64")

    # Log-scale skewed numeric inputs — log1p is exact in float64.
    out["tx_amount_usd"]  = np.log1p(raw_amount).astype("float32")
    out["tx_count_1h"]    = np.log1p(raw_count_1h).astype("float32")
    out["tx_count_24h"]   = np.log1p(raw_count_24h).astype("float32")

    # Volume = amount × count, then log-compressed.
    raw_volume_1h  = (raw_amount * raw_count_1h).clip(0, 5e8)
    raw_volume_24h = (raw_amount * raw_count_24h).clip(0, 5e9)
    out["tx_volume_1h_usd"]  = np.log1p(raw_volume_1h).astype("float32")
    out["tx_volume_24h_usd"] = np.log1p(raw_volume_24h).astype("float32")

    # Geo velocity — dist1 (miles) / (D1 days × 24 h) → km/h, log-scaled.
    dist = df["dist1"].fillna(0).clip(0, 10000).astype("float64")
    days = df["D1"].fillna(1).clip(0.01, 365).astype("float64")
    out["geo_velocity_kmh"] = np.log1p(
        ((dist * 1.60934) / (days * 24.0)).clip(0, 2000)
    ).astype("float32")

    # Second distance feature.
    dist2 = df["dist2"].fillna(0).clip(0, 10000).astype("float64")
    out["dist2_km"] = np.log1p(dist2 * 1.60934).astype("float32")

    # Card type signal — card6 encodes the funding source.
    card6_map = {"debit": 0, "credit": 1, "charge card": 2, "debit or credit": 3}
    out["card6_code"] = df["card6"].map(card6_map).fillna(4).astype("float32")

    # Days since last transaction (D1) and account age proxy (D3).
    d1 = df["D1"].fillna(0).clip(0, 365).astype("float64")
    d3 = df["D3"].fillna(0).clip(0, 10000).astype("float64")
    out["days_since_last_tx"] = np.log1p(d1).astype("float32")
    out["account_age_days"]   = np.log1p(d3).astype("float32")

    # Temporal features — TransactionDT is seconds from a reference epoch.
    dt = df["TransactionDT"].astype("float64")
    out["hour_of_day_local"] = ((dt // 3600) % 24).astype("float32")
    out["day_of_week"]       = ((dt // 86400) % 7).astype("float32")

    # ProductCD one-hot — lets the model distinguish product categories
    # even after FedAvg mixes client weights.
    for cat in ["W", "H", "C", "S", "R"]:
        out[f"prod_{cat}"] = (tx_ref["ProductCD"] == cat).astype("float32")

    # Card identity — card1/card2 are hashed card-number prefixes.
    card1 = df["card1"].fillna(0).clip(0, 20000).astype("float64")
    card2 = df["card2"].fillna(0).clip(0, 1000).astype("float64")
    out["card1_norm"] = np.log1p(card1).astype("float32")
    out["card2_norm"] = np.log1p(card2).astype("float32")

    # Address — addr1 = billing ZIP, addr2 = shipping country code.
    addr1 = df["addr1"].fillna(0).clip(0, 1000).astype("float64")
    out["addr1_norm"] = np.log1p(addr1).astype("float32")
    out["addr2_norm"] = df["addr2"].fillna(0).clip(0, 100).astype("float32")

    # Top Vesta proprietary features — V258/V257 = payment-network velocity,
    # V201 = account behaviour. Computed in float64, cast at save.
    out["V258"] = df["V258"].fillna(0).clip(-100, 100).astype("float64").astype("float32")
    out["V257"] = df["V257"].fillna(0).clip(-100, 100).astype("float64").astype("float32")
    out["V201"] = df["V201"].fillna(0).clip(-100, 100).astype("float64").astype("float32")

    # M-flags — binary Vesta match indicators.
    # M4 = order match, M6 = address match; "F" = mismatch = fraud signal.
    # NaN → 0.5 (unknown) rather than assuming a match.
    m_map = {"T": 1.0, "F": 0.0}
    out["M4_flag"] = df["M4"].map(m_map).fillna(0.5).astype("float32")
    out["M6_flag"] = df["M6"].map(m_map).fillna(0.5).astype("float32")

    # C5 = prior chargeback count on this card — near-direct fraud indicator.
    c5 = df["C5"].fillna(0).clip(0, 100).astype("float64")
    out["c5_chargeback"] = np.log1p(c5).astype("float32")

    # Email domain features.
    # Domain mismatch between buyer (P) and recipient (R) is a strong signal.
    # Free/anonymous providers have higher fraud rates than corporate domains.
    _free_mail = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "icloud.com", "live.com", "protonmail.com",
    }
    p_dom = df["P_emaildomain"].fillna("unknown").str.lower().str.strip()
    r_dom = df["R_emaildomain"].fillna("unknown").str.lower().str.strip()
    out["email_domain_match"] = (p_dom == r_dom).astype("float32")
    out["p_email_free"]       = p_dom.isin(_free_mail).astype("float32")
    out["r_email_free"]       = r_dom.isin(_free_mail).astype("float32")

    # card3 = issuing-bank sub-code; card4 = network (visa/mc/amex/discover).
    out["card3_norm"] = df["card3"].fillna(0).clip(0, 200).astype("float32")
    card4_map = {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}
    out["card4_code"] = (
        df["card4"].str.lower().map(card4_map).fillna(4).astype("float32")
    )

    # Label
    out[LABEL] = df["isFraud"].astype("int8")

    # No FX metadata columns — currency conversion is the serving layer's job.
    full_cols = FEATURE_ORDER + [LABEL]
    out = out[full_cols].fillna(0)

    assert list(out.columns[:len(FEATURE_ORDER)]) == FEATURE_ORDER, "Column order mismatch"
    assert out.isnull().sum().sum() == 0, "Nulls found after fillna"
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading IEEE-CIS data...")
    tx  = pd.read_csv(RAW_DIR / "train_transaction.csv")
    id_ = pd.read_csv(RAW_DIR / "train_identity.csv")
    df  = tx.merge(id_, on="TransactionID", how="left")
    print(f"Merged: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("\nEngineering features...")
    featured = engineer(df, tx)
    print(f"Feature matrix: {featured.shape}")
    print(f"Fraud rate: {featured[LABEL].mean() * 100:.2f}%")

    # ── Step 3: Split into 3 clients by TransactionDT (temporal thirds) ───────
    # IEEE-CIS CSV rows are already in TransactionDT order, so row-index slicing
    # gives a proper temporal split: ~3.5% fraud and mixed ProductCD per client.
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

    # ── Step 4: Federated normalisation ───────────────────────────────────────
    # Each client contributes its local sufficient statistics (n, sum, sum_sq).
    # The server aggregates them to compute a global mean/std without seeing
    # raw data. All accumulation in float64 to avoid precision loss at scale.
    print("\nComputing federated normalization params...")

    all_stats = []
    for c in clients_raw.values():
        stats = {
            col: {
                "n":      len(c),
                "sum":    float(c[col].astype("float64").sum()),
                "sum_sq": float((c[col].astype("float64") ** 2).sum()),
            }
            for col in NUMERIC
        }
        all_stats.append(stats)

    global_params: dict = {}
    for col in NUMERIC:
        N  = sum(s[col]["n"]      for s in all_stats)
        S  = sum(s[col]["sum"]    for s in all_stats)
        SQ = sum(s[col]["sum_sq"] for s in all_stats)
        mean     = S / N
        variance = max(SQ / N - mean ** 2, 0.0)
        std      = max(np.sqrt(variance), 1e-8)
        # 8 decimal places — enough for float32 training without false precision.
        global_params[col] = {"mean": round(mean, 8), "std": round(std, 8)}
        print(f"  {col:25s}: mean={mean:14.6f}, std={std:14.6f}")

    with open("contracts/normalization_params.json", "w") as f:
        json.dump(global_params, f, indent=2)
    print("\nSaved contracts/normalization_params.json")

    # ── Step 5: Apply normalisation and save processed parquets ───────────────
    # Normalise in float64 then cast to float32 for storage.
    print("\nNormalizing and saving processed data...")

    for cid, c in clients_raw.items():
        c = c.copy()
        for col in NUMERIC:
            c[col] = (
                (c[col].astype("float64") - global_params[col]["mean"])
                / global_params[col]["std"]
            ).astype("float32")

        out_dir  = RAW_PROC / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "transactions_normalized.parquet"
        c.to_parquet(out_path, index=False)

        assert c[NUMERIC].notna().all().all(), \
            f"Normalization produced NaN values for client {cid}"
        assert np.isfinite(c[NUMERIC].values).all(), \
            f"Normalization produced non-finite values for client {cid}"
        print(f"  Client {cid}: saved {len(c):,} rows → {out_path}")

    print("\n✓ IEEE-CIS data pipeline complete.")
    print("  Processed data is available under data/processed/client_{0,1,2}/")
    print("  Next: start the server and clients using the repo's Python entrypoints.")


if __name__ == "__main__":
    main()