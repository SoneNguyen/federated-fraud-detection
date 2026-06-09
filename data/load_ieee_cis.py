"""
Load IEEE-CIS Fraud Detection dataset, engineer schema v3 features,
split into 3 heterogeneous federated clients, and write normalized
parquet files to data/processed/client_{0,1,2}/.

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
FEATURE_ORDER = _s["feature_schema"]["feature_order"]       # 11 items
NUMERIC       = [f["name"] for f in _s["feature_schema"]["numeric_features"]]  # 9 items
LABEL         = _s["feature_schema"]["label"]["name"]       # is_fraud

RAW_DIR  = Path("data/ieee_cis")
RAW_PROC = Path("data/processed")

# ── Step 1: Load and merge ────────────────────────────────────────────────────

def engineer(df, tx_ref):
    out = pd.DataFrame(index=df.index)

    # tx_amount_usd — TransactionAmt is already USD
    out["tx_amount_usd"] = df["TransactionAmt"].fillna(0).clip(0, 1e9).astype("float32")

    # Velocity counts — C1/C2 are pre-computed count features in IEEE-CIS
    out["tx_count_1h"]  = df["C1"].fillna(0).clip(0, 500).astype("int32")
    out["tx_count_24h"] = df["C2"].fillna(0).clip(0, 5000).astype("int32")

    # Volume = amount × count (approximation — IEEE-CIS has no raw volume cols)
    out["tx_volume_1h_usd"]  = (out["tx_amount_usd"] * out["tx_count_1h"]).clip(0, 5e8).astype("float32")
    out["tx_volume_24h_usd"] = (out["tx_amount_usd"] * out["tx_count_24h"]).clip(0, 5e9).astype("float32")

    # Merchant category deviation — ProductCD z-scored
    prod_map  = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
    prod_code = tx_ref["ProductCD"].map(prod_map).fillna(0)
    mean, std = prod_code.mean(), prod_code.std() + 1e-8
    out["merchant_cat_dev"] = ((prod_code - mean) / std).clip(-5, 5).astype("float32")

    # Geo velocity — dist1 (miles) / (D1 days × 24h) → km/h
    dist = df["dist1"].fillna(0).clip(0, 10000)
    days = df["D1"].fillna(1).clip(0.01, 365)
    out["geo_velocity_kmh"] = ((dist * 1.60934) / (days * 24)).clip(0, 2000).astype("float32")

    # Days since last transaction
    out["days_since_last_tx"] = df["D1"].fillna(0).clip(0, 365).astype("float32")

    # Account age proxy — D3 = days since last tx with this card
    out["account_age_days"] = df["D3"].fillna(0).clip(0, 10000).astype("int32")

    # Local hour — TransactionDT is seconds from a reference epoch
    out["hour_of_day_local"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    out["day_of_week"]       = ((df["TransactionDT"] // 86400) % 7).astype("int8")

    # Label
    out[LABEL] = df["isFraud"].astype("int8")

    # Metadata columns (not model inputs)
    out["orig_currency"]  = "USD"
    out["stale_fx_flag"]  = 0

    # Enforce order: FEATURE_ORDER + metadata + label
    full_cols = FEATURE_ORDER + ["orig_currency", "stale_fx_flag", LABEL]
    out = out[full_cols].fillna(0)

    assert list(out.columns[:11]) == FEATURE_ORDER, "Column order mismatch"
    assert out.isnull().sum().sum() == 0, "Nulls found after fillna"
    return out


def main() -> None:
    print("Loading IEEE-CIS data...")
    tx  = pd.read_csv(RAW_DIR / "train_transaction.csv")
    id_ = pd.read_csv(RAW_DIR / "train_identity.csv")
    df  = tx.merge(id_, on="TransactionID", how="left")
    print(f"Merged: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

    print("\nEngineering features...")
    featured = engineer(df, tx)
    print(f"Feature matrix: {featured.shape}")
    print(f"Fraud rate: {featured[LABEL].mean()*100:.2f}%")

    # ── Step 3: Split into 3 clients by ProductCD ─────────────────────────────────
    print("\nSplitting into 3 federated clients...")
    prod = tx["ProductCD"].reindex(featured.index)

    clients_raw = {
        0: featured[prod == "W"].reset_index(drop=True),
        1: featured[prod.isin(["C", "H"])].reset_index(drop=True),
        2: featured[prod.isin(["S", "R"])].reset_index(drop=True),
    }

    for cid, c in clients_raw.items():
        print(f"  Client {cid}: {len(c):,} rows | fraud={c[LABEL].mean()*100:.2f}%")

    # ── Step 4: Federated normalization ───────────────────────────────────────────
    print("\nComputing federated normalization params...")

    all_stats = []
    for c in clients_raw.values():
        stats = {col: {"n": len(c), "sum": float(c[col].sum()),
                       "sum_sq": float((c[col]**2).sum())}
                 for col in NUMERIC}
        all_stats.append(stats)

    global_params = {}
    for col in NUMERIC:
        N  = sum(s[col]["n"]      for s in all_stats)
        S  = sum(s[col]["sum"]    for s in all_stats)
        SQ = sum(s[col]["sum_sq"] for s in all_stats)
        mean = S / N
        std  = max(np.sqrt(SQ/N - mean**2), 1e-8)
        global_params[col] = {"mean": round(mean, 6), "std": round(std, 6)}
        print(f"  {col:25s}: mean={mean:12.4f}, std={std:12.4f}")

# Save normalization params for the IEEE-CIS workflow
    with open("contracts/normalization_params.json", "w") as f:
        json.dump(global_params, f, indent=2)
    print("\nSaved contracts/normalization_params.json")

    # ── Step 5: Apply normalization and save processed parquets ───────────────────
    print("\nNormalizing and saving processed data...")

    for cid, c in clients_raw.items():
        c = c.copy()
        for col in NUMERIC:
            c[col] = ((c[col] - global_params[col]["mean"]) /
                      global_params[col]["std"]).astype("float32")

        out_dir = RAW_PROC / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "transactions_normalized.parquet"
        c.to_parquet(out_path, index=False)

        # Validation
        assert abs(c[NUMERIC[0]].mean()) < 0.5, f"Normalization off for client {cid}"
        print(f"  Client {cid}: saved {len(c):,} rows → {out_path}")

    print("\n✓ IEEE-CIS data pipeline complete.")
    print("  Processed data is available under data/processed/client_{0,1,2}/")
    print("  Next: start the server and clients using the repo's Python entrypoints.")


if __name__ == "__main__":
    main()