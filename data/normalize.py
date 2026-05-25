# data/normalize.py
# normalize numeric features across all client datasets
import json
from pathlib import Path

import numpy as np
import pandas as pd

with open("contracts/schema.json") as f:
    _s = json.load(f)
NUMERIC = [f["name"] for f in _s["feature_schema"]["numeric_features"]]

# = ["tx_amount_usd","tx_count_1h","tx_count_24h",
#    "tx_volume_1h_usd","tx_volume_24h_usd","merchant_cat_dev",
#    "geo_velocity_kmh","days_since_last_tx","account_age_days"]

def local_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        c: {
            "n": len(df),
            "sum": float(df[c].sum()),
            "sum_sq": float((df[c] ** 2).sum()),
        }
        for c in NUMERIC
    }


def main() -> None:
    all_stats = [
        local_stats(pd.read_parquet(f"data/raw/client_{i}/transactions.parquet"))
        for i in range(3)
    ]

    global_params: dict[str, dict[str, float]] = {}
    for col in NUMERIC:
        N = sum(s[col]["n"] for s in all_stats)
        S = sum(s[col]["sum"] for s in all_stats)
        SQ = sum(s[col]["sum_sq"] for s in all_stats)
        mean = S / N
        std = max(np.sqrt(SQ / N - mean**2), 1e-8)
        global_params[col] = {"mean": round(mean, 6), "std": round(std, 6)}
        print(f"  {col:25s}: mean={mean:12.3f}, std={std:12.3f}")

    with open("contracts/normalization_params.json", "w") as f:
        json.dump(global_params, f, indent=2)
    print("Saved contracts/normalization_params.json")

    for i in range(3):
        df = pd.read_parquet(f"data/raw/client_{i}/transactions.parquet")
        for col in NUMERIC:
            df[col] = (df[col] - global_params[col]["mean"]) / global_params[col]["std"]
        assert abs(df[NUMERIC[0]].mean()) < 0.1, f"Norm failed client {i}"
        out = Path(f"data/processed/client_{i}")
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "transactions_normalized.parquet", index=False)
        print(f"Client {i}: normalized, tx_amount_usd mean={df['tx_amount_usd'].mean():.4f}")


if __name__ == "__main__":
    main()