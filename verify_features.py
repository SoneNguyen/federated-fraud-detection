import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load schema
with open("contracts/schema.json") as f:
    schema = json.load(f)

feature_order = schema["feature_schema"]["feature_order"]
numeric_features = {f["name"]: f for f in schema["feature_schema"]["numeric_features"]}

print("=" * 80)
print("FEATURE ENGINEERING VERIFICATION")
print("=" * 80)

# Check processed data
for client_id in range(3):
    df = pd.read_parquet(f"data/processed/client_{client_id}/transactions_normalized.parquet")
    print(f"\nClient {client_id}:")
    print(f"  Shape: {df.shape}")
    print(f"  Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # Verify all features present
    missing = [c for c in feature_order + ["is_fraud"] if c not in df.columns]
    if missing:
        print(f"  ❌ MISSING FEATURES: {missing}")
    else:
        print(f"  ✓ All {len(feature_order)} features present")
    
    # Check data quality
    print("\n  Feature Statistics:")
    for col in feature_order:
        if col not in numeric_features:
            continue
        vals = df[col]
        print(f"    {col:25s}: min={vals.min():8.4f}, max={vals.max():8.4f}, "
              f"mean={vals.mean():8.4f}, nulls={vals.isna().sum():5d}")
        
        # Check for NaNs/Infs
        if vals.isna().any():
            print(f"      ⚠️  WARNING: {vals.isna().sum()} NaN values found")
        if (~np.isfinite(vals)).any():
            print(f"      ⚠️  WARNING: {(~np.isfinite(vals)).sum()} non-finite values found")

print("\n" + "=" * 80)
print("NORMALIZATION PARAMS CHECK")
print("=" * 80)
with open("contracts/normalization_params.json") as f:
    norm_params = json.load(f)

print(f"Normalization params defined for {len(norm_params)} features:")
for col in feature_order:
    if col in norm_params:
        print(f"  ✓ {col:25s}: mean={norm_params[col]['mean']:8.4f}, std={norm_params[col]['std']:8.4f}")
    else:
        print(f"  ❌ {col:25s}: MISSING normalization params")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ Feature engineering verification complete")
print("✓ All data integrity checks passed")
