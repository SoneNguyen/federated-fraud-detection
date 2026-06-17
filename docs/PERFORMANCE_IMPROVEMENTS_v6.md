
# Performance Improvement Changes (v6.0)

## Target: AUPRC 0.70, F1 Score 0.70
**Current baseline: AUPRC 0.53** → Need +32% improvement

## Changes Made

### 1. Enhanced Feature Engineering (Priority: HIGHEST)
**File**: `data/load_ieee_cis.py`, `config/schema.json`
**Rationale**: Better fraud-specific features directly improve model capacity

#### New Interaction Features:
- `amount_x_velocity`: High-spend rapid movement (amount × geographic velocity) 
- `amount_per_tx_1h`: Average amount per transaction in 1h window
- `amount_per_tx_24h`: Average amount per transaction in 24h window
- `spending_velocity_1h`: Combined spending + frequency signal

#### New Temporal Risk Patterns:
- `risky_hour_flag`: Transactions at unusual hours (midnight-6am, 11pm-midnight)
- `early_morning_high_value`: Early morning + high transaction amount combo
- `weekend_high_value`: Weekend + high-value transaction combo

#### New Identity/Email Grouping:
- `both_emails_free`: Both payer and receiver using free email (suspicious)
- `email_mismatch_high_value`: Email domain mismatch + high amount

#### New Device/Account Consistency:
- `has_device_info`: Whether device info is available
- `card_device_mismatch`: Multiple cards on same device inconsistency
- `new_account_high_value`: Brand new account (age < 2 days) + high value

**Result**: 37 features → 50 features
**Schema version updated**: 5.0 → 6.0

---

### 2. Less Aggressive Imbalance Handling (Priority: HIGH)
**File**: `src/client/client.py`

#### Focal Loss Tuning:
- Alpha: 0.75 → **0.5** (less aggressive positive class weighting)
- Gamma: 2.0 (unchanged - good balance)
- Effect: Allows more natural data distribution to guide training

#### WeightedRandomSampler Tuning:
- Oversampling cap: 30% → **15%** (less distortion)
- Multiplier: 5× → **2.5×** natural fraud rate
- Effect: Training distribution stays much closer to real distribution, improving calibration

**Why**: 
- Aggressive imbalance handling causes model to overpredict fraud
- Lower performance on calibration/thresholding
- Natural distribution guidance typically works better in production

---

### 3. LightGBM Baseline Model (Priority: MEDIUM)
**File**: `scripts/lightgbm_baseline.py`

**Purpose**: Validate that the feature set and data split can achieve ≥0.66 AUPRC
- Tests data quality and feature engineering independently of neural network
- Helps diagnose if problem is model architecture vs. features/data
- Useful for hybrid ensemble approaches later

**Run**: `uv run python -m scripts.lightgbm_baseline`
**Output**:
- Prints AUPRC, AUROC, F1 on test set
- Saves results to `outputs/lightgbm_baseline_results.json`

---

### 4. Early Stopping Support (Priority: MEDIUM)
**File**: `src/server/strategy.py` (already implemented)

**Current implementation**:
- Tracks best AUPRC per round
- Saves "best" checkpoint when new AUPRC record achieved
- Logs warning after 10 rounds without improvement
- Ready for external stopping trigger

**Usage**: Monitor `results/evaluation_history.json` for stalled progress
**Future enhancement**: Stop server when patience exhausted

---

## Expected Improvements

### Feature Additions (5× interaction features):
- **AUPRC**: +0.05-0.08 (typical for well-designed interactions)
- Captures non-linear patterns (amount×velocity) common in fraud

### Less Aggressive Imbalance Handling:
- **Calibration**: Improved (model confidence matches empirical frequency)
- **AUPRC**: +0.02-0.05 (better threshold flexibility)
- Reduced overprediction at low fraud rates

### Ensemble Potential (future):
- LightGBM baseline can be blended if it outperforms MLP
- Use best model for each client based on their data distribution

---

## Validation Steps

### 1. Verify Feature Encoding:
```powershell
uv run python data/load_ieee_cis.py
# Check: Feature matrix shape should be (n, 50) not (n, 37)
# Check: All new features computed without NaN/inf
```

### 2. Run LightGBM Baseline:
```powershell
uv run python -m scripts.lightgbm_baseline
# Expected: AUPRC ≥ 0.60 (at minimum validates features)
# If AUPRC < 0.60: Data/features need further work
```

### 3. Train Federated Model (with new features):
```powershell
# Terminal 1: Start server
uv run python -m scripts.run_server

# Terminal 2-4: Start 3 clients
$env:CLIENT_ID = 0; $env:DATA_PATH = "data/processed/client_0/transactions_normalized.parquet"; uv run python -m scripts.run_client
$env:CLIENT_ID = 1; $env:DATA_PATH = "data/processed/client_1/transactions_normalized.parquet"; uv run python -m scripts.run_client
$env:CLIENT_ID = 2; $env:DATA_PATH = "data/processed/client_2/transactions_normalized.parquet"; uv run python -m scripts.run_client
```

### 4. Check Results:
```powershell
# After ~10 rounds, inspect evaluation history
cat results/evaluation_history.json | python -m json.tool
# Look for: val_auprc, val_auroc, val_f1
# Goal: AUPRC >= 0.70, F1 >= 0.70
```

---

## Config Changes Summary

| Setting | Old | New | Reason |
|---------|-----|-----|--------|
| `focal_alpha` | 0.75 | 0.5 | Less aggressive positive weighting |
| `oversample_cap` | 30% | 15% | Closer to real distribution |
| `oversample_mult` | 5.0× | 2.5× | Reduce training distortion |
| `feature_count` | 37 | 50 | Add fraud-specific interactions |
| `schema_version` | 5.0 | 6.0 | New feature set |

---

## Next Steps if Not Hitting Target

**If AUPRC < 0.65 after changes**:
1. Check LightGBM baseline - if it's also <0.65, the data needs more work
2. Try additional V-features (more PCA components)
3. Experiment with different thresholding strategies

**If AUPRC 0.65-0.70**:
1. Tune learning rates (try 1e-4, 2e-3)
2. Increase local epochs (try 3, 4)
3. Blend with LightGBM predictions

**If AUPRC > 0.70** ✓:
1. Monitor validation on production-like data drift
2. Set up automated retraining pipeline
3. Add confidence calibration for business thresholds

---

## Files Modified

1. `data/load_ieee_cis.py` - Enhanced feature engineering
2. `config/schema.json` - Updated schema to v6.0 with 50 features  
3. `src/client/client.py` - Less aggressive loss/sampling
4. `scripts/lightgbm_baseline.py` - New baseline for validation

**No changes to federated core logic** - All improvements are feature/training tweaks
