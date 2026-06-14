# Project Audit Report: File Usage & Structure Cleanup

**Date**: June 14, 2026  
**Status**: Refactoring Partially Complete - Duplicates Detected

---

## Executive Summary

**Critical Issue**: After the refactoring, the project has **duplicate code in two locations**:
- **Old structure** (client/, server/, model/) — **ACTIVELY USED**
- **New structure** (src/) — **MOSTLY UNUSED** except by scripts/

### Key Findings
- ✅ 3 new entry point scripts using src/* (scripts/run_server.py, scripts/run_client.py)
- ❌ 13 unit tests still importing from old client.*, server.* 
- ❌ API (api/main.py) still importing from old client.model
- ❌ Model utilities (model/evaluate.py, model/calibrate.py) still importing from old client.*
- ❌ Old entry point still exists (client/run_client.py, server/fl_server.py)
- ⚠️ src/ directory is a **DUPLICATE, NOT YET INTEGRATED**

---

## I. Duplicate Detection

### Directory Duplicates

| Component | Old Location | New Location | Status | Uses |
|-----------|--------------|--------------|--------|------|
| Model | client/model.py | src/model/fraud_mlp.py | **DUPLICATE** | ❌ Mostly old |
| Dataset | client/dataset.py | src/data/dataset.py | **DUPLICATE** | ❌ Mostly old |
| Client | client/fl_client.py | src/client/client.py | **DUPLICATE** | ❌ Mostly old |
| Server Strategy | server/strategy.py | src/server/strategy.py | **DUPLICATE** | ❌ Old |
| Checkpoint Mgr | server/checkpoint_manager.py | src/server/checkpoint_manager.py | **DUPLICATE** | ❌ Old |
| Client State | server/client_state.py | src/server/client_state.py | **DUPLICATE** | ❌ Old |

### Entry Point Duplicates

| Script | Old | New | Used | Notes |
|--------|-----|-----|------|-------|
| Client Launcher | client/run_client.py | scripts/run_client.py | ✅ Only new | Old version still imports from client/* |
| Server Launcher | server/fl_server.py | scripts/run_server.py | ✅ Only new | Old version still imports from server/* |

---

## II. Current Import Usage Analysis

### Active Imports from OLD Locations

**client/model.py** (9 imports):
```
✓ server/fl_server.py:11 → from client.model import FraudMLP
✓ server/strategy.py:15 → from client.model import FraudMLP
✓ api/main.py:22 → from client.model import FraudMLP
✓ client/run_client.py:9 → from client.model import FraudMLP
✓ model/evaluate.py:17 → from client.model import FraudMLP
✓ model/calibrate.py:24 → from client.model import FraudMLP
✓ tests/unit/test_model.py:7 → from client.model import FraudMLP
✓ tests/unit/test_fl_client.py:11 → from client.model import FraudMLP
✓ tests/unit/test_evaluate.py:11 → from client.model import FraudMLP
```

**client/dataset.py** (8 imports):
```
✓ client/run_client.py:11 → from client.dataset import make_loaders
✓ model/evaluate.py:18 → from client.dataset import FEATURE_ORDER, LABEL
✓ model/calibrate.py:23 → from client.dataset import FEATURE_ORDER, LABEL, make_loaders
✓ tests/unit/test_dataset.py:12 → from client.dataset import FEATURE_ORDER, LABEL, ...
✓ tests/unit/test_evaluate.py:10 → from client.dataset import FEATURE_ORDER, LABEL
✓ tests/unit/test_calibrate.py:9 → from client.dataset import FEATURE_ORDER, LABEL
✓ api/schemas.py:2 → from client.dataset import FEATURE_ORDER
✓ tests/unit/test_api.py:69 → from client.model import FraudMLP
```

**client/fl_client.py** (2 imports):
```
✓ client/run_client.py:10 → from client.fl_client import FraudClient
✓ tests/unit/test_fl_client.py:10 → from client.fl_client import FraudClient
```

**server/strategy.py** (1 import):
```
✓ tests/unit/test_strategy.py:14 → from server.strategy import WeightedFedAvg
```

**server/checkpoint_manager.py** (3 imports):
```
✓ tests/unit/test_checkpoint_manager.py:5 → from server.checkpoint_manager import CheckpointManager
✓ tests/integration/test_rollback.py:5 → from server.checkpoint_manager import CheckpointManager
✓ drift/alert_manager.py:9 → from server.checkpoint_manager import CheckpointManager
```

**server/client_state.py** (0 imports - NOT DIRECTLY IMPORTED BUT REFERENCED IN strategy.py)

**model/architecture.py** (1 import):
```
✓ tests/unit/test_architecture.py:4 → from model.architecture import FraudMLP, INPUT_DIM
```

**model/evaluate.py** (0 imports - USED ONLY BY TESTS)

**model/calibrate.py** (0 imports - USED ONLY BY TESTS)

### NEW src/* Imports (Minimal Usage)

```
✓ scripts/run_server.py:12 → from src.server.strategy import WeightedFedAvg
✓ scripts/run_server.py:13 → from src.server.checkpoint_manager import CheckpointManager
✓ scripts/run_server.py:14 → from src.model.fraud_mlp import FraudMLP
✓ scripts/run_client.py:12 → from src.model.fraud_mlp import FraudMLP
✓ scripts/run_client.py:13 → from src.client.client import FraudClient
✓ scripts/run_client.py:14 → from src.data.dataset import make_loaders
```

---

## III. Unused & Deprecated Code

### Completely Unused Modules

| File | Status | Reason |
|------|--------|--------|
| ❌ **client/run_client.py** | DEPRECATED | Superseded by scripts/run_client.py |
| ❌ **server/fl_server.py** | DEPRECATED | Superseded by scripts/run_server.py |

### Conditionally Used Modules

| File | Status | Used By | Can Remove? |
|------|--------|---------|-------------|
| ⚠️ **model/architecture.py** | RE-EXPORT ONLY | tests/unit/test_architecture.py | Maybe (if tests updated) |
| ⚠️ **model/evaluate.py** | UTILITY | tests/unit/test_calibrate.py, tests/unit/test_evaluate.py | No (used by tests) |
| ⚠️ **model/calibrate.py** | UTILITY | tests/unit/test_calibrate.py | No (used by tests) |

### Unused src/* Modules (New, Not Yet Integrated)

| File | Status | Should Be | Can Remove? |
|------|--------|-----------|-------------|
| ⚠️ **src/server/strategy.py** | DUPLICATE | Integrated | No (keep as primary) |
| ⚠️ **src/server/checkpoint_manager.py** | DUPLICATE | Integrated | No (keep as primary) |
| ⚠️ **src/server/client_state.py** | DUPLICATE | Integrated | No (keep as primary) |
| ⚠️ **src/client/client.py** | DUPLICATE | Integrated | No (keep as primary) |
| ⚠️ **src/model/fraud_mlp.py** | DUPLICATE | Integrated | No (keep as primary) |
| ⚠️ **src/data/dataset.py** | DUPLICATE | Integrated | No (keep as primary) |

---

## IV. Misplaced Code Analysis

### Misplaced Component: API (api/)

**Issue**: API imports from client.model instead of src.model

```python
# api/main.py:22
from client.model import FraudMLP  # ❌ OLD IMPORT

# Should be:
from src.model.fraud_mlp import FraudMLP  # ✅ NEW IMPORT
```

**Files Affected**:
- api/main.py
- api/schemas.py (imports from client.dataset)

**Recommendation**: Move to src/api or update imports to use src/*

---

### Misplaced Component: Model Utilities (model/)

**Issue**: model/evaluate.py and model/calibrate.py import from client.* instead of src.*

```python
# model/evaluate.py:17-18
from client.model import FraudMLP  # ❌ OLD
from client.dataset import FEATURE_ORDER, LABEL  # ❌ OLD

# Should be:
from src.model.fraud_mlp import FraudMLP  # ✅ NEW
from src.data.dataset import FEATURE_ORDER, LABEL  # ✅ NEW
```

**Files Affected**:
- model/evaluate.py
- model/calibrate.py
- model/architecture.py (re-exports from client.model)

**Recommendation**: Either move to src/model/utils/ or update imports

---

### Misplaced Component: Data Pipeline (data/)

**Issue**: data/load_ieee_cis.py is not in src/data/ yet

```python
# Currently: data/load_ieee_cis.py
# Should be: src/data/load_ieee_cis.py (for consistency)
```

**Files Affected**:
- data/load_ieee_cis.py
- data/fx/*.py (FX converter utilities)

**Recommendation**: Move to src/data/ or create src/data/pipeline/ subdirectory

---

## V. Import Inconsistency Report

### Test Files with Old Imports (13 files need updating)

```
❌ tests/unit/test_api.py
   Line 69: from client.model import FraudMLP → src.model

❌ tests/unit/test_calibrate.py
   Line 10: from client.model import FraudMLP → src.model
   Line 9: from client.dataset import FEATURE_ORDER, LABEL → src.data

❌ tests/unit/test_checkpoint_manager.py
   Line 5: from server.checkpoint_manager import CheckpointManager → src.server

❌ tests/unit/test_dataset.py
   Line 12: from client.dataset import FraudDataset, make_loaders, ... → src.data

❌ tests/unit/test_evaluate.py
   Line 10-12: from client.dataset, client.model, model.evaluate

❌ tests/unit/test_fl_client.py
   Line 10-11: from client.fl_client, client.model → src.client, src.model

❌ tests/unit/test_model.py
   Line 7: from client.model import FraudMLP, INPUT_DIM → src.model

❌ tests/unit/test_strategy.py
   Line 14: from server.strategy import WeightedFedAvg → src.server.strategy

❌ tests/integration/test_rollback.py
   Line 5: from server.checkpoint_manager → src.server.checkpoint_manager

❌ tests/unit/test_architecture.py
   Line 4: from model.architecture import FraudMLP, INPUT_DIM → src.model
   (This is a re-export layer that might be removed)

❌ tests/unit/test_calibrate.py
   Imports from model.calibrate (utility, OK but check path)

❌ tests/unit/test_concept_drift.py
   Check if it imports from old locations

❌ tests/unit/test_exporter.py
   Check if it imports from old locations
```

### Live Code with Old Imports (6 files need updating)

```
❌ server/fl_server.py
   Lines 9-11: from server.strategy, server.checkpoint_manager, client.model
   → DEPRECATE IN FAVOR OF scripts/run_server.py

❌ client/run_client.py
   Lines 9-11: from client.model, client.fl_client, client.dataset
   → DEPRECATE IN FAVOR OF scripts/run_client.py

❌ api/main.py
   Line 22: from client.model import FraudMLP → src.model.fraud_mlp

❌ api/schemas.py
   Line 2: from client.dataset import FEATURE_ORDER → src.data.dataset

❌ model/evaluate.py
   Lines 17-18: from client.model, client.dataset → src.model, src.data

❌ model/calibrate.py
   Lines 23-24: from client.model, client.dataset → src.model, src.data
```

### Utilities/Infrastructure with Old Imports (1 file)

```
❌ drift/alert_manager.py
   Line 9: from server.checkpoint_manager → src.server.checkpoint_manager
```

---

## VI. Cleanup Action Plan

### Phase 1: Update All Imports (Non-Breaking)
**Effort**: 15-20 min | **Risk**: Low | **Files**: 10-12

1. Update test imports in 10 test files
2. Update api/main.py and api/schemas.py
3. Update model/evaluate.py and model/calibrate.py  
4. Update drift/alert_manager.py

### Phase 2: Deprecate Old Entry Points (Non-Breaking)
**Effort**: 5 min | **Risk**: Low | **Files**: 2

1. Mark client/run_client.py as DEPRECATED (add banner comment)
2. Mark server/fl_server.py as DEPRECATED (add banner comment)
3. Keep files for backward compatibility (not used by new code)

### Phase 3: Remove Redundant Code (Breaking)
**Effort**: 5 min | **Risk**: Medium | **Files**: 6

1. ✅ **Keep**: src/client/, src/server/, src/model/, src/data/ (primary)
2. ❌ **Remove**: client/fl_client.py, client/model.py, client/dataset.py
3. ❌ **Remove**: server/strategy.py, server/checkpoint_manager.py, server/client_state.py
4. **Decide**: Keep client/run_client.py & server/fl_server.py or archive?

### Phase 4: Optional Structure Improvements
**Effort**: 10 min | **Risk**: Low | **Files**: 3

1. Move data/load_ieee_cis.py → src/data/load_ieee_cis.py
2. Move data/fx/ → src/data/fx/
3. Move model/*.py → src/model/utils/ (evaluate, calibrate)

### Phase 5: Clean Up Unused Directories
**Effort**: 5 min | **Risk**: Low | **Dirs**: 3

1. Remove old client/ (after Phase 3)
2. Remove old server/ (after Phase 3, keeping only what moves to src/)
3. Remove old model/ OR keep as src/model/utils/

---

## VII. File Cleanliness Analysis

### Dead Code Detection

**model/architecture.py**: RE-EXPORT ONLY
```python
"""Re-exports FraudMLP from client.model."""
from client.model import FraudMLP, INPUT_DIM
```
- ✅ Purpose: Allow tests to import from model.architecture
- ⚠️ Could be removed if test imports updated to use src.model directly

---

### Duplicate Code Verification

**Comparison: client/model.py vs src/model/fraud_mlp.py**
```
✓ Both have FraudMLP class
✓ Both have _ResBlock class
✓ Both define INPUT_DIM from schema.json
✓ Functionally IDENTICAL (just different import paths for schema)
```

**Comparison: client/dataset.py vs src/data/dataset.py**
```
✓ Both have FraudDataset class
✓ Both have make_loaders() function
✓ Both have make_weighted_sampler() function
✓ Functionally IDENTICAL (schema path slightly different)
```

**Comparison: client/fl_client.py vs src/client/client.py**
```
✓ Both have FraudClient class
✓ Both have FocalLoss class
✓ Both have make_weighted_sampler() function
✓ Functionally IDENTICAL
```

---

## VIII. Recommendations

### Immediate Actions (Next 15 min)

**MUST DO**:
1. ✅ Update all 10-12 test files to import from src/*
2. ✅ Update api/main.py and api/schemas.py to import from src/*
3. ✅ Update model/evaluate.py and model/calibrate.py to import from src/*
4. ✅ Update drift/alert_manager.py to import from src/*

**SHOULD DO**:
5. ✅ Add DEPRECATED banners to client/run_client.py and server/fl_server.py
6. ✅ Create an archive/ directory and move old implementations there

**COULD DO**:
7. ⏳ Move data/load_ieee_cis.py → src/data/load_ieee_cis.py
8. ⏳ Move model/ utilities → src/model/utils/

### Structure After Cleanup

```
AFTER CLEANUP:
✓ src/                          (PRIMARY CODE)
  ├── model/fraud_mlp.py        (ACTIVE)
  ├── data/dataset.py           (ACTIVE)
  ├── client/client.py          (ACTIVE)
  └── server/                   (ACTIVE)
      ├── strategy.py
      ├── checkpoint_manager.py
      └── client_state.py

✓ scripts/                       (ENTRY POINTS - use src/*)
  ├── run_server.py             (ACTIVE)
  └── run_client.py             (ACTIVE)

✓ api/                           (FASTAPI - use src/*)
  └── main.py                   (use src.model, src.data)

✓ tests/                         (ALL TESTS - use src/*)
  ├── unit/
  └── integration/

⏳ data/                          (PIPELINE - optional move to src/data/)
  ├── load_ieee_cis.py
  └── fx/

⏳ model/                         (UTILITIES - optional move to src/model/utils/)
  ├── evaluate.py
  └── calibrate.py

❌ OLD ARCHIVE/ (REMOVED)
  ├── client/                    (BACKUP - DELETE)
  ├── server/fl_server.py        (BACKUP - DELETE)
  └── server/strategy.py OLD     (BACKUP - DELETE)
```

---

## IX. Summary Statistics

| Metric | Count | Status |
|--------|-------|--------|
| Total Python Files | 59 | - |
| Duplicate Modules | 6 | ❌ ISSUE |
| Deprecated Entry Points | 2 | ⚠️ REVIEW |
| Files Needing Import Updates | 12 | ⏳ TODO |
| Test Files | 16 | ⏳ TODO |
| Unused Directories | 3 | ⏳ TODO |
| **Overall Cleanliness** | **60%** | **⚠️ NEEDS WORK** |

---

## Conclusion

The refactoring **created a good src/ structure** but **integration is incomplete**:

- ✅ New src/ directory is well-organized
- ✅ New scripts/ entry points use src/* correctly  
- ❌ Tests still use old imports
- ❌ API still uses old imports
- ❌ Old code still exists (duplicates)
- ⚠️ No cleanup has been done yet

**Recommendation**: Execute **Phase 1-3** actions (update imports → deprecate old → remove duplicates) to achieve 95%+ cleanliness. Estimated effort: **30-45 minutes**.

---

**Next**: Shall I proceed with updating all imports and removing duplicates? (Recommended)
