# client/dataset.py
# simple data loader for the fraud detection model
import torch, pandas as pd, json
from torch.utils.data import Dataset, DataLoader

with open("contracts/schema.json") as f:
    _s = json.load(f)
FEATURE_ORDER = _s["feature_schema"]["feature_order"]     # 11 items
LABEL         = _s["feature_schema"]["label"]["name"]     # "is_fraud"
PASS_ONLY     = [p["name"] for p in _s["feature_schema"]["passthrough_only"]]
# = ["orig_currency","stale_fx_flag"]

# Hard assertions — fail loudly at import time
assert len(FEATURE_ORDER) == 11
assert LABEL not in FEATURE_ORDER
for p in PASS_ONLY:
    assert p not in FEATURE_ORDER, f"{p} must not be in feature_order"

class FraudDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_parquet(path)
        missing = [c for c in FEATURE_ORDER + [LABEL] if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"
        self.X    = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
        self.y    = torch.tensor(df[LABEL].values,         dtype=torch.float32)
        # keep extra columns for drift checks, not model input
        self.meta = df[PASS_ONLY].copy() if all(c in df.columns for c in PASS_ONLY) else None
        print(f"Loaded {len(df):,} rows | fraud={df[LABEL].mean()*100:.2f}% | X.shape={self.X.shape}")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loaders(path, val_split=0.15, batch_size=256, seed=42):
    ds = FraudDataset(path)
    n_val = int(len(ds)*val_split)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds)-n_val, n_val],
        generator=torch.Generator().manual_seed(seed))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2))

if __name__ == "__main__":
    # quick check to verify loader shape and labels
    for i in range(3):
        p = f"data/processed/client_{i}/transactions_normalized.parquet"
        tl, vl = make_loaders(p)
        X, y = next(iter(tl))
        assert X.shape[1] == 11, f"Bad feature count: {X.shape[1]}"
        assert set(y.unique().tolist()).issubset({0.0,1.0})
        print(f"Client {i}: batch X={X.shape}, y={y.shape} — OK")