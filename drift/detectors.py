import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
from scipy import stats
from pathlib import Path

with open("contracts/schema.json") as f:
    _s = json.load(f)
NUMERIC = [f["name"] for f in _s["feature_schema"]["numeric_features"]]
# 9 features: tx_amount_usd, tx_count_1h, ... account_age_days

@dataclass
class DriftReport:
    timestamp:            str
    feature_psi:          Dict[str, float]
    feature_ks_pval:      Dict[str, float]
    severity:             str
    triggered_features:   List[str]
    score_shift:          Optional[float] = None
    stale_fx_rate:        float = 0.0    # v3 new field

def psi(ref, cur, bins=10):
    eps = 1e-6
    rh, edges = np.histogram(ref, bins=bins)
    ch, _     = np.histogram(cur, bins=edges)
    rp = (rh + eps) / (len(ref) + eps * bins)
    cp = (ch + eps) / (len(cur) + eps * bins)
    return float(np.sum((rp - cp) * np.log(rp / cp)))

class FeatureMonitor:
    PSI_WARN = 0.10
    PSI_CRIT = 0.20

    def __init__(self, reference_df: pd.DataFrame):
        self.ref = reference_df[NUMERIC].copy()
        Path("data/drift_ref").mkdir(exist_ok=True)
        self.ref.to_parquet("data/drift_ref/reference_window.parquet", index=False)

    @classmethod
    def from_disk(cls, path: Path | str = "data/drift_ref/reference_window.parquet") -> "FeatureMonitor":
        reference_path = Path(path)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

        reference_df = pd.read_parquet(reference_path)
        obj = object.__new__(cls)
        obj.ref = reference_df[NUMERIC].copy()
        return obj

    def check(self, current_df: pd.DataFrame) -> DriftReport:
        psi_s, ks_s, triggered = {}, {}, []
        for feat in NUMERIC:
            r = self.ref[feat].dropna().values
            c = current_df[feat].dropna().values
            
            p = psi(r, c)
            # FIX: Access .pvalue explicitly instead of unpacking
            res: Any = stats.ks_2samp(r, c)
            ksp = res.pvalue 
            
            psi_s[feat] = round(p, 5)
            ks_s[feat] = round(float(ksp), 5)
            
            if p > self.PSI_WARN: 
                triggered.append(feat)
                
        mx = max(psi_s.values()) if psi_s else 0.0
        sev = "CRITICAL" if mx >= self.PSI_CRIT else "WARNING" if mx >= self.PSI_WARN else "INFO"
        
        # v3: compute stale_fx_rate from metadata column if present
        stale_rate = 0.0
        if "stale_fx_flag" in current_df.columns:
            stale_rate = float(current_df["stale_fx_flag"].mean())
            
        return DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            feature_psi=psi_s, 
            feature_ks_pval=ks_s,
            severity=sev, 
            triggered_features=triggered,
            stale_fx_rate=round(stale_rate, 4)
        )