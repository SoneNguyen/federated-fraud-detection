import numpy as np
import pandas as pd

from drift.detectors import NUMERIC, FeatureMonitor


def make_df(n: int = 10_000, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({feature: rng.normal(shift, 1.0, n) for feature in NUMERIC})


def make_stale_df(n: int = 10_000, stale_frac: float = 0.8, seed: int = 5) -> pd.DataFrame:
    df = make_df(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    df["stale_fx_flag"] = (rng.random(n) < stale_frac).astype(int)
    return df


def test_feature_monitor_reports_info_without_distribution_shift(tmp_path):
    monitor = FeatureMonitor(make_df(seed=0), persist_path=tmp_path / "reference.parquet")

    report = monitor.check(make_df(seed=99))

    assert report.severity == "INFO"
    assert report.stale_fx_rate == 0.0


def test_feature_monitor_reports_critical_for_severe_shift(tmp_path):
    monitor = FeatureMonitor(make_df(seed=0), persist_path=tmp_path / "reference.parquet")
    severe = make_df(seed=2)
    severe["tx_amount_usd"] = np.random.default_rng(2).normal(-3.0, 0.5, 10_000)
    severe["tx_volume_1h_usd"] = np.random.default_rng(3).normal(4.0, 0.5, 10_000)

    report = monitor.check(severe)

    assert report.severity == "CRITICAL"
    assert "tx_amount_usd" in report.triggered_features


def test_feature_monitor_reports_stale_fx_rate(tmp_path):
    monitor = FeatureMonitor(make_df(seed=0), persist_path=tmp_path / "reference.parquet")

    report = monitor.check(make_stale_df(stale_frac=0.8))

    assert report.stale_fx_rate > 0.7
