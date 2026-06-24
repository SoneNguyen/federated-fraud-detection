"""Generate Vietnam-style synthetic transaction data for federated fraud tests.

The generated file is not real bank data. It is a reproducible, documented
proxy shaped around Vietnamese payment rails: e-wallet checkout, payment
gateway checkout, VietQR/NAPAS-style transfer, bill payment, top-up, and card
payment behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROVINCES = [
    ("Ho Chi Minh", 0.28),
    ("Ha Noi", 0.22),
    ("Da Nang", 0.07),
    ("Binh Duong", 0.06),
    ("Dong Nai", 0.05),
    ("Hai Phong", 0.04),
    ("Can Tho", 0.04),
    ("Khanh Hoa", 0.03),
    ("Quang Ninh", 0.03),
    ("Other", 0.18),
]

PROVIDER_CHANNELS = {
    "momo": ["wallet_payment", "topup", "bill_payment", "ecommerce"],
    "vnpay": ["ecommerce", "card_payment", "qr_transfer", "bill_payment"],
    "napas": ["qr_transfer", "bank_transfer", "card_payment"],
}

CHANNEL_PRODUCT_WEIGHTS = {
    "wallet_payment": 0.24,
    "qr_transfer": 0.22,
    "bank_transfer": 0.14,
    "ecommerce": 0.18,
    "bill_payment": 0.10,
    "topup": 0.07,
    "card_payment": 0.05,
}

BANKS = [
    "VCB",
    "BIDV",
    "CTG",
    "MB",
    "TCB",
    "VPB",
    "ACB",
    "TPB",
    "VIB",
    "MSB",
    "STB",
    "OCB",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "company.vn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Vietnam-style payment transactions.")
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--customers", type=int, default=35_000)
    parser.add_argument("--merchants", type=int, default=4_000)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--output", default="dataset/vietnam_synthetic/transactions.csv")   
    parser.add_argument("--report", default="dataset/vietnam_synthetic/report.json")
    return parser.parse_args()


def _choice(rng: np.random.Generator, values: list[str], probs: list[float], size: int) -> np.ndarray:
    weights = np.array(probs, dtype=np.float64)
    weights = weights / weights.sum()
    return rng.choice(np.array(values, dtype=object), size=size, p=weights)


def _hour_distribution(rng: np.random.Generator, rows: int) -> np.ndarray:
    buckets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    weights = np.array(
        [0.010, 0.007, 0.005, 0.004, 0.004, 0.006, 0.018, 0.035,
         0.050, 0.055, 0.060, 0.065, 0.070, 0.060, 0.055, 0.055,
         0.060, 0.065, 0.075, 0.075, 0.070, 0.050, 0.030, 0.016],
        dtype=np.float64,
    )
    return rng.choice(buckets, size=rows, p=weights / weights.sum())


def _amounts_vnd(rng: np.random.Generator, channels: np.ndarray, providers: np.ndarray) -> np.ndarray:
    rows = len(channels)
    amount = np.zeros(rows, dtype=np.float64)
    channel_specs = {
        "wallet_payment": (11.1, 0.75, 1_000, 8_000_000),
        "topup": (10.7, 0.55, 10_000, 3_000_000),
        "bill_payment": (12.2, 0.65, 20_000, 20_000_000),
        "ecommerce": (12.0, 0.85, 10_000, 50_000_000),
        "qr_transfer": (12.5, 1.05, 10_000, 499_000_000),
        "bank_transfer": (12.9, 1.05, 10_000, 499_000_000),
        "card_payment": (12.0, 0.85, 10_000, 80_000_000),
    }
    for channel, (mean, sigma, low, high) in channel_specs.items():
        mask = channels == channel
        values = rng.lognormal(mean=mean, sigma=sigma, size=int(mask.sum()))
        amount[mask] = np.clip(values, low, high)
    momo_mask = providers == "momo"
    amount[momo_mask] = np.clip(amount[momo_mask], 1_000, 50_000_000)
    return np.round(amount / 1000) * 1000


def _make_ids(prefix: str, indices: np.ndarray) -> np.ndarray:
    return np.char.add(prefix, np.char.zfill(indices.astype(str), 7))


def _build_base_frame(args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    rows = args.rows
    customer_rank = rng.zipf(1.35, size=rows) % args.customers
    merchant_rank = rng.zipf(1.25, size=rows) % args.merchants
    customer_id = _make_ids("CUS", customer_rank)
    merchant_id = _make_ids("MER", merchant_rank)

    providers = _choice(rng, ["momo", "vnpay", "napas"], [0.35, 0.30, 0.35], rows)
    channels = np.empty(rows, dtype=object)
    for provider, provider_channels in PROVIDER_CHANNELS.items():
        mask = providers == provider
        weights = [CHANNEL_PRODUCT_WEIGHTS[channel] for channel in provider_channels]
        channels[mask] = _choice(rng, provider_channels, weights, int(mask.sum()))

    provinces = _choice(rng, [p for p, _ in PROVINCES], [w for _, w in PROVINCES], rows)
    days = rng.integers(0, args.days, size=rows)
    hours = _hour_distribution(rng, rows)
    minutes = rng.integers(0, 60, size=rows)
    seconds = rng.integers(0, 60, size=rows)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timestamp = start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h")
    timestamp = timestamp + pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(seconds, unit="s")

    amount_vnd = _amounts_vnd(rng, channels, providers)
    device_base = rng.integers(0, max(args.customers // 2, 1), size=rows)
    device_noise = rng.random(rows) < 0.12
    device_base[device_noise] = rng.integers(args.customers // 2, args.customers * 2, size=int(device_noise.sum()))

    card_type = np.where(channels == "card_payment", rng.choice(["debit", "credit"], size=rows, p=[0.78, 0.22]), "debit")
    card_brand = np.where(providers == "napas", "napas", rng.choice(["visa", "mastercard", "napas"], size=rows, p=[0.22, 0.18, 0.60]))
    device_type = rng.choice(["mobile", "desktop"], size=rows, p=[0.88, 0.12])
    account_age_days = np.clip(rng.gamma(shape=2.0, scale=220.0, size=rows), 0, 3650).round()

    return pd.DataFrame(
        {
            "transaction_id": [f"VN{idx:012d}" for idx in range(rows)],
            "timestamp": timestamp.astype(str),
            "provider": providers,
            "channel": channels,
            "amount_vnd": amount_vnd.astype("int64"),
            "currency": "VND",
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "payer_bank": rng.choice(BANKS, size=rows),
            "receiver_bank": rng.choice(BANKS, size=rows),
            "device_id": _make_ids("DEV", device_base),
            "device_type": device_type,
            "province": provinces,
            "card_type": card_type,
            "card_brand": card_brand,
            "payer_email": np.char.add(np.char.add(customer_id.astype(str), "@"), rng.choice(EMAIL_DOMAINS, size=rows)),
            "receiver_email": np.char.add(np.char.add(merchant_id.astype(str), "@merchant."), "vn"),
            "account_age_days": account_age_days.astype("int64"),
        }
    ).sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def _add_velocity_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    seconds = ts.astype("int64") // 1_000_000_000
    counts_1h = np.zeros(len(frame), dtype=np.int16)
    counts_24h = np.zeros(len(frame), dtype=np.int16)
    days_since_prev = np.ones(len(frame), dtype=np.float32)

    helper = pd.DataFrame({"idx": np.arange(len(frame)), "customer_id": frame["customer_id"], "seconds": seconds})
    for _, group in helper.groupby("customer_id", sort=False):
        idx = group["idx"].to_numpy()
        t = group["seconds"].to_numpy(dtype=np.int64)
        left_1h = 0
        left_24h = 0
        for pos, current in enumerate(t):
            while left_1h < pos and current - t[left_1h] > 3600:
                left_1h += 1
            while left_24h < pos and current - t[left_24h] > 86400:
                left_24h += 1
            counts_1h[idx[pos]] = pos - left_1h
            counts_24h[idx[pos]] = pos - left_24h
            if pos > 0:
                days_since_prev[idx[pos]] = max((current - t[pos - 1]) / 86400.0, 0.01)

    frame["transactions_1h"] = counts_1h
    frame["transactions_24h"] = counts_24h
    frame["days_since_last_transaction"] = days_since_prev
    frame["distance_km"] = np.where(
        frame["province"].eq(frame["province"].shift()).fillna(True),
        0.0,
        np.random.default_rng(12345).uniform(20, 1200, size=len(frame)),
    ).round(2)
    return frame


def _label_fraud(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frame = frame.copy()
    hour = pd.to_datetime(frame["timestamp"], utc=True).dt.hour.to_numpy()
    amount = frame["amount_vnd"].to_numpy(dtype=np.float64)
    velocity_1h = frame["transactions_1h"].to_numpy(dtype=np.float64)
    velocity_24h = frame["transactions_24h"].to_numpy(dtype=np.float64)
    account_age = frame["account_age_days"].to_numpy(dtype=np.float64)
    distance = frame["distance_km"].to_numpy(dtype=np.float64)

    risky_merchant = frame["merchant_id"].isin(frame["merchant_id"].value_counts().tail(max(len(frame) // 800, 25)).index).to_numpy()
    late_night = (hour <= 5) | (hour >= 23)
    high_amount = amount > np.quantile(amount, 0.94)
    very_high_amount = amount > np.quantile(amount, 0.985)
    young_account = account_age < 30
    burst = (velocity_1h >= 3) | (velocity_24h >= 9)
    far_jump = distance > 300
    qr_or_transfer = frame["channel"].isin(["qr_transfer", "bank_transfer"]).to_numpy()
    wallet_or_topup = frame["channel"].isin(["wallet_payment", "topup"]).to_numpy()

    logit = np.full(len(frame), -6.15, dtype=np.float64)
    logit += 1.25 * late_night
    logit += 1.10 * high_amount
    logit += 1.20 * very_high_amount
    logit += 0.95 * young_account
    logit += 1.15 * burst
    logit += 1.00 * far_jump
    logit += 0.90 * risky_merchant
    logit += 0.90 * (qr_or_transfer & high_amount)
    logit += 0.70 * (wallet_or_topup & burst)
    logit += 1.00 * ((frame["provider"].to_numpy() == "napas") & qr_or_transfer & late_night)
    logit += 0.75 * ((frame["provider"].to_numpy() == "momo") & wallet_or_topup & young_account)
    logit += rng.normal(0.0, 0.35, size=len(frame))

    probability = 1.0 / (1.0 + np.exp(-logit))
    frame["fraud_probability"] = probability.round(6)
    frame["is_fraud"] = (rng.random(len(frame)) < probability).astype("int8")
    frame["chargeback_count"] = np.where(frame["is_fraud"].eq(1), rng.poisson(0.35, len(frame)), rng.poisson(0.02, len(frame)))
    return frame


def _write_report(frame: pd.DataFrame, args: argparse.Namespace, output_path: Path, report_path: Path) -> None:
    report = {
        "dataset_name": "vietnam_synthetic_payments",
        "rows": int(len(frame)),
        "fraud_rate": float(frame["is_fraud"].mean()),
        "output": str(output_path),
        "mapping": "config/vietnam_synthetic_mapping.json",
        "seed": args.seed,
        "notes": [
            "Synthetic only: no real customer, bank, wallet, or merchant data.",
            "Amounts are VND and are converted by mapping at a fixed 1 USD = 25,000 VND assumption.",
            "Behavior is shaped around MoMo wallet checkout, VNPAY-style gateway checkout, and NAPAS/VietQR-style transfers.",
        ],
        "by_provider": frame.groupby("provider")["is_fraud"].agg(["count", "mean"]).round(6).to_dict("index"),
        "by_channel": frame.groupby("channel")["is_fraud"].agg(["count", "mean"]).round(6).to_dict("index"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def generate(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    frame = _build_base_frame(args, rng)
    frame = _add_velocity_features(frame)
    frame = _label_fraud(frame, rng)
    return frame


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(args)
    frame.to_csv(output_path, index=False)
    _write_report(frame, args, output_path, report_path)
    print(f"Saved {len(frame):,} rows to {output_path}")
    print(f"Fraud rate: {frame['is_fraud'].mean() * 100:.2f}%")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
