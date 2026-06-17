"""Active feature contract for fraud training and inference."""

from __future__ import annotations

SCHEMA_VERSION = "fraud-history"
LABEL = "is_fraud"

CORE_FEATURES = [
    "tx_amount_usd",
    "tx_count_1h",
    "tx_count_24h",
    "tx_volume_1h_usd",
    "tx_volume_24h_usd",
    "geo_velocity_kmh",
    "dist2_km",
    "amount_x_velocity",
    "amount_per_tx_1h",
    "amount_per_tx_24h",
    "spending_velocity_1h",
    "card6_debit",
    "card6_credit",
    "card6_charge_card",
    "card6_debit_or_credit",
    "days_since_last_tx",
    "account_age_days",
    "hour_of_day_local",
    "day_of_week",
    "tx_time_norm",
    "week_of_period",
    "risky_hour_flag",
    "early_morning_high_value",
    "weekend_high_value",
    "prod_W",
    "prod_H",
    "prod_C",
    "prod_S",
    "prod_R",
    "card1_norm",
    "card2_norm",
    "addr1_norm",
    "addr2_norm",
    "V258",
    "V257",
    "V201",
    "c5_chargeback",
    "email_domain_match",
    "p_email_free",
    "r_email_free",
    "both_emails_free",
    "email_mismatch_high_value",
    "card3_norm",
    "card4_code",
    "has_device_info",
    "card_device_mismatch",
    "new_account_high_value",
]

C_LOG_FEATURES = [f"C{i}_log" for i in range(3, 15)]
D_BASE_COLUMNS = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
D_FEATURES = [f"D{i}_log" for i in D_BASE_COLUMNS] + [f"D{i}_missing" for i in D_BASE_COLUMNS]

FREQ_FEATURES = [
    "card1_freq",
    "card2_freq",
    "card5_freq",
    "addr1_freq",
    "p_email_freq",
    "r_email_freq",
    "device_info_freq",
]

HISTORY_ENTITY_PREFIXES = [
    "card1",
    "card2",
    "addr1",
    "p_email",
    "r_email",
    "device_info",
    "card_pair",
    "email_pair",
]
HISTORY_FEATURES = [
    feature
    for prefix in HISTORY_ENTITY_PREFIXES
    for feature in (
        f"hist_{prefix}_count_log",
        f"hist_{prefix}_since_prev_log",
        f"hist_{prefix}_amount_mean_log",
        f"hist_{prefix}_amount_ratio",
        f"hist_{prefix}_fraud_count_log",
        f"hist_{prefix}_fraud_rate",
        f"hist_{prefix}_since_prev_fraud_log",
    )
]

ID_NUMERIC_COLUMNS = [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 17, 19, 20]
ID_FEATURES = [f"id_{i:02d}_norm" for i in ID_NUMERIC_COLUMNS] + [
    f"id_{i:02d}_missing" for i in ID_NUMERIC_COLUMNS
]
ID_FLAG_FEATURES = [
    "device_type_desktop",
    "device_type_mobile",
    "id_12_found",
    "id_15_found",
    "id_16_found",
    "id_28_found",
    "id_29_found",
]

SELECTED_V_COLUMNS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20,
    29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 44, 45, 46, 47, 48, 49,
    53, 54, 55, 56, 57, 58, 61, 62, 70,
    75, 76, 77, 78, 81, 82, 83, 87,
    95, 96, 97, 99,
    126, 127, 128, 130, 131, 132, 133, 134, 136, 137,
    143, 149, 150, 151, 152, 156, 159, 160, 164, 165,
    170, 171, 187, 188, 189, 190,
    202, 203, 204, 205, 206, 207, 208, 209, 210,
    221, 222, 243, 244, 259, 260, 261, 262, 263, 264, 265,
    266, 267, 268, 271, 272, 273, 274, 275, 276, 277, 278,
    283, 285, 291, 294, 303, 304, 306, 307, 308, 310, 312,
    313, 314, 315, 316, 317, 318, 320, 321,
]
V_FEATURES = [f"V{i}" for i in SELECTED_V_COLUMNS if f"V{i}" not in CORE_FEATURES]

FEATURE_ORDER = (
    CORE_FEATURES
    + C_LOG_FEATURES
    + D_FEATURES
    + FREQ_FEATURES
    + HISTORY_FEATURES
    + ID_FEATURES
    + ID_FLAG_FEATURES
    + V_FEATURES
)

BINARY_FEATURES = {
    name
    for name in FEATURE_ORDER
    if name.startswith("prod_")
    or name.startswith("card6_")
    or name.endswith("_missing")
    or name.endswith("_flag")
    or name.startswith("id_") and name.endswith("_found")
    or name.startswith("device_type_")
    or name
    in {
        "early_morning_high_value",
        "weekend_high_value",
        "email_domain_match",
        "p_email_free",
        "r_email_free",
        "both_emails_free",
        "email_mismatch_high_value",
        "has_device_info",
        "card_device_mismatch",
        "new_account_high_value",
    }
}

assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), "Duplicate feature names"
