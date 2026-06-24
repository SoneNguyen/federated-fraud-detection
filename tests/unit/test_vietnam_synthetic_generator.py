import argparse

from dataset.generate_vietnam_synthetic import generate


def test_vietnam_synthetic_generator_outputs_mapped_contract_fields():
    args = argparse.Namespace(
        rows=1000,
        customers=150,
        merchants=40,
        days=10,
        seed=7,
        output="unused.csv",
        report="unused.json",
    )

    frame = generate(args)

    required = {
        "transaction_id",
        "timestamp",
        "amount_vnd",
        "channel",
        "customer_id",
        "merchant_id",
        "device_id",
        "province",
        "transactions_1h",
        "transactions_24h",
        "is_fraud",
    }
    assert required.issubset(frame.columns)
    assert len(frame) == 1000
    assert frame["amount_vnd"].min() >= 1000
    assert set(frame["is_fraud"].unique()).issubset({0, 1})
    assert frame["is_fraud"].sum() > 0
