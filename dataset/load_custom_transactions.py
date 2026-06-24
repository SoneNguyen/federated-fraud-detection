"""Prepare a mapped local transaction dataset for federated fraud training."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset.custom_transaction_adapter import load_mapping, load_table, to_ieee_like_frame
from dataset.load_ieee_cis import engineer, write_processed_clients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map an anonymized transaction CSV/parquet into the active 328-feature "
            "fraud schema and write federated client parquet files."
        )
    )
    parser.add_argument("--input", required=True, help="Source CSV or parquet file.")
    parser.add_argument("--mapping", required=True, help="Mapping JSON file.")
    parser.add_argument("--output-root", default="dataset/processed_custom")
    parser.add_argument("--normalization-output", default="config/normalization_params.json")
    parser.add_argument("--num-clients", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    dataset_name = str(mapping.get("dataset_name", "custom-transactions"))
    source_name = str(mapping.get("source", args.input))

    print(f"Loading custom transaction dataset: {args.input}")
    raw = load_table(args.input)
    print(f"Source rows={len(raw):,} columns={len(raw.columns):,}")

    print("Mapping source columns into fraud transaction contract...")
    ieee_like = to_ieee_like_frame(raw, mapping)
    print(f"Mapped rows={len(ieee_like):,} fraud={ieee_like['isFraud'].mean() * 100:.2f}%")

    print("Engineering active fraud-history feature schema...")
    featured = engineer(ieee_like, ieee_like)
    print(f"Feature matrix={featured.shape}")

    write_processed_clients(
        featured,
        output_root=Path(args.output_root),
        num_clients=args.num_clients,
        normalization_path=Path(args.normalization_output),
        dataset_name=dataset_name,
        source=source_name,
    )
    print("\n[COMPLETE] Custom transaction data pipeline complete.")


if __name__ == "__main__":
    main()
