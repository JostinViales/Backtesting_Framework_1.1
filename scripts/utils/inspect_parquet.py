"""
Inspect Parquet files utility.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd


def inspect_parquet_files():
    """Show summary of all Parquet files."""
    input_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'

    parquet_files = sorted(input_dir.glob('**/*.parquet'))

    print("=" * 80)
    print("📊 PARQUET FILES SUMMARY")
    print("=" * 80)
    print(f"{'Symbol':<12} {'TF':<6} {'Rows':>12} {'Size':>10} {'Date Range':<40}")
    print("-" * 80)

    total_rows = 0
    total_size = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        size_mb = pf.stat().st_size / (1024 ** 2)

        symbol = pf.parent.name.replace('_', '/')
        timeframe = pf.stem
        date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"

        print(f"{symbol:<12} {timeframe:<6} {len(df):>12,} {size_mb:>9.2f}M {date_range:<40}")

        total_rows += len(df)
        total_size += size_mb

    print("-" * 80)
    print(f"{'TOTAL':<12} {'':<6} {total_rows:>12,} {total_size:>9.2f}M")
    print("=" * 80)


if __name__ == "__main__":
    inspect_parquet_files()