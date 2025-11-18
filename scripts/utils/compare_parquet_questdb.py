"""
Compare Parquet files vs QuestDB contents.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.data.storage.questdb import QuestDBStorage


def compare_parquet_vs_questdb():
    """Compare Parquet file counts vs QuestDB counts."""
    input_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    parquet_files = sorted(input_dir.glob('**/*.parquet'))

    storage = QuestDBStorage()

    print("=" * 90)
    print("🔍 COMPARISON: PARQUET FILES VS QUESTDB")
    print("=" * 90)
    print(f"{'Symbol':<12} {'TF':<6} {'Parquet':>12} {'QuestDB':>12} {'Match':>10}")
    print("-" * 90)

    all_match = True

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        symbol = pf.parent.name.replace('_', '/')
        timeframe = pf.stem
        exchange = 'binance'

        parquet_count = len(df)
        questdb_count = storage.count_candles(symbol, timeframe, exchange)

        match = "✅" if parquet_count == questdb_count else "❌"

        if parquet_count != questdb_count:
            all_match = False

        print(f"{symbol:<12} {timeframe:<6} {parquet_count:>12,} {questdb_count:>12,} {match:>10}")

    print("-" * 90)

    if all_match:
        print("✅ All files match QuestDB!")
    else:
        print("❌ Some mismatches found. Re-run: python scripts/2_parquet_to_questdb.py")

    print("=" * 90)

    storage.close()


if __name__ == "__main__":
    compare_parquet_vs_questdb()