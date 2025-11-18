"""
Stage 2: Load Parquet files into QuestDB.

This script:
- Reads Parquet files from data/raw/
- Loads into QuestDB with batching
- Verifies each insert
- Can be run multiple times safely
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.storage.questdb import QuestDBStorage
import time


def load_parquet_to_questdb():
    """
    Load all Parquet files into QuestDB.
    """
    # Input directory
    input_dir = Path(__file__).parent.parent / 'data' / 'raw'

    print("=" * 70)
    print("📤 STAGE 2: LOAD PARQUET FILES TO QUESTDB")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print("=" * 70)
    print()

    # Find all Parquet files
    parquet_files = list(input_dir.glob('**/*.parquet'))

    if not parquet_files:
        print("❌ No Parquet files found!")
        print(f"   Expected location: {input_dir}")
        print("   Run: python scripts/1_download_to_parquet.py first")
        return

    print(f"📁 Found {len(parquet_files)} Parquet files")
    print()

    # Sort files for predictable order
    parquet_files.sort()

    total_files = len(parquet_files)
    current_file = 0
    total_inserted = 0

    for parquet_file in parquet_files:
        current_file += 1

        # Extract symbol and timeframe from path
        # Example: data/raw/BTC_USDT/5m.parquet
        symbol = parquet_file.parent.name.replace('_', '/')
        timeframe = parquet_file.stem

        print(f"\n{'=' * 70}")
        print(f"[{current_file}/{total_files}] {symbol} {timeframe}")
        print(f"File: {parquet_file.name}")
        print(f"{'=' * 70}")

        # Create new storage connection for each file
        storage = QuestDBStorage()

        try:
            # Read Parquet file
            print(f"📂 Reading Parquet file...")
            df = pd.read_parquet(parquet_file)

            print(f"📦 Loaded {len(df):,} rows from file")

            # Get exchange from dataframe (or default to binance)
            exchange = df['exchange'].iloc[0] if 'exchange' in df.columns else 'binance'

            # Check if data already exists in database
            existing_count = storage.count_candles(symbol, timeframe, exchange)

            if existing_count > 0:
                print(f"⚠️  Database already has {existing_count:,} candles for {symbol} {timeframe}")
                response = input("   Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("⏭️  Skipping...")
                    storage.close()
                    continue

                # Delete existing data
                print(f"🗑️  Deleting existing data...")
                cursor = storage.conn.cursor()
                cursor.execute("""
                    DELETE FROM ohlcv 
                    WHERE symbol = %s AND timeframe = %s AND exchange = %s
                """, (symbol, timeframe, exchange))
                storage.conn.commit()
                cursor.close()
                print(f"✅ Deleted {existing_count:,} existing rows")

            # Insert into QuestDB
            print(f"💾 Inserting into QuestDB...")
            storage.insert_ohlcv(
                df,
                symbol,
                timeframe,
                exchange,
                batch_size=10000
            )

            # Verify
            print(f"🔍 Verifying...")
            final_count = storage.count_candles(symbol, timeframe, exchange)
            print(f"✅ Verified: {final_count:,} candles in database")

            if final_count != len(df):
                print(f"⚠️  WARNING: Mismatch! File has {len(df):,} but database has {final_count:,}")

            total_inserted += final_count

            # Close connection
            storage.close()

            # Small delay
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            storage.close()
            continue

    print(f"\n{'=' * 70}")
    print(f"✅ LOAD COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total candles in database: {total_inserted:,}")
    print(f"{'=' * 70}")
    print()
    print("📊 Next steps:")
    print("1. Verify: open http://localhost:9000")
    print("2. Query: SELECT symbol, timeframe, COUNT(*) FROM ohlcv GROUP BY symbol, timeframe;")
    print("3. Start Phase 2: Build indicators!")


if __name__ == "__main__":
    print()
    print("🚀 Starting Parquet → QuestDB load...")
    print("⏱️  Estimated time: 5-10 minutes")
    print()

    input("Press Enter to start... ")

    load_parquet_to_questdb()