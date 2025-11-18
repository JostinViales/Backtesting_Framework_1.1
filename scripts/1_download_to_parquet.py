"""
Stage 1: Download historical data and save as Parquet files.

This script:
- Downloads data from Binance
- Saves locally as compressed Parquet files
- Creates one file per symbol/timeframe
- Can be run multiple times (resumes if file exists)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.data.providers.ccxt_provider import CCXTDataProvider
import pandas as pd
import os


def download_to_parquet():
    """
    Download historical data and save as Parquet files.
    """
    # Configuration
    symbols = [
        'BTC/USDT',
        'SOL/USDT',
        'OP/USDT'
    ]

    timeframes = [
        '5m',
        '15m',
        '1h',
        '1d',
    ]

    exchange = 'binance'
    start_date = datetime(2022, 7, 1, 0, 0, 0)
    end_date = datetime.now()

    # Output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'raw'

    print("=" * 70)
    print("📥 STAGE 1: DOWNLOAD TO PARQUET FILES")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()

    provider = CCXTDataProvider(exchange)

    total_files = len(symbols) * len(timeframes)
    current_file = 0
    total_downloaded = 0

    for symbol in symbols:
        # Create symbol directory
        symbol_dir = output_dir / symbol.replace('/', '_')
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for timeframe in timeframes:
            current_file += 1

            # Output file path
            output_file = symbol_dir / f"{timeframe}.parquet"

            print(f"\n{'=' * 70}")
            print(f"[{current_file}/{total_files}] {symbol} {timeframe}")
            print(f"{'=' * 70}")

            # Check if file already exists
            if output_file.exists():
                print(f"⚠️  File already exists: {output_file}")
                response = input("Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("⏭️  Skipping...")
                    continue

            try:
                # Fetch data
                print(f"📥 Downloading from {exchange}...")
                df = provider.fetch_ohlcv(symbol, timeframe, start_date, end_date)

                if df.empty:
                    print(f"⚠️  No data available")
                    continue

                print(f"📦 Fetched {len(df):,} candles")

                # Add metadata columns
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                df['exchange'] = exchange

                # Save as Parquet
                print(f"💾 Saving to {output_file.name}...")
                df.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy',  # Fast compression
                    index=False
                )

                # Get file size
                file_size_mb = output_file.stat().st_size / (1024 * 1024)

                print(f"✅ Saved {len(df):,} rows ({file_size_mb:.2f} MB)")

                total_downloaded += len(df)

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    print(f"\n{'=' * 70}")
    print(f"✅ DOWNLOAD COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total candles downloaded: {total_downloaded:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}")
    print()
    print("📊 Next step:")
    print("   python scripts/2_parquet_to_questdb.py")


if __name__ == "__main__":
    print()
    print("🚀 Starting download to Parquet files...")
    print("⏱️  Estimated time: 30-45 minutes")
    print("💾 Data will be saved to: data/raw/")
    print()

    input("Press Enter to start... ")

    download_to_parquet()