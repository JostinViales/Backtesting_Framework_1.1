"""
Test all indicators with real data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta
from src.data.loaders.data_loader import DataLoader
from src.indicators.trend import SMA, EMA, MACD
from src.indicators.momentum import RSI, Stochastic
from src.indicators.volatility import ATR, BollingerBands
from src.indicators.statistical import ZScore
from src.indicators.volume import VolumeMA


def test_all_indicators():
    """Test all indicators with real BTC data."""

    print("=" * 70)
    print("🧪 TESTING ALL INDICATORS")
    print("=" * 70)
    print()

    # Load data
    loader = DataLoader('binance')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print("📥 Loading BTC/USDT 5m data (last 30 days)...")
    df = loader.load('BTC/USDT', '5m', start_date, end_date)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Test each indicator
    indicators = [
        ('SMA(20)', SMA(period=20)),
        ('EMA(20)', EMA(period=20)),
        ('MACD', MACD()),
        ('RSI(14)', RSI(period=14)),
        ('Stochastic', Stochastic()),
        ('ATR(14)', ATR(period=14)),
        ('Bollinger Bands', BollingerBands(period=20, std_dev=2)),
        ('Z-Score(20)', ZScore(period=20)),
        ('Volume MA(20)', VolumeMA(period=20)),
    ]

    for name, indicator in indicators:
        print(f"🔧 Testing {name}...")
        try:
            df = indicator.calculate(df)
            print(f"✅ {name} - Success!")
            print(
                f"   Added columns: {[col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'exchange']][-3:]}")
        except Exception as e:
            print(f"❌ {name} - Failed: {e}")
        print()

    # Display sample results
    print("=" * 70)
    print("📊 SAMPLE DATA WITH INDICATORS (Last 5 rows)")
    print("=" * 70)

    display_cols = ['timestamp', 'close', 'sma_20', 'rsi_14', 'zscore_20', 'bb_upper', 'bb_lower']
    print(df[display_cols].tail())

    print()
    print("=" * 70)
    print("✅ ALL INDICATORS TESTED!")
    print("=" * 70)

    loader.close()

    return df


if __name__ == "__main__":
    df_with_indicators = test_all_indicators()