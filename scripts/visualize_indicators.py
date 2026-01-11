"""
Visualize indicators on real BTC data.

Creates multiple charts showing:
- Price with moving averages
- Bollinger Bands mean reversion
- RSI momentum
- Z-Score statistical signals
- MACD trend
- Volume analysis
- Comprehensive dashboard
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.data.loaders.data_loader import DataLoader
from src.indicators.trend import SMA, EMA, MACD
from src.indicators.momentum import RSI, Stochastic
from src.indicators.volatility import ATR, BollingerBands
from src.indicators.statistical import ZScore
from src.indicators.volume import VolumeMA
from src.visualization.indicator_plots import IndicatorPlotter


def visualize_all_indicators():
    """Create all visualization charts."""

    print("=" * 70)
    print("📊 CREATING INDICATOR VISUALIZATIONS")
    print("=" * 70)
    print()

    # Load data (last 30 days for clearer charts)
    loader = DataLoader('binance')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print("📥 Loading SOL/USDT 15 data...")
    df = loader.load('SOL/USDT', '15m', start_date, end_date)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Calculate all indicators
    print("🔧 Calculating indicators...")
    df = SMA(period=20).calculate(df)
    df = SMA(period=50).calculate(df)
    df = EMA(period=20).calculate(df)
    df = MACD().calculate(df)
    df = RSI(period=14).calculate(df)
    df = Stochastic().calculate(df)
    df = ATR(period=14).calculate(df)
    df = BollingerBands(period=20, std_dev=2.5).calculate(df)
    df = ZScore(period=20).calculate(df)
    df = VolumeMA(period=20).calculate(df)
    print("✅ All indicators calculated!")
    print()

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs' / 'charts'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize plotter
    plotter = IndicatorPlotter(figsize=(16, 10))

    charts = [
        ("Price with Moving Averages", lambda: plotter.plot_price_with_ma(df, ma_periods=[20, 50])),
        ("Bollinger Bands Mean Reversion", lambda: plotter.plot_bollinger_bands(df)),
        ("RSI Momentum", lambda: plotter.plot_rsi(df)),
        ("Z-Score Mean Reversion", lambda: plotter.plot_zscore(df)),
        ("MACD Trend", lambda: plotter.plot_macd(df)),
        ("Volume Analysis", lambda: plotter.plot_volume(df)),
        ("Comprehensive Dashboard", lambda: plotter.plot_comprehensive_dashboard(df)),
    ]

    print("🎨 Creating charts...")
    print()

    for i, (name, func) in enumerate(charts, 1):
        print(f"[{i}/{len(charts)}] Creating: {name}")

        try:
            fig = func()
            if fig:
                filename = name.lower().replace(' ', '_') + '.png'
                filepath = output_dir / filename
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    ✅ Saved: {filepath}")
            else:
                print(f"    ⚠️  Skipped (missing data)")
        except Exception as e:
            print(f"    ❌ Error: {e}")

        print()

    loader.close()

    print("=" * 70)
    print("✅ ALL CHARTS CREATED!")
    print("=" * 70)
    print()
    print(f"📁 Charts saved to: {output_dir}")
    print()
    print("📊 Charts created:")
    for i, (name, _) in enumerate(charts, 1):
        print(f"   {i}. {name}")
    print()
    print("💡 Next: Open the PNG files to see your indicators!")
    print("💡 Or run: open outputs/charts/")


if __name__ == "__main__":
    print()
    print("🚀 Starting indicator visualization...")
    print()

    input("Press Enter to create charts... ")

    visualize_all_indicators()