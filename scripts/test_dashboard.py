"""
Test the trading dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.data.loaders.data_loader import DataLoader
from src.indicators.momentum import RSI
from src.indicators.statistical import ZScore
from src.visualization.dashboard import TradingDashboard


def test_dashboard():
    """Create multi-symbol dashboard."""

    print("=" * 70)
    print("📊 CREATING TRADING DASHBOARD")
    print("=" * 70)
    print()

    # Load data
    loader = DataLoader('binance')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last 7 days

    symbols = ['BTC/USDT', 'SOL/USDT', 'OP/USDT']

    print(f"📥 Loading data for {len(symbols)} symbols...")
    data_dict = {}

    for symbol in symbols:
        print(f"   Loading {symbol}...")
        df = loader.load(symbol, '5m', start_date, end_date)

        # Calculate indicators
        df = RSI(period=14).calculate(df)
        df = ZScore(period=20).calculate(df)

        data_dict[symbol] = df
        print(f"   ✅ {symbol}: {len(df):,} candles")

    print()
    print("🎨 Creating dashboard...")

    # Create dashboard
    dashboard = TradingDashboard(symbols)
    fig = dashboard.create_dashboard(data_dict)

    # Save
    output_path = Path('outputs/charts/interactive/trading_dashboard.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))

    loader.close()

    print(f"✅ Saved: {output_path}")
    print()
    print("=" * 70)
    print("✅ DASHBOARD CREATED!")
    print("=" * 70)
    print()
    print("🌐 Open in browser:")
    print(f"   open {output_path}")
    print()
    print("💡 Features:")
    print("   - 3 symbols side-by-side")
    print("   - Price + RSI + Z-Score for each")
    print("   - Interactive zoom/pan")
    print("   - Quick market overview")


if __name__ == "__main__":
    test_dashboard()