"""
Test interactive Plotly visualizations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from src.data.loaders.data_loader import DataLoader
from src.indicators.trend import SMA, MACD
from src.indicators.momentum import RSI
from src.indicators.volatility import BollingerBands, ATR  # ← FIXED
from src.indicators.statistical import ZScore
from src.indicators.volume import VolumeMA
from src.visualization.interactive_plots import InteractivePlotter


def test_interactive_charts():
    """Create all interactive charts."""

    print("="*70)
    print("🎨 CREATING INTERACTIVE PLOTLY CHARTS")
    print("="*70)
    print()

    # Load data
    loader = DataLoader('binance')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print("📥 Loading BTC/USDT 5m data...")
    df = loader.load('BTC/USDT', '5m', start_date, end_date)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Calculate indicators
    print("🔧 Calculating indicators...")
    df = SMA(period=20).calculate(df)
    df = MACD().calculate(df)
    df = RSI(period=14).calculate(df)
    df = BollingerBands(period=20).calculate(df)
    df = ATR(period=14).calculate(df)  # ← FIXED
    df = ZScore(period=20).calculate(df)
    df = VolumeMA(period=20).calculate(df)
    print("✅ Indicators calculated!")
    print()

    # Create plotter
    plotter = InteractivePlotter(template='plotly_dark')

    # 1. Candlestick with indicators
    print("[1/4] Creating interactive candlestick chart...")
    fig1 = plotter.plot_candlestick_with_indicators(
        df,
        symbol='BTC/USDT',
        indicators=['rsi_14', 'zscore_20', 'macd', 'volume']
    )
    plotter.save(fig1, 'interactive_candlestick', 'outputs/charts/interactive')

    # 2. Correlation heatmap
    print("[2/4] Creating correlation heatmap...")
    fig2 = plotter.plot_correlation_heatmap(
        df,
        columns=['close', 'volume', 'rsi_14', 'zscore_20', 'atr_14'],  # Now atr_14 exists!
        title='Indicator Correlation Matrix'
    )
    plotter.save(fig2, 'correlation_heatmap', 'outputs/charts/interactive')

    # 3. Multi-symbol comparison
    print("[3/4] Creating multi-symbol comparison...")

    # Load multiple symbols
    symbols_data = {}
    for symbol in ['BTC/USDT', 'SOL/USDT', 'OP/USDT']:
        print(f"   Loading {symbol}...")
        data = loader.load(symbol, '5m', start_date, end_date)
        symbols_data[symbol] = data

    fig3 = plotter.plot_multi_symbol_comparison(symbols_data, normalize=True)
    plotter.save(fig3, 'multi_symbol_comparison', 'outputs/charts/interactive')

    # 4. Returns distribution (simulated for now)
    print("[4/4] Creating returns distribution...")
    df['returns'] = df['close'].pct_change()
    fig4 = plotter.plot_returns_distribution(df['returns'].dropna(), 'BTC/USDT')
    plotter.save(fig4, 'returns_distribution', 'outputs/charts/interactive')

    loader.close()

    print()
    print("="*70)
    print("✅ ALL INTERACTIVE CHARTS CREATED!")
    print("="*70)
    print()
    print("📁 Charts saved to: outputs/charts/interactive/")
    print()
    print("💡 Open the HTML files in your browser!")
    print("💡 Features: Zoom, pan, hover, export")
    print()
    print("🌐 To open:")
    print("   open outputs/charts/interactive/interactive_candlestick.html")
    print("   open outputs/charts/interactive/correlation_heatmap.html")
    print("   open outputs/charts/interactive/multi_symbol_comparison.html")


if __name__ == "__main__":
    test_interactive_charts()