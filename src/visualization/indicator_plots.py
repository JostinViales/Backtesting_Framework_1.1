"""
Indicator Visualization - Plot technical indicators on price charts.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Optional, List


class IndicatorPlotter:
    """
    Create publication-quality charts for technical indicators.
    """

    def __init__(self, figsize=(16, 10)):
        """
        Initialize plotter.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_price_with_ma(self, data: pd.DataFrame, symbol: str = 'BTC/USDT',
                           ma_periods: List[int] = [20, 50]):
        """
        Plot price with moving averages.

        Args:
            data: DataFrame with OHLCV and indicator columns
            symbol: Symbol name for title
            ma_periods: List of MA periods to plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot price
        ax.plot(data['timestamp'], data['close'], label='Price',
                color='black', linewidth=1.5, alpha=0.8)

        # Plot moving averages
        colors = ['blue', 'red', 'green', 'purple']
        for i, period in enumerate(ma_periods):
            col = f'sma_{period}'
            if col in data.columns:
                ax.plot(data['timestamp'], data[col],
                        label=f'SMA({period})',
                        color=colors[i % len(colors)],
                        linewidth=1.2,
                        alpha=0.7)

        ax.set_title(f'{symbol} - Price with Moving Averages',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USDT)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_bollinger_bands(self, data: pd.DataFrame, symbol: str = 'BTC/USDT'):
        """
        Plot price with Bollinger Bands.

        Perfect visualization for mean reversion!
        Shows when price touches bands (buy/sell signals).
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot price
        ax.plot(data['timestamp'], data['close'],
                label='Price', color='black', linewidth=1.5)

        # Plot Bollinger Bands
        if 'bb_upper' in data.columns:
            ax.plot(data['timestamp'], data['bb_upper'],
                    label='Upper Band', color='red',
                    linewidth=1, linestyle='--', alpha=0.7)
            ax.plot(data['timestamp'], data['bb_middle'],
                    label='Middle Band (SMA)', color='blue',
                    linewidth=1.2, alpha=0.7)
            ax.plot(data['timestamp'], data['bb_lower'],
                    label='Lower Band', color='green',
                    linewidth=1, linestyle='--', alpha=0.7)

            # Fill between bands
            ax.fill_between(data['timestamp'],
                            data['bb_upper'],
                            data['bb_lower'],
                            alpha=0.1, color='gray')

            # Highlight touches
            touch_upper = data[data['close'] >= data['bb_upper']]
            touch_lower = data[data['close'] <= data['bb_lower']]

            ax.scatter(touch_upper['timestamp'], touch_upper['close'],
                       color='red', s=50, marker='v',
                       label='Touch Upper (Sell Signal)', zorder=5)
            ax.scatter(touch_lower['timestamp'], touch_lower['close'],
                       color='green', s=50, marker='^',
                       label='Touch Lower (Buy Signal)', zorder=5)

        ax.set_title(f'{symbol} - Bollinger Bands Mean Reversion',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USDT)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_rsi(self, data: pd.DataFrame, symbol: str = 'BTC/USDT',
                 period: int = 14):
        """
        Plot RSI indicator with overbought/oversold levels.

        Shows momentum and mean reversion opportunities.
        """
        col = f'rsi_{period}'
        if col not in data.columns:
            print(f"⚠️  Column '{col}' not found in data")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                       height_ratios=[2, 1])

        # Top: Price
        ax1.plot(data['timestamp'], data['close'],
                 color='black', linewidth=1.5)
        ax1.set_title(f'{symbol} - Price and RSI({period})',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Bottom: RSI
        ax2.plot(data['timestamp'], data[col],
                 color='purple', linewidth=1.5, label=f'RSI({period})')

        # Overbought/Oversold lines
        ax2.axhline(y=70, color='red', linestyle='--',
                    linewidth=1, alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=20, color='green', linestyle='--',
                    linewidth=1, alpha=0.7, label='Oversold (20)')
        ax2.axhline(y=50, color='gray', linestyle=':',
                    linewidth=1, alpha=0.5, label='Neutral (50)')

        # Fill overbought/oversold regions
        ax2.fill_between(data['timestamp'], 70, 100,
                         alpha=0.1, color='red')
        ax2.fill_between(data['timestamp'], 0, 20,
                         alpha=0.1, color='green')

        # Highlight signals
        oversold = data[data[col] < 20]
        overbought = data[data[col] > 70]

        ax2.scatter(oversold['timestamp'], oversold[col],
                    color='green', s=50, marker='^',
                    label='Buy Signal', zorder=5)
        ax2.scatter(overbought['timestamp'], overbought[col],
                    color='red', s=50, marker='v',
                    label='Sell Signal', zorder=5)

        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_zscore(self, data: pd.DataFrame, symbol: str = 'BTC/USDT',
                    period: int = 20):
        """
        Plot Z-Score - THE MOST IMPORTANT for mean reversion!

        Shows exactly when to buy/sell based on statistical deviation.
        """
        col = f'zscore_{period}'
        if col not in data.columns:
            print(f"⚠️  Column '{col}' not found in data")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                       height_ratios=[2, 1])

        # Top: Price with SMA
        ax1.plot(data['timestamp'], data['close'],
                 color='black', linewidth=1.5, label='Price')
        if f'sma_{period}' in data.columns:
            ax1.plot(data['timestamp'], data[f'sma_{period}'],
                     color='blue', linewidth=1.2,
                     linestyle='--', alpha=0.7, label=f'SMA({period})')

        ax1.set_title(f'{symbol} - Z-Score Mean Reversion Strategy',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom: Z-Score
        ax2.plot(data['timestamp'], data[col],
                 color='purple', linewidth=1.5, label=f'Z-Score({period})')

        # Critical levels
        ax2.axhline(y=2, color='red', linestyle='--',
                    linewidth=1.5, alpha=0.7, label='Overbought (+2σ)')
        ax2.axhline(y=-2, color='green', linestyle='--',
                    linewidth=1.5, alpha=0.7, label='Oversold (-2σ)')
        ax2.axhline(y=0, color='gray', linestyle='-',
                    linewidth=1, alpha=0.5, label='Mean (0)')

        # Extreme levels
        ax2.axhline(y=3, color='darkred', linestyle=':',
                    linewidth=1, alpha=0.5, label='Extreme Overbought (+3σ)')
        ax2.axhline(y=-3, color='darkgreen', linestyle=':',
                    linewidth=1, alpha=0.5, label='Extreme Oversold (-3σ)')

        # Fill regions
        ax2.fill_between(data['timestamp'], 2, 10,
                         alpha=0.1, color='red')
        ax2.fill_between(data['timestamp'], -10, -2,
                         alpha=0.1, color='green')

        # Highlight signals
        strong_buy = data[data[col] < -2]
        strong_sell = data[data[col] > 2]

        ax2.scatter(strong_buy['timestamp'], strong_buy[col],
                    color='green', s=50, marker='^',
                    label='BUY Signal', zorder=5)
        ax2.scatter(strong_sell['timestamp'], strong_sell[col],
                    color='red', s=50, marker='v',
                    label='SELL Signal', zorder=5)

        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Z-Score (σ)', fontsize=12)
        ax2.set_ylim(-4, 4)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_macd(self, data: pd.DataFrame, symbol: str = 'BTC/USDT'):
        """Plot MACD indicator."""
        if 'macd' not in data.columns:
            print("⚠️  MACD not found in data")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                       height_ratios=[2, 1])

        # Top: Price
        ax1.plot(data['timestamp'], data['close'],
                 color='black', linewidth=1.5)
        ax1.set_title(f'{symbol} - MACD Indicator',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Bottom: MACD
        ax2.plot(data['timestamp'], data['macd'],
                 color='blue', linewidth=1.5, label='MACD')
        ax2.plot(data['timestamp'], data['macd_signal'],
                 color='red', linewidth=1.2, label='Signal')

        # Histogram
        colors = ['green' if x >= 0 else 'red' for x in data['macd_histogram']]
        ax2.bar(data['timestamp'], data['macd_histogram'],
                color=colors, alpha=0.3, label='Histogram')

        ax2.axhline(y=0, color='gray', linestyle='-',
                    linewidth=1, alpha=0.5)

        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('MACD', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_volume(self, data: pd.DataFrame, symbol: str = 'BTC/USDT'):
        """Plot volume with moving average."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                       height_ratios=[2, 1])

        # Top: Price
        ax1.plot(data['timestamp'], data['close'],
                 color='black', linewidth=1.5)
        ax1.set_title(f'{symbol} - Price and Volume',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Bottom: Volume
        colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i]
                  else 'red' for i in range(len(data))]

        ax2.bar(data['timestamp'], data['volume'],
                color=colors, alpha=0.5, label='Volume')

        # Volume MA
        if 'volume_ma_20' in data.columns:
            ax2.plot(data['timestamp'], data['volume_ma_20'],
                     color='blue', linewidth=1.5, label='Volume MA(20)')

        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_comprehensive_dashboard(self, data: pd.DataFrame,
                                     symbol: str = 'BTC/USDT'):
        """
        Create comprehensive dashboard with all indicators.

        Perfect overview of the market!
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2)

        # 1. Price with Bollinger Bands
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data['timestamp'], data['close'],
                 color='black', linewidth=1.5, label='Price')
        if 'bb_upper' in data.columns:
            ax1.plot(data['timestamp'], data['bb_upper'],
                     color='red', linestyle='--', alpha=0.6, label='BB Upper')
            ax1.plot(data['timestamp'], data['bb_middle'],
                     color='blue', linestyle='--', alpha=0.6, label='BB Middle')
            ax1.plot(data['timestamp'], data['bb_lower'],
                     color='green', linestyle='--', alpha=0.6, label='BB Lower')
            ax1.fill_between(data['timestamp'], data['bb_upper'], data['bb_lower'],
                             alpha=0.1, color='gray')
        ax1.set_title(f'{symbol} - Comprehensive Technical Analysis',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = fig.add_subplot(gs[1, 0])
        if 'rsi_14' in data.columns:
            ax2.plot(data['timestamp'], data['rsi_14'],
                     color='purple', linewidth=1.5)
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(20, color='green', linestyle='--', alpha=0.5)
            ax2.fill_between(data['timestamp'], 70, 100, alpha=0.1, color='red')
            ax2.fill_between(data['timestamp'], 0, 20, alpha=0.1, color='green')
        ax2.set_title('RSI(14)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # 3. Z-Score
        ax3 = fig.add_subplot(gs[1, 1])
        if 'zscore_20' in data.columns:
            ax3.plot(data['timestamp'], data['zscore_20'],
                     color='purple', linewidth=1.5)
            ax3.axhline(2, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(-2, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax3.fill_between(data['timestamp'], 2, 10, alpha=0.1, color='red')
            ax3.fill_between(data['timestamp'], -10, -2, alpha=0.1, color='green')
        ax3.set_title('Z-Score(20)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Z-Score (σ)', fontsize=10)
        ax3.set_ylim(-4, 4)
        ax3.grid(True, alpha=0.3)

        # 4. MACD
        ax4 = fig.add_subplot(gs[2, 0])
        if 'macd' in data.columns:
            ax4.plot(data['timestamp'], data['macd'],
                     color='blue', linewidth=1.2, label='MACD')
            ax4.plot(data['timestamp'], data['macd_signal'],
                     color='red', linewidth=1, label='Signal')
            colors = ['green' if x >= 0 else 'red' for x in data['macd_histogram']]
            ax4.bar(data['timestamp'], data['macd_histogram'],
                    color=colors, alpha=0.3)
            ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_title('MACD', fontsize=12, fontweight='bold')
        ax4.set_ylabel('MACD', fontsize=10)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. ATR
        ax5 = fig.add_subplot(gs[2, 1])
        if 'atr_14' in data.columns:
            ax5.plot(data['timestamp'], data['atr_14'],
                     color='orange', linewidth=1.5)
        ax5.set_title('ATR(14) - Volatility', fontsize=12, fontweight='bold')
        ax5.set_ylabel('ATR', fontsize=10)
        ax5.grid(True, alpha=0.3)

        # 6. Volume
        ax6 = fig.add_subplot(gs[3, :])
        colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i]
                  else 'red' for i in range(len(data))]
        ax6.bar(data['timestamp'], data['volume'],
                color=colors, alpha=0.5)
        if 'volume_ma_20' in data.columns:
            ax6.plot(data['timestamp'], data['volume_ma_20'],
                     color='blue', linewidth=1.5, label='Volume MA(20)')
        ax6.set_title('Volume', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Date', fontsize=10)
        ax6.set_ylabel('Volume', fontsize=10)
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)

        # Format all x-axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        return fig