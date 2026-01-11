"""
Interactive Plotly Charts - Modern, interactive visualizations.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime


class InteractivePlotter:
    """
    Create interactive Plotly charts for technical analysis.

    Features:
    - Zoom, pan, hover
    - Export to HTML
    - Multi-timeframe analysis
    - Trade markers
    - Performance analytics
    """

    def __init__(self, template: str = 'plotly_dark'):
        """
        Initialize plotter.

        Args:
            template: Plotly template ('plotly_dark', 'plotly_white', 'seaborn', etc.)
        """
        self.template = template

    def plot_candlestick_with_indicators(
            self,
            data: pd.DataFrame,
            symbol: str = 'BTC/USDT',
            indicators: List[str] = None,
            height: int = 1000
    ):
        """
        Interactive candlestick chart with indicators.

        Shows:
        - Candlestick price chart
        - Multiple indicators in subplots
        - Interactive hover data
        - Zoom/pan functionality
        """
        if indicators is None:
            indicators = ['rsi_14', 'macd', 'volume']

        # Calculate number of subplots
        n_subplots = 1 + len(indicators)
        subplot_heights = [0.5] + [0.5 / len(indicators)] * len(indicators)

        # Create subplots
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=subplot_heights,
            subplot_titles=['Price'] + [ind.upper() for ind in indicators]
        )

        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Add Bollinger Bands if available
        if 'bb_upper' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['bb_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(250, 128, 114, 0.5)', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['bb_middle'],
                    name='BB Middle',
                    line=dict(color='rgba(173, 216, 230, 0.8)', width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['bb_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(144, 238, 144, 0.5)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.1)'
                ),
                row=1, col=1
            )

        # Add SMA if available
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['sma_20'],
                    name='SMA(20)',
                    line=dict(color='orange', width=1.5)
                ),
                row=1, col=1
            )

        # 2. Add indicator subplots
        for i, indicator in enumerate(indicators, start=2):
            if indicator == 'rsi_14' and 'rsi_14' in data.columns:
                # RSI
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['rsi_14'],
                        name='RSI(14)',
                        line=dict(color='purple', width=1.5)
                    ),
                    row=i, col=1
                )
                # Overbought/Oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=i, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=i, col=1)
                fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=i, col=1)
                fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=i, col=1)

            elif indicator == 'zscore_20' and 'zscore_20' in data.columns:
                # Z-Score
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['zscore_20'],
                        name='Z-Score(20)',
                        line=dict(color='purple', width=1.5)
                    ),
                    row=i, col=1
                )
                fig.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.7, row=i, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", opacity=0.7, row=i, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=i, col=1)
                fig.add_hrect(y0=2, y1=4, fillcolor="red", opacity=0.1, row=i, col=1)
                fig.add_hrect(y0=-4, y1=-2, fillcolor="green", opacity=0.1, row=i, col=1)

            elif indicator == 'macd' and 'macd' in data.columns:
                # MACD
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['macd'],
                        name='MACD',
                        line=dict(color='blue', width=1.5)
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['macd_signal'],
                        name='Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=i, col=1
                )
                # Histogram
                colors = ['green' if val >= 0 else 'red' for val in data['macd_histogram']]
                fig.add_trace(
                    go.Bar(
                        x=data['timestamp'],
                        y=data['macd_histogram'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.3
                    ),
                    row=i, col=1
                )

            elif indicator == 'volume':
                # Volume
                colors = ['green' if data['close'].iloc[j] >= data['open'].iloc[j]
                          else 'red' for j in range(len(data))]
                fig.add_trace(
                    go.Bar(
                        x=data['timestamp'],
                        y=data['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5
                    ),
                    row=i, col=1
                )
                if 'volume_ma_20' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamp'],
                            y=data['volume_ma_20'],
                            name='Volume MA',
                            line=dict(color='blue', width=1.5)
                        ),
                        row=i, col=1
                    )

        # Update layout
        fig.update_layout(
            title=f'{symbol} - Interactive Technical Analysis',
            xaxis_title='Date',
            height=height,
            template=self.template,
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )

        return fig

    def plot_equity_curve(
            self,
            data: pd.DataFrame,
            strategy_name: str = 'Strategy',
            benchmark_column: str = None
    ):
        """
        Interactive equity curve with drawdown.

        Args:
            data: DataFrame with 'equity' column and optional 'benchmark'
            strategy_name: Name of the strategy
            benchmark_column: Column name for benchmark (e.g., 'buy_hold_equity')
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=['Equity Curve', 'Drawdown']
        )

        # 1. Equity curve
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['equity'],
                name=strategy_name,
                line=dict(color='#2196F3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ),
            row=1, col=1
        )

        # Add benchmark if provided
        if benchmark_column and benchmark_column in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data[benchmark_column],
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ),
                row=1, col=1
            )

        # 2. Drawdown
        if 'drawdown' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['drawdown'] * 100,  # Convert to percentage
                    name='Drawdown',
                    line=dict(color='red', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'{strategy_name} - Performance',
            height=800,
            template=self.template,
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig

    def plot_trade_markers(
            self,
            data: pd.DataFrame,
            trades: pd.DataFrame,
            symbol: str = 'BTC/USDT'
    ):
        """
        Price chart with trade entry/exit markers.

        Args:
            data: OHLCV DataFrame
            trades: DataFrame with columns: timestamp, type (buy/sell), price
            symbol: Symbol name
        """
        fig = go.Figure()

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            )
        )

        # Buy markers
        buys = trades[trades['type'] == 'buy']
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys['timestamp'],
                    y=buys['price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(color='darkgreen', width=2)
                    ),
                    text=[f"Buy: ${p:.2f}" for p in buys['price']],
                    hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>'
                )
            )

        # Sell markers
        sells = trades[trades['type'] == 'sell']
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells['timestamp'],
                    y=sells['price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(color='darkred', width=2)
                    ),
                    text=[f"Sell: ${p:.2f}" for p in sells['price']],
                    hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>'
                )
            )

        fig.update_layout(
            title=f'{symbol} - Trade Entries & Exits',
            xaxis_title='Date',
            yaxis_title='Price (USDT)',
            height=800,
            template=self.template,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        return fig

    def plot_returns_distribution(self, returns: pd.Series, strategy_name: str = 'Strategy'):
        """
        Histogram of strategy returns distribution.
        """
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=returns * 100,  # Convert to percentage
                nbinsx=50,
                name='Returns',
                marker=dict(
                    color='blue',
                    line=dict(color='darkblue', width=1)
                ),
                opacity=0.7
            )
        )

        # Add mean line
        mean_return = returns.mean() * 100
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )

        fig.update_layout(
            title=f'{strategy_name} - Returns Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            height=600,
            template=self.template,
            showlegend=True
        )

        return fig

    def plot_correlation_heatmap(
            self,
            data: pd.DataFrame,
            columns: List[str] = None,
            title: str = 'Correlation Heatmap'
    ):
        """
        Interactive correlation heatmap.

        Args:
            data: DataFrame with multiple columns
            columns: List of column names to correlate
            title: Chart title
        """
        if columns is None:
            # Use all numeric columns
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate correlation matrix
        corr_matrix = data[columns].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=title,
            height=600,
            template=self.template
        )

        return fig

    def plot_multi_symbol_comparison(
            self,
            data_dict: Dict[str, pd.DataFrame],
            normalize: bool = True
    ):
        """
        Compare multiple symbols on one chart.

        Args:
            data_dict: Dictionary of {symbol: dataframe}
            normalize: If True, normalize all to start at 100
        """
        fig = go.Figure()

        for symbol, data in data_dict.items():
            if normalize:
                # Normalize to 100
                y_values = (data['close'] / data['close'].iloc[0]) * 100
                y_label = 'Normalized Price (Starting at 100)'
            else:
                y_values = data['close']
                y_label = 'Price (USDT)'

            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=y_values,
                    name=symbol,
                    mode='lines',
                    line=dict(width=2)
                )
            )

        fig.update_layout(
            title='Multi-Symbol Comparison',
            xaxis_title='Date',
            yaxis_title=y_label,
            height=700,
            template=self.template,
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def save(self, fig, filename: str, output_dir: str = 'outputs/charts'):
        """
        Save interactive figure as HTML.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            output_dir: Output directory
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / f"{filename}.html"
        fig.write_html(str(filepath))

        print(f"✅ Saved: {filepath}")

        return filepath