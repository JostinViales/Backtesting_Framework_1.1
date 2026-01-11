"""
Real-time Dashboard - Live market monitoring.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from typing import Dict, List


class TradingDashboard:
    """
    Real-time trading dashboard with live updates.

    Features:
    - Multiple symbol monitoring
    - Live price updates
    - Signal generation
    - Performance tracking
    """

    def __init__(self, symbols: List[str], template: str = 'plotly_dark'):
        self.symbols = symbols
        self.template = template

    def create_dashboard(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Create comprehensive dashboard.

        Args:
            data_dict: Dictionary of {symbol: dataframe with indicators}
        """
        n_symbols = len(data_dict)

        # Create subplot titles properly
        subplot_titles = []
        for sym in data_dict.keys():
            subplot_titles.extend([f'{sym} Price', f'{sym} RSI', f'{sym} Z-Score'])

        # Create subplot specs
        specs = [[{'type': 'candlestick'}, {'type': 'scatter'}, {'type': 'scatter'}]
                 for _ in range(n_symbols)]

        fig = make_subplots(
            rows=n_symbols,
            cols=3,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            specs=specs
        )

        for i, (symbol, data) in enumerate(data_dict.items(), start=1):
            # 1. Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=symbol,
                    showlegend=False,
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=i, col=1
            )

            # Add SMA if available
            if 'sma_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['sma_20'],
                        name='SMA(20)',
                        line=dict(color='orange', width=1),
                        showlegend=False
                    ),
                    row=i, col=1
                )

            # 2. RSI
            if 'rsi_14' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['rsi_14'],
                        name='RSI',
                        line=dict(color='purple', width=1.5),
                        showlegend=False
                    ),
                    row=i, col=2
                )
                # Overbought/Oversold lines
                fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.5,
                    row=i,
                    col=2
                )
                fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    row=i,
                    col=2
                )

                # Highlight current RSI level
                current_rsi = data['rsi_14'].iloc[-1]
                if current_rsi > 70:
                    color = 'red'
                    signal = 'OVERBOUGHT'
                elif current_rsi < 30:
                    color = 'green'
                    signal = 'OVERSOLD'
                else:
                    color = 'gray'
                    signal = 'NEUTRAL'

                # Add annotation
                fig.add_annotation(
                    x=data['timestamp'].iloc[-1],
                    y=current_rsi,
                    text=f'{current_rsi:.1f}',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    font=dict(size=10, color=color),
                    row=i,
                    col=2
                )

            # 3. Z-Score
            if 'zscore_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['zscore_20'],
                        name='Z-Score',
                        line=dict(color='purple', width=1.5),
                        showlegend=False
                    ),
                    row=i, col=3
                )
                # Signal lines
                fig.add_hline(
                    y=2,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    row=i,
                    col=3
                )
                fig.add_hline(
                    y=-2,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.7,
                    row=i,
                    col=3
                )
                fig.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                    row=i,
                    col=3
                )

                # Highlight current Z-Score
                current_zscore = data['zscore_20'].iloc[-1]
                if current_zscore < -2:
                    color = 'green'
                    signal = 'BUY'
                elif current_zscore > 2:
                    color = 'red'
                    signal = 'SELL'
                else:
                    color = 'gray'
                    signal = 'NEUTRAL'

                # Add annotation
                fig.add_annotation(
                    x=data['timestamp'].iloc[-1],
                    y=current_zscore,
                    text=f'{current_zscore:.2f} ({signal})',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    font=dict(size=10, color=color),
                    row=i,
                    col=3
                )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Trading Dashboard - Real-time Market Monitor',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=300 * n_symbols,
            template=self.template,
            showlegend=False,
            hovermode='x unified'
        )

        # Update y-axes labels
        for i in range(1, n_symbols + 1):
            fig.update_yaxes(title_text="Price", row=i, col=1)
            fig.update_yaxes(title_text="RSI", row=i, col=2)
            fig.update_yaxes(title_text="Z-Score", row=i, col=3)

        # Update x-axes (only show on bottom row)
        fig.update_xaxes(title_text="Date", row=n_symbols, col=1)
        fig.update_xaxes(title_text="Date", row=n_symbols, col=2)
        fig.update_xaxes(title_text="Date", row=n_symbols, col=3)

        # Remove rangeslider
        fig.update_xaxes(rangeslider_visible=False)

        return fig