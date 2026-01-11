"""
Trend Indicators - Moving averages and trend-following tools.
"""
import pandas as pd
import numpy as np
from src.indicators.base import TrendIndicator


class SMA(TrendIndicator):
    """
    Simple Moving Average (SMA)

    The most basic indicator - average price over N periods.
    Essential for mean reversion (price reverts to SMA).

    Formula:
        SMA = (P1 + P2 + ... + Pn) / n

    Usage:
        sma = SMA(period=20)
        df = sma.calculate(data)
        # Adds 'sma_20' column
    """

    def __init__(self, period: int = 20, column: str = 'close'):
        """
        Args:
            period: Lookback period (default: 20)
            column: Price column to use (default: 'close')
        """
        super().__init__(name='SMA')
        self.period = period
        self.column = column
        self._params = {'period': period, 'column': column}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA."""
        self.validate_data(data, [self.column])

        df = data.copy()
        col_name = f'sma_{self.period}'
        df[col_name] = df[self.column].rolling(window=self.period).mean()

        return df


class EMA(TrendIndicator):
    """
    Exponential Moving Average (EMA)

    Weighted average that gives more importance to recent prices.
    More responsive than SMA, better for faster mean reversion.

    Formula:
        EMA = (Close - EMA_prev) * multiplier + EMA_prev
        where multiplier = 2 / (period + 1)

    Usage:
        ema = EMA(period=20)
        df = ema.calculate(data)
        # Adds 'ema_20' column
    """

    def __init__(self, period: int = 20, column: str = 'close'):
        super().__init__(name='EMA')
        self.period = period
        self.column = column
        self._params = {'period': period, 'column': column}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA."""
        self.validate_data(data, [self.column])

        df = data.copy()
        col_name = f'ema_{self.period}'
        df[col_name] = df[self.column].ewm(span=self.period, adjust=False).mean()

        return df


class MACD(TrendIndicator):
    """
    Moving Average Convergence Divergence (MACD)

    Shows relationship between two moving averages.
    Useful for identifying trend strength and reversals.

    Formula:
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line

    Usage:
        macd = MACD()
        df = macd.calculate(data)
        # Adds: 'macd', 'macd_signal', 'macd_histogram'
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = 'close'):
        super().__init__(name='MACD')
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.column = column
        self._params = {'fast': fast, 'slow': slow, 'signal': signal}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        self.validate_data(data, [self.column])

        df = data.copy()

        # Calculate EMAs
        ema_fast = df[self.column].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df[self.column].ewm(span=self.slow, adjust=False).mean()

        # MACD line
        df['macd'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=self.signal, adjust=False).mean()

        # Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df