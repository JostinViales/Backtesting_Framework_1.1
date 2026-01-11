"""
Volatility Indicators - Measure price fluctuation.
"""
import pandas as pd
import numpy as np
from src.indicators.base import VolatilityIndicator


class ATR(VolatilityIndicator):
    """
    Average True Range (ATR)

    Measures market volatility.
    High ATR = high volatility, Low ATR = low volatility.
    Essential for position sizing and stop losses.

    Formula:
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA of TR

    Usage:
        atr = ATR(period=14)
        df = atr.calculate(data)

    Mean Reversion Use:
        - Trade more in low ATR (stable conditions)
        - Reduce size in high ATR (volatile conditions)
        - Stop loss = Entry ± (ATR * 2)
    """

    def __init__(self, period: int = 14):
        super().__init__(name='ATR')
        self.period = period
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR."""
        self.validate_data(data, ['high', 'low', 'close'])

        df = data.copy()

        # True Range components
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))

        # True Range = max of the three
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR = EMA of TR
        df[f'atr_{self.period}'] = tr.ewm(span=self.period, adjust=False).mean()

        return df


class BollingerBands(VolatilityIndicator):
    """
    Bollinger Bands

    Shows price relative to volatility bands.
    PERFECT for mean reversion!

    Formula:
        Middle Band = SMA(period)
        Upper Band = Middle + (std * std_dev)
        Lower Band = Middle - (std * std_dev)

    Usage:
        bb = BollingerBands(period=20, std_dev=2)
        df = bb.calculate(data)
        # Adds: 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width'

    Mean Reversion Strategy:
        - Buy when price touches lower band
        - Sell when price touches upper band
        - Exit when price returns to middle band
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0, column: str = 'close'):
        super().__init__(name='BollingerBands')
        self.period = period
        self.std_dev = std_dev
        self.column = column
        self._params = {'period': period, 'std_dev': std_dev}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        self.validate_data(data, [self.column])

        df = data.copy()

        # Middle band (SMA)
        df['bb_middle'] = df[self.column].rolling(window=self.period).mean()

        # Standard deviation
        rolling_std = df[self.column].rolling(window=self.period).std()

        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * self.std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * self.std_dev)

        # Bandwidth (useful for volatility measurement)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # Percent B (where price is within bands)
        df['bb_percent'] = (df[self.column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df


class StandardDeviation(VolatilityIndicator):
    """
    Rolling Standard Deviation

    Measures price dispersion.
    Higher std = more volatile.

    Usage:
        std = StandardDeviation(period=20)
        df = std.calculate(data)
    """

    def __init__(self, period: int = 20, column: str = 'close'):
        super().__init__(name='StdDev')
        self.period = period
        self.column = column
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard deviation."""
        self.validate_data(data, [self.column])

        df = data.copy()
        df[f'std_{self.period}'] = df[self.column].rolling(window=self.period).std()

        return df