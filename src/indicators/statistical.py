"""
Statistical Indicators - Statistical measures for mean reversion.
"""
import pandas as pd
import numpy as np
from src.indicators.base import StatisticalIndicator


class ZScore(StatisticalIndicator):
    """
    Z-Score (Standard Score)

    **THE MOST IMPORTANT INDICATOR FOR MEAN REVERSION!**

    Measures how many standard deviations price is from mean.
    Z-Score of -2 means price is 2 std devs below average (oversold).
    Z-Score of +2 means price is 2 std devs above average (overbought).

    Formula:
        Z-Score = (Price - SMA) / Standard Deviation

    Usage:
        zscore = ZScore(period=20)
        df = zscore.calculate(data)
        # Adds 'zscore_20' column

    Mean Reversion Strategy:
        - Buy when Z-Score < -2 (very oversold)
        - Sell when Z-Score > +2 (very overbought)
        - Exit when Z-Score returns to 0

    Interpretation:
        Z < -3: Extremely oversold (strong buy signal)
        Z < -2: Oversold (buy signal)
        Z = 0:  At mean (neutral)
        Z > +2: Overbought (sell signal)
        Z > +3: Extremely overbought (strong sell signal)
    """

    def __init__(self, period: int = 20, column: str = 'close'):
        super().__init__(name='ZScore')
        self.period = period
        self.column = column
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Z-Score."""
        self.validate_data(data, [self.column])

        df = data.copy()

        # Calculate rolling mean and std
        rolling_mean = df[self.column].rolling(window=self.period).mean()
        rolling_std = df[self.column].rolling(window=self.period).std()

        # Z-Score = (price - mean) / std
        df[f'zscore_{self.period}'] = (df[self.column] - rolling_mean) / rolling_std

        return df


class PercentFromMA(StatisticalIndicator):
    """
    Percent Distance from Moving Average

    Another way to measure mean reversion opportunity.
    Shows what % price is above/below moving average.

    Formula:
        % from MA = ((Price - MA) / MA) * 100

    Usage:
        pct_ma = PercentFromMA(period=20)
        df = pct_ma.calculate(data)

    Mean Reversion:
        - Buy when < -5% (price 5% below MA)
        - Sell when > +5% (price 5% above MA)
    """

    def __init__(self, period: int = 20, column: str = 'close'):
        super().__init__(name='PercentFromMA')
        self.period = period
        self.column = column
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate percent from moving average."""
        self.validate_data(data, [self.column])

        df = data.copy()

        ma = df[self.column].rolling(window=self.period).mean()
        df[f'pct_from_ma_{self.period}'] = ((df[self.column] - ma) / ma) * 100

        return df