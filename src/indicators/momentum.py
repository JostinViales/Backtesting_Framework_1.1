"""
Momentum Indicators - Measure rate of price change.
"""
import pandas as pd
import numpy as np
from src.indicators.base import MomentumIndicator


class RSI(MomentumIndicator):
    """
    Relative Strength Index (RSI)

    Measures momentum on 0-100 scale.
    Key for mean reversion: RSI < 30 = oversold, RSI > 70 = overbought

    Formula:
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    Usage:
        rsi = RSI(period=14)
        df = rsi.calculate(data)
        # Adds 'rsi_14' column

    Mean Reversion Strategy:
        - Buy when RSI < 30 (oversold)
        - Sell when RSI > 70 (overbought)
    """

    def __init__(self, period: int = 14, column: str = 'close'):
        super().__init__(name='RSI')
        self.period = period
        self.column = column
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI."""
        self.validate_data(data, [self.column])

        df = data.copy()

        # Calculate price changes
        delta = df[self.column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain/loss using EMA
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df[f'rsi_{self.period}'] = 100 - (100 / (1 + rs))

        return df


class Stochastic(MomentumIndicator):
    """
    Stochastic Oscillator

    Compares close price to price range over period.
    Also oscillates 0-100, similar interpretation to RSI.

    Formula:
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA of %K

    Usage:
        stoch = Stochastic(period=14)
        df = stoch.calculate(data)
        # Adds 'stoch_k' and 'stoch_d' columns
    """

    def __init__(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        super().__init__(name='Stochastic')
        self.period = period
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        self._params = {'period': period, 'smooth_k': smooth_k, 'smooth_d': smooth_d}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic."""
        self.validate_data(data, ['high', 'low', 'close'])

        df = data.copy()

        # Lowest low and highest high
        lowest_low = df['low'].rolling(window=self.period).min()
        highest_high = df['high'].rolling(window=self.period).max()

        # %K
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)

        # Smooth %K
        df['stoch_k'] = stoch_k.rolling(window=self.smooth_k).mean()

        # %D (signal line)
        df['stoch_d'] = df['stoch_k'].rolling(window=self.smooth_d).mean()

        return df


class ROC(MomentumIndicator):
    """
    Rate of Change (ROC)

    Measures percentage change over period.
    Positive = upward momentum, Negative = downward momentum.

    Formula:
        ROC = ((Close - Close_n_periods_ago) / Close_n_periods_ago) * 100

    Usage:
        roc = ROC(period=10)
        df = roc.calculate(data)
    """

    def __init__(self, period: int = 10, column: str = 'close'):
        super().__init__(name='ROC')
        self.period = period
        self.column = column
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ROC."""
        self.validate_data(data, [self.column])

        df = data.copy()

        df[f'roc_{self.period}'] = (
                (df[self.column] - df[self.column].shift(self.period)) /
                df[self.column].shift(self.period) * 100
        )

        return df