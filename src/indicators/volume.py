"""
Volume Indicators - Analyze trading volume.
"""
import pandas as pd
import numpy as np
from src.indicators.base import VolumeIndicator


class VolumeMA(VolumeIndicator):
    """
    Volume Moving Average

    Average volume over period.
    Compare current volume to average to gauge interest.

    Usage:
        vol_ma = VolumeMA(period=20)
        df = vol_ma.calculate(data)
    """

    def __init__(self, period: int = 20):
        super().__init__(name='VolumeMA')
        self.period = period
        self._params = {'period': period}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume moving average."""
        self.validate_data(data, ['volume'])

        df = data.copy()
        df[f'volume_ma_{self.period}'] = df['volume'].rolling(window=self.period).mean()

        return df


class VWAP(VolumeIndicator):
    """
    Volume Weighted Average Price

    Average price weighted by volume.
    Institutions use this as benchmark.

    Formula:
        VWAP = Cumulative(Price * Volume) / Cumulative(Volume)

    Usage:
        vwap = VWAP()
        df = vwap.calculate(data)
    """

    def __init__(self):
        super().__init__(name='VWAP')
        self._params = {}

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP."""
        self.validate_data(data, ['high', 'low', 'close', 'volume'])

        df = data.copy()

        # Typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # VWAP
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return df