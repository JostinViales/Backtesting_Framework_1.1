"""
Base Indicator Class - Foundation for all technical indicators.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union, List


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.

    All indicators inherit from this class and must implement:
    - calculate() method
    """

    def __init__(self, name: str = None):
        """
        Initialize indicator.

        Args:
            name: Optional custom name for the indicator
        """
        self.name = name or self.__class__.__name__
        self._params = {}

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator and add columns to dataframe.

        Args:
            data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            DataFrame with indicator column(s) added
        """
        pass

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Allow indicator to be called like a function."""
        return self.calculate(data)

    def validate_data(self, data: pd.DataFrame, required_columns: List[str] = None):
        """
        Validate input data has required columns.

        Args:
            data: Input DataFrame
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        if required_columns is None:
            required_columns = ['close']

        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if data.empty:
            raise ValueError("Input DataFrame is empty")

    def get_params(self) -> dict:
        """Get indicator parameters."""
        return self._params.copy()

    def __repr__(self):
        params_str = ', '.join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.name}({params_str})"


class TrendIndicator(BaseIndicator):
    """Base class for trend-following indicators."""
    pass


class MomentumIndicator(BaseIndicator):
    """Base class for momentum indicators."""
    pass


class VolatilityIndicator(BaseIndicator):
    """Base class for volatility indicators."""
    pass


class VolumeIndicator(BaseIndicator):
    """Base class for volume indicators."""
    pass


class StatisticalIndicator(BaseIndicator):
    """Base class for statistical indicators."""
    pass