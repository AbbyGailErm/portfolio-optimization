import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import clean_data, scale_data


# ---------------------------------------------------------------------------
# clean_data tests
# ---------------------------------------------------------------------------

def test_clean_data_fills_missing_values():
    """clean_data should fill NaN values using forward- then backward-fill."""
    df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [np.nan, 2.0, 4.0]})
    result = clean_data(df)
    assert result.isnull().sum().sum() == 0


def test_clean_data_converts_index_to_datetime():
    """clean_data should convert a string index to a DatetimeIndex."""
    df = pd.DataFrame(
        {'Close': [100.0, 101.0, 102.0]},
        index=['2023-01-01', '2023-01-02', '2023-01-03'],
    )
    result = clean_data(df)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_clean_data_preserves_datetimeindex():
    """clean_data should leave an existing DatetimeIndex unchanged."""
    dates = pd.date_range('2023-01-01', periods=3)
    df = pd.DataFrame({'Close': [100.0, 101.0, 102.0]}, index=dates)
    result = clean_data(df)
    assert isinstance(result.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(result.index, dates)


def test_clean_data_raises_on_non_dataframe():
    """clean_data should raise TypeError for non-DataFrame input."""
    with pytest.raises(TypeError):
        clean_data([1, 2, 3])


def test_clean_data_raises_on_empty_dataframe():
    """clean_data should raise ValueError for an empty DataFrame."""
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame())


# ---------------------------------------------------------------------------
# scale_data tests
# ---------------------------------------------------------------------------

def test_scale_data_range():
    """scale_data should scale the specified columns to the range [0, 1]."""
    df = pd.DataFrame({'A': [10.0, 20.0, 30.0], 'B': [100.0, 200.0, 300.0]})
    scaled, _ = scale_data(df, ['A', 'B'])
    assert scaled['A'].min() == pytest.approx(0.0)
    assert scaled['A'].max() == pytest.approx(1.0)
    assert scaled['B'].min() == pytest.approx(0.0)
    assert scaled['B'].max() == pytest.approx(1.0)


def test_scale_data_unscaled_columns_unchanged():
    """scale_data should not modify columns that are not in the columns list."""
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [10.0, 20.0, 30.0]})
    scaled, _ = scale_data(df, ['A'])
    pd.testing.assert_series_equal(scaled['B'], df['B'])


def test_scale_data_returns_scaler():
    """scale_data should return a fitted MinMaxScaler as the second element."""
    from sklearn.preprocessing import MinMaxScaler
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
    _, scaler = scale_data(df, ['A'])
    assert isinstance(scaler, MinMaxScaler)
