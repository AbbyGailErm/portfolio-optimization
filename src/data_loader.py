import yfinance as yf
import pandas as pd
import logging
from typing import List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler


# Setup logging to track what happens
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_financial_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Fetches historical price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols (e.g. ``['AAPL', 'GOOG']``).
        start_date: Start date string in ``YYYY-MM-DD`` format.
        end_date: End date string in ``YYYY-MM-DD`` format.

    Returns:
        A DataFrame of historical price data, or ``None`` on failure.
    """
    logging.info(f"Fetching {tickers} data...")
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        
        # If no data comes back, warn the user
        if data.empty:
            logging.error("No data fetched! Check your internet or ticker names.")
            return None
            
        logging.info("Data fetched successfully.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values and ensures the index is a DatetimeIndex.

    Missing values are filled using forward-fill first, then backward-fill
    so that leading NaNs are also resolved.

    Args:
        df: Raw input DataFrame, potentially containing NaN values.

    Returns:
        A cleaned DataFrame with no NaN values and a DatetimeIndex.

    Raises:
        TypeError: If ``df`` is not a :class:`pandas.DataFrame`.
        ValueError: If the DataFrame is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Forward fill: If data is missing on Tuesday, use Monday's price.
    df = df.ffill()
    # Backward fill: If the first day is missing, use the second day's price.
    df = df.bfill()
    
    # Ensure the index (the dates) are actually read as Dates, not text
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    return df

def scale_data(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Scales selected columns to the range [0, 1] using MinMaxScaler.

    Useful for preparing data for LSTM models or other neural networks that
    require normalized inputs.

    Args:
        df: Input DataFrame containing the columns to scale.
        columns: List of column names to scale.

    Returns:
        A tuple of ``(scaled_df, scaler)`` where ``scaled_df`` is a copy of
        ``df`` with the specified columns scaled, and ``scaler`` is the fitted
        :class:`~sklearn.preprocessing.MinMaxScaler` instance.
    """
    scaler = MinMaxScaler()
    scaled_data = df.copy()
    scaled_data[columns] = scaler.fit_transform(df[columns])
    logging.info(f"Data scaled for columns: {columns}")
    return scaled_data, scaler