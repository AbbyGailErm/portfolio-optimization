import yfinance as yf
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler


# Setup logging to track what happens
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_financial_data(tickers, start_date, end_date):
    """
    Fetches data from Yahoo Finance.
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

def clean_data(df):
    """
    Fixes missing values and ensures the index is a date.
    """
    # Forward fill: If data is missing on Tuesday, use Monday's price.
    df = df.ffill()
    # Backward fill: If the first day is missing, use the second day's price.
    df = df.bfill()
    
    # Ensure the index (the dates) are actually read as Dates, not text
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    return df

def scale_data(df, columns):
    """
    Scales data to range [0, 1] using MinMaxScaler.
    Useful for LSTM models or neural networks.
    """
    scaler = MinMaxScaler()
    scaled_data = df.copy()
    scaled_data[columns] = scaler.fit_transform(df[columns])
    logging.info(f"Data scaled for columns: {columns}")
    return scaled_data, scaler