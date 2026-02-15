from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

def save_model(model, filepath='../models/arima_model.pkl'):
    """
    Saves the trained ARIMA model to a file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def split_data(series, split_ratio=0.8):
    """
    Splits data chronologically. 
    We CANNOT shuffle time series data because order matters.
    """
    split_point = int(len(series) * split_ratio)
    train = series[:split_point]
    test = series[split_point:]
    return train, test

def train_arima(train_data):
    """
    Finds the best ARIMA model.
    """
    # auto_arima acts like a grid search to find optimal p, d, q
    model = auto_arima(train_data, 
                       seasonal=False, # Set to True if you want SARIMA
                       trace=True,     # Shows us the search progress
                       error_action='ignore', 
                       suppress_warnings=True)
    return model

def evaluate_forecast(test_data, predictions):
    """
    Calculates error metrics.
    """
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return {'MAE': mae, 'RMSE': rmse}