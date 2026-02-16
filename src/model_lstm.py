import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(data, seq_length=60):
    """
    Converts a time series into sequences for LSTM training.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(train_data, seq_length=60, epochs=10, batch_size=32):
    """
    Trains an LSTM model on the provided data.
    """
    # 1. Scale the data (LSTMs are sensitive to scale)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # 2. Create Sequences
    X_train, y_train = create_sequences(train_scaled, seq_length)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # 3. Build Model Architecture
    model = Sequential()
    # Layer 1: LSTM with 50 units
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Prevent overfitting
    
    # Layer 2: LSTM with 50 units
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer 3: Output Layer (Prediction)
    model.add(Dense(units=1))
    
    # 4. Compile and Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Training LSTM Model...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, scaler, history

def predict_lstm(model, scaler, test_data, seq_length=60):
    """
    Generates predictions using the trained LSTM model.
    """
    inputs = test_data.values.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)
    
    X_test = []
    # Create sequences
    for i in range(seq_length, len(inputs_scaled)):
        X_test.append(inputs_scaled[i-seq_length:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predict
    predicted_price_scaled = model.predict(X_test)
    
    # Inverse transform to get actual prices
    predicted_prices = scaler.inverse_transform(predicted_price_scaled)
    
    return predicted_prices