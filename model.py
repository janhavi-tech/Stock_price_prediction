import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA

def train_and_predict_lstm(X_train, y_train, X_test, epochs=20, batch_size=32):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    predictions = model.predict(X_test)
    return predictions

def train_and_predict_arima(series, order=(5, 1, 0), steps=10):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=steps)
    return predictions

def get_models():
    models = {
        'LSTM': train_and_predict_lstm,
        'ARIMA': train_and_predict_arima
    }
    return models
