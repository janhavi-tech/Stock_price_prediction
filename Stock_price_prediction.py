import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import model as mod

# App Title
st.title("📈 Stock Price Prediction with Streamlit & ML")
st.sidebar.header("Stock Input")

# Sidebar: User Input
try:
    stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., Reliance.NS):", value="Reliance.NS")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-12"))
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
except Exception as e:
    st.sidebar.error(f"Invalid input: {e}")

# Load Stock Data using yfinance
@st.cache_data
def load_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("No data found for the given ticker and date range.")
        data = data.resample('M').last()  # Resample to monthly frequency
        return data
    except Exception as e:
        st.error(f"Failed to load stock data: {e}")
        return pd.DataFrame()

stock_data = load_stock_data(stock_ticker, start_date, end_date)

if not stock_data.empty:
    # Display Stock Data
    st.subheader("Stock Data")
    st.write(stock_data.head(20))

    # Plot Historical Closing Prices
    st.subheader("Stock Closing Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data.index, stock_data['Close'], color='blue', linewidth=2)
    ax.set_title(f"{stock_ticker} Stock Closing Prices (Monthly)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Closing Price (USD)", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    # Scale Data
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(stock_data[['Close']])
    train_size = int(len(scaled_close) * 0.8)
    test_data = scaled_close[train_size:]
    
    # Prepare data for LSTM (using a simple sequence based on indices)
    X = np.arange(len(scaled_close)).reshape(-1, 1)
    y = scaled_close.flatten()
    # Reshape for LSTM: (samples, time_steps, features)
    X_train = X[:train_size].reshape(-1, 1, 1)
    y_train = y[:train_size]
    X_test = X[train_size:].reshape(-1, 1, 1)

    # Get available models (only LSTM and ARIMA now)
    models = mod.get_models()
    
    # Initialize performance dictionaries
    predictions = {}
    errors = {}
    r2_scores = {}

    st.subheader("Model Comparison")
    for model_name, model_func in models.items():
        try:
            if model_name == "LSTM":
                pred = model_func(X_train, y_train, X_test, epochs=10, batch_size=16)
                predicted_prices = scaler.inverse_transform(pred.reshape(-1, 1))
            elif model_name == "ARIMA":
                series = stock_data['Close']
                steps = len(stock_data) - train_size
                pred = model_func(series, order=(5,1,0), steps=steps)
                predicted_prices = pred  # Already in original scale

            if model_name == "LSTM":
                actual_prices = scaler.inverse_transform(test_data.reshape(-1, 1))
            else:
                actual_prices = stock_data['Close'].iloc[train_size:]
            dates = stock_data.index[train_size:]
            
            rmse = sqrt(mean_squared_error(actual_prices, predicted_prices))
            r2 = r2_score(actual_prices, predicted_prices)
            errors[model_name] = rmse
            r2_scores[model_name] = r2
            
            st.write(f"**{model_name}:**")
            st.write(f"- RMSE: {rmse:.2f}")
            st.write(f"- R² Score: {r2:.2f}")
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(dates, actual_prices, label="Actual", color="green", linewidth=2)
            ax2.plot(dates, predicted_prices, label=f"Predicted ({model_name})", color="red", linestyle="--", linewidth=2)
            ax2.set_title(f"Actual vs Predicted Prices ({model_name})", fontsize=16)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("Closing Price (USD)", fontsize=12)
            ax2.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error processing model {model_name}: {e}")

    # Future Predictions
    st.subheader("Future Price Predictions")
    future_days = 12  # Predict for next 12 months
    future_dates = [stock_data.index[-1] + timedelta(days=i*30) for i in range(1, future_days + 1)]
    future_X = np.arange(len(scaled_close), len(scaled_close) + future_days).reshape(-1, 1)
    
    for model_name, model_func in models.items():
        try:
            if model_name == "LSTM":
                future_preds = model_func(X_train, y_train, future_X.reshape(-1, 1, 1), epochs=10, batch_size=16)
                future_prices = scaler.inverse_transform(future_preds.reshape(-1, 1))
            elif model_name == "ARIMA":
                series = stock_data['Close']
                future_preds = model_func(series, order=(5,1,0), steps=future_days)
                future_prices = future_preds

            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(stock_data.index, stock_data['Close'], label="Historical Data", color='green', linewidth=2)
            ax3.plot(future_dates, future_prices, label=f"Future Predictions ({model_name})", color='blue', linestyle="--", linewidth=2)
            ax3.set_title(f"Future Stock Price Prediction ({model_name})", fontsize=16)
            ax3.set_xlabel("Date", fontsize=12)
            ax3.set_ylabel("Closing Price (USD)", fontsize=12)
            ax3.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Failed to predict future prices with {model_name}: {e}")
    
    st.subheader("Model Performance Summary")
    performance_df = pd.DataFrame({"RMSE": errors, "R² Score": r2_scores})
    st.dataframe(performance_df)
else:
    st.error("No data to display. Please adjust the input parameters.")
