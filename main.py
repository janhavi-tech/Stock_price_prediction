import yfinance as yf
import pandas as pd
import streamlit as st

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
