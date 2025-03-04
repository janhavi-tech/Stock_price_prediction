# 📈 Stock Price Prediction

This project predicts stock prices using machine learning models like **LSTM, ARIMA, and Prophet**. It includes a **FastAPI backend** for model inference and a **Streamlit frontend** for visualization.

---

## 📁 Project Structure

stock-price-prediction/ 
├── main.py # FastAPI Backend for Predictions 
├── model.py # ML Models for Stock Prediction
├── app.py # Streamlit Frontend for Visualization ├── requirements.txt # Dependencies 
├── README.md # Project Documentation 
└── .gitignore # Files to Ignore in Git

## 🚀 Features

- Fetch stock price data using `yfinance`
- Predict future stock prices using **LSTM, ARIMA, and Prophet** models
- Interactive **Streamlit dashboard** for visualization
- FastAPI backend to handle model inference requests
- Beautifully plotted **graphs using Matplotlib & Seaborn**

---

## 🛠️ Tech Stack

- **Python**
- **FastAPI** (Backend)
- **Streamlit** (Frontend)
- **Pandas & NumPy** (Data Processing)
- **Matplotlib & Seaborn** (Visualization)
- **TensorFlow/Keras, ARIMA, Prophet** (ML Models)
- **yfinance** (Stock Data Fetching)

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/janhavi-tech/Stock_price_prediction.git
   cd Stock_price_prediction
Set up a virtual environment:

bash
Copy
Edit
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Start the FastAPI Backend:

bash
Copy
Edit
uvicorn main:app --reload
The backend will run on http://127.0.0.1:8000.

Start the Streamlit Frontend:

bash
Copy
Edit
streamlit run app.py
🖥️ How to Use
Enter a stock ticker (e.g., AAPL for Apple) via the sidebar.
Choose a prediction model (LSTM, ARIMA, or Prophet).
Click "Run Prediction" to fetch stock data and generate predictions.
View the predicted stock prices in a table and graph.
🔗 API Endpoints
Method	Endpoint	Description
GET	/predict/lstm/{ticker}	Predict stock price using LSTM
GET	/predict/arima/{ticker}	Predict stock price using ARIMA
GET	/predict/prophet/{ticker}	Predict stock price using Prophet
Data Source
The app uses Yahoo Finance (via the yfinance library) to fetch historical stock data.