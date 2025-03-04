# 📈 Stock Price Prediction

This project predicts stock prices using machine learning models like **LSTM, ARIMA, and Prophet**. It includes a **FastAPI backend** for model inference and a **Streamlit frontend** for visualization.

---
**📁 Project Structure**
stock-price-prediction/ 
├── main.py               # FastAPI Backend for Predictions 
├── model.py              # ML Models for Stock Prediction 
├── app.py                # Streamlit Frontend for Visualization 
├── requirements.txt      # Dependencies 
├── README.md             # Project Documentation 
├── .gitignore            # Files to Ignore in Git



## **🚀 Features**
✅ Fetch stock price data using `yfinance`  
✅ Predict future stock prices using **LSTM, ARIMA, Prophet** models  
✅ Interactive **Streamlit dashboard** for visualization  
✅ FastAPI backend to handle model inference requests  
✅ Beautifully plotted **graphs using Matplotlib & Seaborn**  

---

## **🛠️ Tech Stack**
- **Python**
- **FastAPI** (Backend)
- **Streamlit** (Frontend)
- **Pandas & NumPy** (Data Processing)
- **Matplotlib & Seaborn** (Visualization)
- **TensorFlow/Keras, ARIMA, Prophet** (ML Models)
- **yfinance** (Stock Data Fetching)

---

## **📦 Installation**
1. Clone the repository

git clone https://github.com/janhavi30499/SPP.git
cd SPP


2.  Set a virtual environment
python -m venv venv


3.Install dependencies
pip install -r requirements.txt

4. Start the FastAPI Backend
uvicorn main:app --reload
It should run on http://127.0.0.1:8000.

5️. Start the Streamlit Frontend
streamlit run app.py



🖥️ How to Use
1️⃣ Enter a stock ticker (e.g., AAPL for Apple).
2️⃣ Choose a prediction model (LSTM, ARIMA, Prophet).
3️⃣ Click "Predict" to fetch stock prices and generate predictions.
4️⃣ View the predicted stock prices in a table and graph.

🔗 API Endpoints
Method	Endpoint	Description
GET	/predict/lstm/{ticker}	Predict stock price using LSTM
GET	/predict/arima/{ticker}	Predict stock price using ARIMA
GET	/predict/prophet/{ticker}	Predict stock price using Prophet
