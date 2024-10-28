import streamlit as st
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import timedelta

import pandas as pd
from vnstock3 import Vnstock  # Ensure vnstock3 is installed

# Initialize vnstock3 client
client = Vnstock()

# Streamlit App
st.title("Stock Price Prediction with ARIMA")
# Select Market
market_choice = st.selectbox("Choose Market:", ["Global Stocks", "Vietnam Stocks"])

# Global and Vietnam ticker lists (for demo purposes; expand as needed)
global_ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
vietnam_ticker_list = ["VIC", "VHM", "VNM", "VCB", "HPG"]

# Choose ticker based on selected market
if market_choice == "Global Stocks":
    ticker = st.selectbox("Select global stock ticker:", global_ticker_list)
    data_fetch_function = yf.download
else:
    ticker = st.selectbox("Select Vietnam stock ticker:", vietnam_ticker_list)
    data_fetch_function = client.stock_price

end_date = st.date_input("End date")

# Fetch data for selected ticker and date
if ticker and end_date:
    if market_choice == "Global Stocks":
        data = data_fetch_function(ticker, end=end_date.strftime("%Y-%m-%d"))
    else:
        data = data_fetch_function(ticker=ticker, end_date=end_date.strftime("%Y-%m-%d"))

    # Check if data is empty
    if data is None or data.empty:
        st.error("No data found for this ticker.")
    else:
        # Ensure data processing is consistent for both data sources
        if market_choice == "Global Stocks":
            data['Date'] = data.index
        else:
            data['Date'] = pd.to_datetime(data['date'])

        # Display the plot of the stock's historical data
        st.subheader(f"Historical Data for {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if market_choice == "Global Stocks":
            ax.plot(data['Date'], data['Close'], label='Historical Price', color='blue')
        else:
            ax.plot(data['Date'], data['close'], label='Historical Price', color='blue')

        ax.set_title(f'Historical Stock Price for {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid()

        # Display historical plot
        st.pyplot(fig)

        # ARIMA model input section
        st.subheader("ARIMA Model Configuration")
        p = st.number_input("Enter p (autoregressive term):", min_value=0, max_value=10, value=5)
        d = st.number_input("Enter d (difference term):", min_value=0, max_value=2, value=1)
        q = st.number_input("Enter q (moving average term):", min_value=0, max_value=10, value=0)

        if st.button('Run ARIMA Model'):
            # ARIMA Model
            close_prices = data['Close'] if market_choice == "Global Stocks" else data['close']
            model = ARIMA(close_prices, order=(p, d, q))
            model_fit = model.fit()

            # Predictions
            y_predicted = model_fit.predict(start=close_prices.index[0], end=close_prices.index[-1])

            # RMSE
            rmse = np.sqrt(mean_squared_error(close_prices, y_predicted))

            # Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(close_prices.index, close_prices, label='Actual Price', color='blue')
            ax.plot(close_prices.index, y_predicted, label='Predicted Price (ARIMA)', color='orange')
            ax.set_title(f'Stock Price Prediction for {ticker}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid()

            # Display prediction plot
            st.pyplot(fig)

            # Display RMSE
            st.write(f"Root Mean Square Error (RMSE): {rmse}")
