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


# Streamlit App
st.title("Stock Price Prediction with ARIMA")
# Select Market
market_choice = st.selectbox("Choose Market:", ["Global Stocks", "Vietnam Stocks"])

# Global and Vietnam ticker lists (for demo purposes; expand as needed)
global_ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
vietnam_ticker_list = ["VIC", "VHM", "VNM", "VCB", "HPG"]

# Choose ticker based on selected market
client = Vnstock()  # Khởi tạo đối tượng client

if market_choice == "Global Stocks":
    ticker = st.selectbox("Select global stock ticker:", global_ticker_list)
    data_fetch_function = yf.download
else:
    ticker = st.selectbox("Select Vietnam stock ticker:", vietnam_ticker_list)
    data_fetch_function = client.stock_price  # Sử dụng hàm fetch của vnstock3


end_date = st.date_input("End date")

# Fetch data for selected ticker and date
if ticker and end_date:
    if market_choice == "Global Stocks":
        data = data_fetch_function(ticker)
    else:
        data = data_fetch_function(ticker)

    # Check if data is empty
    if data.empty:
        st.error("No data found for this ticker.")
    else:
        # Get the first available date and localize it to None (tz-naive)
        first_date = data.index[0].tz_localize(None) if market_choice == "Global Stocks" else pd.to_datetime(data['date']).min()
        ten_years_ago = pd.Timestamp(end_date).tz_localize(None) - timedelta(days=365*10)

        # Check if data has less than 10 years
        if pd.Timestamp(first_date) > pd.Timestamp(ten_years_ago):
            st.warning(f"Data for {ticker} has less than 10 years. Available from {first_date}.")
            if not st.checkbox("Do you want to continue with the available data?"):
                st.stop()

        # Set default start_date to the maximum between ten_years_ago or first available date
        default_start_date = max(ten_years_ago, first_date)

        # Allow the user to modify the start date
        start_date = st.date_input("Start date", value=default_start_date)

        # Convert start_date and end_date to timezone-naive
        start_date = pd.Timestamp(start_date).tz_localize(None)
        end_date = pd.Timestamp(end_date).tz_localize(None)

        # Calculate the time difference between start_date and end_date
        time_difference = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        ten_years_in_days = 365 * 10

        # Check if the data range is less than 10 years
        if time_difference < ten_years_in_days:
            st.warning(f"The selected date range has less than 10 years of data ({time_difference // 365} years).")

        # Display the plot of the stock's historical data
        st.subheader(f"Historical Data for {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if market_choice == "Global Stocks":
            ax.plot(data['Close'], label='Historical Price', color='blue')
        else:
            ax.plot(pd.to_datetime(data['date']), data['close'], label='Historical Price', color='blue')

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
            # Download data within the selected date range
            if market_choice == "Global Stocks":
                data = yf.download(ticker, start=start_date, end=end_date)
            else:
                data = client.stock_price(ticker=ticker, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))

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
            ax.plot(close_prices, label='Actual Price', color='blue')
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
