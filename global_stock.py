import streamlit as st
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from datetime import timedelta

def global_stock_prediction():
    st.title("Stock Price Prediction with ARIMA")
    # Danh sách mã cổ phiếu toàn cầu
    ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "NFLX", "INTC", "AMD"]
 
    # Searchable dropdown for stock tickers
    ticker = st.selectbox("Select stock ticker:", ticker_list)
    end_date = st.date_input("End date")
    # Download data to get the available date range
    if ticker and end_date:
        data = yf.download(ticker)

        # Check if data is empty
        if data.empty:
            st.error("No data found for this ticker.")
        else:
            # Get the first available date and localize it to None (tz-naive)
            first_date = data.index[0].tz_localize(None)
            ten_years_ago = pd.Timestamp(end_date).tz_localize(None) - timedelta(days=365*10)

            # Check if data has less than 10 years
            if pd.Timestamp(first_date) > pd.Timestamp(ten_years_ago):
                st.warning(f"Data for {ticker} has less than 10 years. Available from {first_date.date()}.")
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
            ax.plot(data['Close'], label='Historical Price', color='blue')
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
                data = yf.download(ticker, start=start_date, end=end_date)

                # ARIMA Model
                model = ARIMA(data['Close'], order=(p, d, q))
                model_fit = model.fit()

                # Predictions
                y_predicted = model_fit.predict(start=data.index[0], end=data.index[-1])

                # RMSE
                rmse = np.sqrt(mean_squared_error(data['Close'], y_predicted))

                # Plot actual vs predicted
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data['Close'], label='Actual Price', color='blue')
                ax.plot(data.index, y_predicted, label='Predicted Price (ARIMA)', color='orange')
                ax.set_title(f'Stock Price Prediction for {ticker}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid()

                # Display prediction plot
                st.pyplot(fig)

                # Display RMSE
                st.write(f"Root Mean Square Error (RMSE): {rmse}")

# Gọi hàm nếu chọn "Cổ Phiếu Toàn Cầu"
if __name__ == "__main__":
    global_stock_prediction()
