import streamlit as st
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import pandas as pd
from vnstock3 import Vnstock

def vietnam_stock_prediction():
    # List of popular stock tickers
    ticker_list = ["VIC", "VHM", "VNM", "VCB", "HPG"]

    # Streamlit App
    st.title("Stock Price Prediction with ARIMA")

    # Searchable dropdown for stock tickers
    ticker = st.selectbox("Select stock ticker:", ticker_list)

    end_date = st.date_input("End date")

    # Download data to get the available date range
    if ticker and end_date:
        try:
            data1 = Vnstock().stock(symbol=ticker, source='VCI')
            ten_years_ago = pd.Timestamp(end_date).tz_localize(None) - timedelta(days=365*10)
            default_start_date = ten_years_ago

            # Allow the user to modify the start date
            start_date = st.date_input("Start date", value=default_start_date)

            # Convert start_date and end_date to timezone-naive
            start_date = pd.Timestamp(start_date).tz_localize(None).strftime('%Y-%m-%d')
            end_date = pd.Timestamp(end_date).tz_localize(None).strftime('%Y-%m-%d')

            # Calculate the time difference between start_date and end_date
            time_difference = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
            ten_years_in_days = 365 * 10

            # Check if the data range is less than 10 years
            if time_difference < ten_years_in_days:
                st.warning(f"The selected date range has less than 10 years of data ({time_difference // 365} years).")

            # Fetch historical data
            data = data1.quote.history(start=start_date, end=end_date)

            # Check if data is returned
            if data.empty:
                st.error("No data returned for the selected date range. Please adjust your dates or try another ticker.")
                return

            # Display the plot of the stock's historical data
            st.subheader(f"Historical Data for {ticker}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['time'], data['close'], label='Historical Price', color='blue')
            ax.set_title(f'Historical Stock Price for {ticker}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            # ARIMA model input section
            st.subheader("ARIMA Model Configuration")
            p = st.number_input("Enter p (autoregressive term):", min_value=0, max_value=10, value=5)
            d = st.number_input("Enter d (difference term):", min_value=0, max_value=2, value=1)
            q = st.number_input("Enter q (moving average term):", min_value=0, max_value=10, value=0)

            if st.button('Run ARIMA Model'):
                # ARIMA Model
                model = ARIMA(data['close'], order=(p, d, q))
                model_fit = model.fit()

                # Predictions
                y_predicted = model_fit.predict(start=0, end=len(data) - 1)

                # Compute RMSE
                rmse = np.sqrt(mean_squared_error(data['close'], y_predicted))

                # Plot actual vs predicted
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data['time'], data['close'], label='Actual Price', color='blue')
                ax.plot(data['time'], y_predicted, label='Predicted Price (ARIMA)', color='orange')
                ax.set_title(f'Stock Price Prediction for {ticker}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                # Display RMSE
                st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    vietnam_stock_prediction()
