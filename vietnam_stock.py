import streamlit as st
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import pandas as pd
from vnstock3 import Vnstock

def vietnam_stock_prediction():


    # List of popular stock tickers (you can add more or load from an external source)
    ticker_list = ["VIC", "VHM", "VNM", "VCB", "HPG"]



    # Searchable dropdown for stock tickers
    ticker = st.selectbox("Select stock ticker:", ticker_list)

    end_date = st.date_input("End date")

    # Download data to get the available date range
    if ticker and end_date:
            data1 = Vnstock().stock(symbol=ticker, source='VCI')
            ten_years_ago = pd.Timestamp(end_date).tz_localize(None) - timedelta(days=365*10)
        
            # Set default start_date to the maximum between ten_years_ago or first available date
            default_start_date = ten_years_ago

            # Allow the user to modify the start date
            start_date = st.date_input("Start date", value=default_start_date)

            # Convert start_date and end_date to timezone-naive
            start_date = pd.Timestamp(start_date).tz_localize(None)
            end_date = pd.Timestamp(end_date).tz_localize(None)
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')
            
            # Calculate the time difference between start_date and end_date
            time_difference = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
            ten_years_in_days = 365 * 10

            # Check if the data range is less than 10 years
            if time_difference < ten_years_in_days:
                st.warning(f"The selected date range has less than 10 years of data ({time_difference // 365} years).")

            data=data1.quote.history(start=start_date, end=end_date)
            # Display the plot of the stock's historical data
            st.subheader(f"Historical Data for {ticker}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['time'], data['close'], label='Historical Price', color='blue')
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
            

                # ARIMA Model
                model = ARIMA(data['close'], order=(p, d, q))
                model_fit = model.fit()

                # Predictions
                data = data.set_index('time')

                # Predict using index positions
                start_index = 0
                end_index = len(data) - 1

                y_predicted = model_fit.predict(start=start_index, end=end_index)

                # Compute RMSE
                rmse = np.sqrt(mean_squared_error(data['close'], y_predicted))

                # Plot actual vs predicted
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data['close'], label='Actual Price', color='blue')
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
# Gọi hàm nếu chọn "Cổ Phiếu Việt Nam"
if __name__ == "__main__":
    vietnam_stock_prediction()
