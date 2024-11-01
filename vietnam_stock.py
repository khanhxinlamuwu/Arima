import streamlit as st
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import pandas as pd
from vnstock3 import Vnstock

def vietnam_stock_prediction():
    ticker_list = ["VIC"]#, "VHM", "VNM", "VCB", "HPG"]
    
    st.title("Dự đoán giá cổ phiếu với ARIMA")
    ticker = st.selectbox("Chọn mã cổ phiếu:", ticker_list)
    end_date = st.date_input("Ngày kết thúc")

    if ticker and end_date:
        data1 = Vnstock().stock(symbol=ticker, source='VCI')
        if data1 is None or not data1.quote:
            st.error("Không thể lấy dữ liệu cho mã cổ phiếu này. Vui lòng kiểm tra lại mã cổ phiếu hoặc thử lại sau.")
            return
        
        ten_years_ago = pd.Timestamp(end_date).tz_localize(None) - timedelta(days=365*10)
        default_start_date = ten_years_ago
        start_date = st.date_input("Ngày bắt đầu", value=default_start_date)

        start_date = pd.Timestamp(start_date).tz_localize(None)
        end_date = pd.Timestamp(end_date).tz_localize(None)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        
        time_difference = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        if time_difference < 0:
            st.error("Ngày bắt đầu không được sau ngày kết thúc.")
            return

        data = data1.quote.history(start=start_date, end=end_date)
        if data.empty:
            st.error("Dữ liệu không đủ để dự đoán. Vui lòng kiểm tra lại ngày tháng.")
            return
        
        st.subheader(f"Dữ liệu lịch sử cho {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['time'], data['close'], label='Giá lịch sử', color='blue')
        ax.set_title(f'Giá cổ phiếu lịch sử cho {ticker}')
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        st.subheader("Cấu hình mô hình ARIMA")
        p = st.number_input("Nhập p (hệ số tự hồi quy):", min_value=0, max_value=10, value=5)
        d = st.number_input("Nhập d (hệ số khác biệt):", min_value=0, max_value=2, value=1)
        q = st.number_input("Nhập q (hệ số trung bình trượt):", min_value=0, max_value=10, value=0)

        if st.button('Chạy mô hình ARIMA'):
            model = ARIMA(data['close'], order=(p, d, q))
            model_fit = model.fit()

            y_predicted = model_fit.predict(start=0, end=len(data) - 1)
            rmse = np.sqrt(mean_squared_error(data['close'], y_predicted))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['close'], label='Giá thực tế', color='blue')
            ax.plot(data.index, y_predicted, label='Giá dự đoán (ARIMA)', color='orange')
            ax.set_title(f'Dự đoán giá cổ phiếu cho {ticker}')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Giá')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            st.write(f"Giá trị RMSE: {rmse}")

if __name__ == "__main__":
    vietnam_stock_prediction()
