import streamlit as st
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from vnstock3 import Vnstock  # Đảm bảo vnstock3 đã được cài đặt

# Khởi tạo đối tượng vnstock3 client
client = Vnstock()

# Streamlit App
st.title("Dự Đoán Giá Cổ Phiếu với ARIMA")

# Chọn Thị Trường
market_choice = st.selectbox("Chọn Thị Trường:", ["Cổ Phiếu Toàn Cầu", "Cổ Phiếu Việt Nam"])

# Danh sách mã cổ phiếu cho cổ phiếu toàn cầu và Việt Nam
global_ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
vietnam_ticker_list = ["VIC", "VHM", "VNM", "VCB", "HPG", "FPT", "MSN", "TPB"]  # Danh sách cổ phiếu Việt Nam

# Chọn mã cổ phiếu dựa trên thị trường đã chọn
if market_choice == "Cổ Phiếu Toàn Cầu":
    ticker = st.selectbox("Chọn mã cổ phiếu toàn cầu:", global_ticker_list)
else:
    ticker = st.selectbox("Chọn mã cổ phiếu Việt Nam:", vietnam_ticker_list)

# Hàm lấy dữ liệu cổ phiếu Việt Nam
def fetch_vietnam_stock(ticker):
    try:
        # Lấy dữ liệu gần nhất
        data = client.stock_price(ticker)
        if data.empty:
            st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
            return pd.DataFrame()
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        return data
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {e}")
        return pd.DataFrame()

# Ngày kết thúc
end_date = st.date_input("Ngày kết thúc")

# Lấy dữ liệu cho mã cổ phiếu đã chọn và ngày
if ticker and end_date:
    if market_choice == "Cổ Phiếu Toàn Cầu":
        data = yf.download(ticker, end=end_date)
    else:
        data = fetch_vietnam_stock(ticker)

    # Kiểm tra nếu dữ liệu rỗng
    if data.empty:
        st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
    else:
        # Hiển thị biểu đồ dữ liệu lịch sử của cổ phiếu
        st.subheader(f"Dữ Liệu Lịch Sử cho {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['close'], label='Giá Lịch Sử', color='blue')
        ax.set_title(f'Giá Cổ Phiếu Lịch Sử cho {ticker}')
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Cấu hình mô hình ARIMA
        st.subheader("Cấu Hình Mô Hình ARIMA")
        p = st.number_input("Nhập p (thành phần hồi quy):", min_value=0, max_value=10, value=5)
        d = st.number_input("Nhập d (thành phần sai khác):", min_value=0, max_value=2, value=1)
        q = st.number_input("Nhập q (thành phần trung bình động):", min_value=0, max_value=10, value=0)

        if st.button('Chạy Mô Hình ARIMA'):
            # Mô hình ARIMA
            close_prices = data['close']
            model = ARIMA(close_prices, order=(p, d, q))
            model_fit = model.fit()

            # Dự đoán
            y_predicted = model_fit.predict(start=close_prices.index[0], end=close_prices.index[-1])

            # RMSE
            rmse = np.sqrt(mean_squared_error(close_prices, y_predicted))

            # Hiển thị biểu đồ giá thực tế so với giá dự đoán
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(close_prices, label='Giá Thực Tế', color='blue')
            ax.plot(close_prices.index, y_predicted, label='Giá Dự Đoán (ARIMA)', color='orange')
            ax.set_title(f'Dự Đoán Giá Cổ Phiếu cho {ticker}')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Giá')
            ax.legend()
            ax.grid()

            # Hiển thị biểu đồ dự đoán
            st.pyplot(fig)

            # Hiển thị RMSE
            st.write(f"Sai số trung bình bình phương căn (RMSE): {rmse}")
