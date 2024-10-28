import streamlit as st
import numpy as np
from vnstock3 import Vnstock  # Đảm bảo vnstock3 đã được cài đặt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

def vietnam_stock_prediction():
    st.title("Dự Đoán Giá Cổ Phiếu Việt Nam với ARIMA")
    
    # Khởi tạo đối tượng vnstock3 client
    client = Vnstock()
    
    # Danh sách mã cổ phiếu Việt Nam
    vietnam_ticker_list = ["VIC", "VHM", "VNM", "VCB", "HPG"]
    
    # Chọn mã cổ phiếu
    ticker = st.selectbox("Chọn mã cổ phiếu Việt Nam:", vietnam_ticker_list)
    end_date = st.date_input("Ngày kết thúc")

    # Lấy dữ liệu
    if ticker and end_date:
        data = client.stock_price(ticker)
        if data.empty:
            st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
        else:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Hiển thị biểu đồ dữ liệu lịch sử
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
                close_prices = data['close']
                model = ARIMA(close_prices, order=(p, d, q))
                model_fit = model.fit()
                y_predicted = model_fit.predict(start=close_prices.index[0], end=close_prices.index[-1])
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
                st.pyplot(fig)
                st.write(f"Sai số trung bình bình phương căn (RMSE): {rmse}")

# Gọi hàm nếu chọn "Cổ Phiếu Việt Nam"
if __name__ == "__main__":
    vietnam_stock_prediction()
