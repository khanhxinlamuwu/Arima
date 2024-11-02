import streamlit as st
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd


st.title("Dự Đoán Giá Cổ Phiếu Toàn Cầu với ARIMA")

# Danh sách mã cổ phiếu toàn cầu với tên tập đoàn
global_ticker_dict = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc."
}

# Tạo danh sách mã cổ phiếu có kèm tên tập đoàn
ticker_display_list = [f"{symbol} - {name}" for symbol, name in global_ticker_dict.items()]

# Chọn mã cổ phiếu với tên tập đoàn
selected_ticker = st.selectbox("Chọn mã cổ phiếu toàn cầu:", ticker_display_list)
ticker = selected_ticker.split(" - ")[0]  # Lấy mã cổ phiếu từ chuỗi được chọn
end_date = st.date_input("Ngày kết thúc")

# Lấy dữ liệu
if ticker and end_date:
    data = yf.download(ticker, end=end_date)
    if data.empty:
        st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
    else:
        # Hiển thị biểu đồ dữ liệu lịch sử
        st.subheader(f"Dữ Liệu Lịch Sử cho {global_ticker_dict[ticker]} ({ticker})")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Giá Lịch Sử', color='blue')
        ax.set_title(f'Giá Cổ Phiếu Lịch Sử cho {global_ticker_dict[ticker]} ({ticker})')
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Phân tích tính dừng của chuỗi
        st.subheader("Kiểm Định Tính Dừng của Chuỗi Thời Gian")
        close_prices = data['Close'].dropna()
        result = adfuller(close_prices)
        st.write("Giá trị p-value của kiểm định ADF:", result[1])
        if result[1] < 0.05:
            st.write("Chuỗi dữ liệu là dừng (theo kiểm định ADF).")
            st.write("=> Không cần sai phân, có thể chọn bậc \( d = 0 \).")
        else:
            st.write("Chuỗi dữ liệu không dừng. Cần sai phân để làm chuỗi ổn định.")
            st.write("=> Chọn bậc \( d = 1 \) và thử lại kiểm định.")

        # Vẽ biểu đồ ACF và PACF
        st.subheader("Biểu Đồ ACF và PACF để Xác Định p và q")
        fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
        plot_acf(close_prices, lags=40, ax=ax_acf)
        ax_acf.set_title("Biểu Đồ ACF")
        st.pyplot(fig_acf)

        fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
        plot_pacf(close_prices, lags=40, ax=ax_pacf)
        ax_pacf.set_title("Biểu Đồ PACF")
        st.pyplot(fig_pacf)

        # Hướng dẫn chọn tham số
        st.subheader("Hướng Dẫn Chọn Tham Số ARIMA")
        st.write("""
        - **Chọn bậc \( d \)**: Dựa trên kết quả kiểm định ADF. Nếu chuỗi không dừng (p-value > 0.05), thử \( d = 1 \) và kiểm tra lại.
        - **Chọn bậc \( p \) (AR)**: Quan sát biểu đồ PACF. Chọn \( p \) là độ trễ (lag) mà các giá trị PACF bắt đầu cắt giảm về 0.
        - **Chọn bậc \( q \) (MA)**: Quan sát biểu đồ ACF. Chọn \( q \) là độ trễ (lag) mà các giá trị ACF bắt đầu cắt giảm về 0.
        """)

        # Cấu hình mô hình ARIMA
        st.subheader("Cấu Hình Mô Hình ARIMA")
        p = st.number_input("Nhập p (thành phần hồi quy):", min_value=0, max_value=10, value=5)
        d = st.number_input("Nhập d (thành phần sai khác):", min_value=0, max_value=2, value=1)
        q = st.number_input("Nhập q (thành phần trung bình động):", min_value=0, max_value=10, value=0)

        if st.button('Chạy Mô Hình ARIMA'):
            model = ARIMA(close_prices, order=(p, d, q))
            model_fit = model.fit()
            y_predicted = model_fit.predict(start=close_prices.index[0], end=close_prices.index[-1])
            rmse = np.sqrt(mean_squared_error(close_prices, y_predicted))

            # Hiển thị biểu đồ giá thực tế so với giá dự đoán
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(close_prices, label='Giá Thực Tế', color='blue')
            ax.plot(close_prices.index, y_predicted, label='Giá Dự Đoán (ARIMA)', color='orange')
            ax.set_title(f'Dự Đoán Giá Cổ Phiếu cho {global_ticker_dict[ticker]} ({ticker})')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Giá')
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            st.write(f"Sai số trung bình bình phương căn (RMSE): {rmse}")


