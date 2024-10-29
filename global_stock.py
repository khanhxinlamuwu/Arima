import streamlit as st
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from pmdarima import auto_arima  # Import Auto-ARIMA

def global_stock_prediction():
    st.title("Dự Đoán Giá Cổ Phiếu Toàn Cầu với ARIMA")

    # Danh sách mã cổ phiếu toàn cầu cùng tên tập đoàn
    global_ticker_list = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc."
    }
    
    # Chọn mã cổ phiếu
    ticker = st.selectbox("Chọn mã cổ phiếu toàn cầu:", list(global_ticker_list.keys()))
    end_date = st.date_input("Ngày kết thúc")

    # Lấy dữ liệu
    if ticker and end_date:
        data = yf.download(ticker, end=end_date)
        if data.empty:
            st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này.")
        else:
            # Hiển thị tên tập đoàn
            st.subheader(f"Cổ phiếu của {global_ticker_list[ticker]} ({ticker})")

            # Hiển thị biểu đồ dữ liệu lịch sử
            st.subheader(f"Dữ Liệu Lịch Sử cho {ticker}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Close'], label='Giá Lịch Sử', color='blue')
            ax.set_title(f'Giá Cổ Phiếu Lịch Sử cho {ticker}')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Giá')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            # Chạy Auto-ARIMA để tìm tham số tối ưu cho p, d, q
            st.subheader("Chạy Auto-ARIMA để tìm tham số tối ưu")
            with st.spinner("Đang chạy Auto-ARIMA..."):
                auto_arima_model = auto_arima(data['Close'], start_p=0, start_q=0,
                                              max_p=5, max_q=5, seasonal=False,
                                              trace=True,  # Hiển thị quá trình thử nghiệm
                                              error_action='ignore',
                                              suppress_warnings=True,
                                              stepwise=True)

            # In ra tham số tối ưu
            st.write("Tham số tối ưu được tìm thấy bởi Auto-ARIMA:")
            st.write(auto_arima_model.summary())

            # Sử dụng các tham số tối ưu để dự đoán
            p, d, q = auto_arima_model.order
            st.write(f"Sử dụng mô hình ARIMA với tham số (p={p}, d={d}, q={q})")

            if st.button('Chạy Mô Hình ARIMA'):
                close_prices = data['Close']
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
                
                # Hiển thị RMSE
                st.write(f"Sai số trung bình bình phương căn (RMSE): {rmse}")

# Gọi hàm nếu chọn "Cổ Phiếu Toàn Cầu"
if __name__ == "__main__":
    global_stock_prediction()
