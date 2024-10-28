import streamlit as st

st.title("Stock Price Prediction with ARIMA")

# Tạo menu chọn lựa
menu = st.sidebar.selectbox("Select", ["Global", "Vietnam"])

if menu == "Global":
    from global_stock import global_stock_prediction
    global_stock_prediction()
else:
    from vietnam_stock import vietnam_stock_prediction
    vietnam_stock_prediction()
