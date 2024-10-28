import streamlit as st

st.title("Dự Đoán Giá Cổ Phiếu")

# Tạo menu chọn lựa
menu = st.sidebar.selectbox("Chọn Chức Năng", ["Cổ Phiếu Toàn Cầu", "Cổ Phiếu Việt Nam"])

if menu == "Cổ Phiếu Toàn Cầu":
    from global_stock import global_stock_prediction
    global_stock_prediction()
else:
    from vietnam_stock import vietnam_stock_prediction
    vietnam_stock_prediction()
