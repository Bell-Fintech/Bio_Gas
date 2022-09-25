import streamlit as st
import pickle
import numpy as np



def cal1():
    with st.form("calc"):
        weight = st.number_input("请输入沼气原料重量：", value=10)
        price = st.number_input("请输入原料单价：", value=10)
        type = st.radio('原料类型:', ('干', '湿'))
        discount_factor = st.number_input('请输入损失因子：')
        total_price = weight * price * discount_factor
        total_price = round(total_price, 2)
        message = {total_price}

        if type == '干':
            message = total_price * 0.88
        elif type == '湿':
            message = total_price * 0.75

        if st.form_submit_button("计算"):
            f"总价是{message}"