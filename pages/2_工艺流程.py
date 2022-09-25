import streamlit as st

st.set_page_config(layout="wide")

st.sidebar.title("沼气学术会议")
st.sidebar.info(
    """
    中国沼气学会学术年会:     
    <http://www.biogaschina.com.cn>         
    能源系统与电气工程会议:       
    <https://www.ncsti.gov.cn>
    """
)

st.sidebar.title("友情链接")
st.sidebar.info(
    """
    国际沼气网: <http://www.biogasintel.com>       
    中国农业农村部: <http://www.moa.gov.cn>          
    香港可再生能源网: <https://re.emsd.gov.hk>

    """
)

st.title("工艺流程计算模块")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.subheader('发酵罐1 - 水力停留时间')

with st.form("cal1"):
    pot1_cal1 = st.number_input("发酵罐1 - 有效容积：", value=1)
    pot2_cal1 = st.number_input("发酵罐2 - 有效容积：", value=1)
    pot3_cal1 = st.number_input("发酵罐3 - 有效容积：", value=1)
    input_daily_cal1 = st.number_input("进料量FM（吨/天） - 日进料量：", value=1)
    product_biogas_efficiency_cal1 = (pot1_cal1+pot2_cal1+pot3_cal1)/input_daily_cal1
    product_biogas_efficiency_cal1 = round(product_biogas_efficiency_cal1, 2)
    message1 = product_biogas_efficiency_cal1

    if st.form_submit_button("计算"):
        #st.subheader(message1)
        st.success("success 🎉")

        r24c1, r24c2, r24c3,r24c4, r24c5 = st.columns(5)
        with r24c1:
            st.info("发酵罐1 - 有效容积")
        with r24c2:
            st.info("发酵罐2 - 有效容积")
        with r24c3:
            st.info("发酵罐3 - 有效容积")
        with r24c4:
            st.info("进料量FM（吨/天） - 日进料量")
        with r24c5:
            st.info("发酵罐1 - 水力停留时间")

        r24b1, r24b2, r24b3,r24b4, r24b5 = st.columns(5)
        with r24b1:
            st.info(pot1_cal1)
        with r24b2:
            st.info(pot2_cal1)
        with r24b3:
            st.info(pot3_cal1)
        with r24b4:
            st.info(input_daily_cal1)
        with r24b5:
            st.info(message1)


st.subheader('发酵罐1 - 有机负荷')

with st.form("cal2"):
    pot1_cal2 = st.number_input("发酵罐1 - 日进料量：", value=1)
    pot2_cal2 = st.number_input("发酵罐1 - 干物质含量 (%)：", value=1)
    pot3_cal2 = st.number_input("发酵罐1 - 可挥发性干物质含量 (%)：", value=1)
    pot_efficiency_cal2 = st.number_input("发酵罐1 - 有效容积：", value=1)
    product_biogas_efficiency_cal2 = (pot1_cal2+pot2_cal2+pot3_cal2)/pot_efficiency_cal2
    product_biogas_efficiency_cal2 = round(product_biogas_efficiency_cal2, 2)
    message2 = product_biogas_efficiency_cal2

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        q24c1, q24c2, q24c3, q24c4, q24c5 = st.columns(5)
        with q24c1:
            st.info("发酵罐1 - 有效容积")
        with q24c2:
            st.info("发酵罐1 - 干物质含量 (%)")
        with q24c3:
            st.info("发酵罐1 - 可挥发性干物质含量 (%)")
        with q24c4:
            st.info("发酵罐1 - 有效容积")
        with q24c5:
            st.info("发酵罐1 - 有机负荷")

        q24b1, q24b2, q24b3,q24b4, q24b5 = st.columns(5)
        with q24b1:
            st.info(pot1_cal2)
        with q24b2:
            st.info(pot2_cal2)
        with q24b3:
            st.info(pot3_cal2)
        with q24b4:
            st.info(pot_efficiency_cal2)
        with q24b5:
            st.info(message2)

st.subheader('发酵罐1 - 容积产气率')

with st.form("cal3"):
    pot1_cal3 = st.number_input("发酵罐1 - 沼气产量：", value=1)
    pot2_cal3 = st.number_input("发酵罐1 - 甲烷含量：", value=1)
    pot_efficiency_cal3 = st.number_input("发酵罐1 - 有效容积：", value=1)
    product_biogas_efficiency_cal3 = pot1_cal3*pot2_cal3/pot_efficiency_cal3
    product_biogas_efficiency_cal3 = round(product_biogas_efficiency_cal3, 2)
    message3 = product_biogas_efficiency_cal3

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        w24c1, w24c2, w24c3, w24c4 = st.columns(4)
        with w24c1:
            st.info("发酵罐1 - 沼气产量")
        with w24c2:
            st.info("发酵罐1 - 甲烷含量")
        with w24c3:
            st.info("发酵罐1 - 有效容积")
        with w24c4:
            st.info("发酵罐1 - 容积产气率")

        w24b1, w24b2, w24b3,w24b4 = st.columns(4)
        with w24b1:
            st.info(pot1_cal3)
        with w24b2:
            st.info(pot2_cal3)
        with w24b3:
            st.info(pot_efficiency_cal3)
        with w24b4:
            st.info(message3)

st.subheader('发酵罐2 - 有机负荷')

with st.form("cal4"):
    pot2_daily_input_cal4 = st.number_input("发酵罐2 - 日进料量：", value=1)
    pot2_dry_material_cal4 = st.number_input("发酵罐2 - 干物质含量 (%)：", value=1)
    pot2_can_dry_material_cal4 = st.number_input("发酵罐2 - 可挥发性干物质含量 (%)：", value=1)
    pot_efficiency_cal4 = st.number_input("发酵罐2 - 有效容积：", value=1)

    product_biogas_efficiency_cal4 = (pot2_daily_input_cal4 * pot2_dry_material_cal4*pot2_can_dry_material_cal4) / pot_efficiency_cal4
    product_biogas_efficiency_cal4 = round(product_biogas_efficiency_cal4, 2)
    message4 = product_biogas_efficiency_cal4

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        e24c1, e24c2, e24c3, e24c4,e24c5 = st.columns(5)
        with e24c1:
            st.info("发酵罐2 - 日进料量")
        with e24c2:
            st.info("发酵罐2 - 干物质含量 (%)")
        with e24c3:
            st.info("发酵罐2 - 可挥发性干物质含量 (%)")
        with e24c4:
            st.info("发酵罐2 - 有效容积")
        with e24c5:
            st.info("发酵罐2 - 有机负荷")

        e24b1, e24b2, e24b3, e24b4,e24b5 = st.columns(5)
        with e24b1:
            st.info(pot2_daily_input_cal4)
        with e24b2:
            st.info(pot2_dry_material_cal4)
        with e24b3:
            st.info(pot2_can_dry_material_cal4)
        with e24b4:
            st.info(pot_efficiency_cal4)
        with e24b5:
            st.info(message4)


st.subheader('发酵罐2 - 容积产气率')

with st.form("cal5"):
    pot1_cal5 = st.number_input("发酵罐2 - 沼气产量：", value=1)
    pot2_cal5 = st.number_input("发酵罐2 - 甲烷含量：", value=1)
    pot_efficiency_cal5 = st.number_input("发酵罐2 - 有效容积：", value=1)
    product_biogas_efficiency_cal5 = pot1_cal5*pot2_cal5/pot_efficiency_cal5
    product_biogas_efficiency_cal5 = round(product_biogas_efficiency_cal5, 2)
    message5 = product_biogas_efficiency_cal5

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        t24c1, t24c2, t24c3, t24c4 = st.columns(4)
        with t24c1:
            st.info("发酵罐2 - 沼气产量")
        with t24c2:
            st.info("发酵罐2 - 甲烷含量")
        with t24c3:
            st.info("发酵罐2 - 有效容积")
        with t24c4:
            st.info("发酵罐2 - 容积产气率")

        t24b1, t24b2, t24b3,t24b4 = st.columns(4)
        with t24b1:
            st.info(pot1_cal5)
        with t24b2:
            st.info(pot2_cal5)
        with t24b3:
            st.info(pot_efficiency_cal5)
        with t24b4:
            st.info(message5)

st.subheader('发酵罐3 - 有机负荷')

with st.form("cal6"):
    pot3_daily_input_cal6 = st.number_input("发酵罐3 - 日进料量：", value=1)
    pot3_dry_material_cal6 = st.number_input("发酵罐3 - 干物质含量 (%)：", value=1)
    pot3_can_dry_material_cal6 = st.number_input("发酵罐3 - 可挥发性干物质含量 (%)：", value=1)
    pot_efficiency_cal6 = st.number_input("发酵罐3 - 有效容积：", value=1)

    product_biogas_efficiency_cal6 = (pot3_daily_input_cal6 * pot3_dry_material_cal6 * pot3_can_dry_material_cal6) / pot_efficiency_cal6
    product_biogas_efficiency_cal6 = round(product_biogas_efficiency_cal6, 2)
    message6 = product_biogas_efficiency_cal6

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        y24c1, y24c2, y24c3, y24c4,y24c5 = st.columns(5)
        with y24c1:
            st.info("发酵罐3 - 日进料量")
        with y24c2:
            st.info("发酵罐3 - 干物质含量 (%)")
        with y24c3:
            st.info("发酵罐23 - 可挥发性干物质含量 (%)")
        with y24c4:
            st.info("发酵罐3 - 有效容积")
        with y24c5:
            st.info("发酵罐3 - 有机负荷")

        y24b1, y24b2, y24b3, y24b4,y24b5 = st.columns(5)
        with y24b1:
            st.info(pot3_daily_input_cal6)
        with y24b2:
            st.info(pot3_dry_material_cal6)
        with y24b3:
            st.info(pot3_can_dry_material_cal6)
        with y24b4:
            st.info(pot_efficiency_cal6)
        with y24b5:
            st.info(message6)

st.subheader('发酵罐3 - 容积产气率')

with st.form("cal7"):
    pot1_cal7 = st.number_input("发酵罐3 - 沼气产量：", value=1)
    pot2_cal7 = st.number_input("发酵罐3 - 甲烷含量：", value=1)
    pot_efficiency_cal7 = st.number_input("发酵罐3 - 有效容积：", value=1)
    product_biogas_efficiency_cal7 = pot1_cal7*pot2_cal7/pot_efficiency_cal7
    product_biogas_efficiency_cal7 = round(product_biogas_efficiency_cal7, 2)
    message7 = product_biogas_efficiency_cal7

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        u24c1, u24c2, u24c3, u24c4 = st.columns(4)
        with u24c1:
            st.info("发酵罐3 - 沼气产量")
        with u24c2:
            st.info("发酵罐3 - 甲烷含量")
        with u24c3:
            st.info("发酵罐3 - 有效容积")
        with u24c4:
            st.info("发酵罐3 - 容积产气率")

        u24b1, u24b2, u24b3,u24b4 = st.columns(4)
        with u24b1:
            st.info(pot1_cal7)
        with u24b2:
            st.info(pot2_cal7)
        with u24b3:
            st.info(pot_efficiency_cal7)
        with u24b4:
            st.info(message7)

st.subheader('二级发酵罐 - 沼液存放时间')

with st.form("cal8"):
    pot1_container_cal8 = st.number_input("二级发酵罐 - 有效容积：", value=1)
    pot2_daily_cal8 = st.number_input("二级发酵罐 - 日进料量：", value=1)
    product_biogas_efficiency_cal8 = pot1_container_cal8/pot2_daily_cal8
    product_biogas_efficiency_cal8 = round(product_biogas_efficiency_cal8, 2)
    message8 = product_biogas_efficiency_cal8

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        p24c1, p24c2, p24c3 = st.columns(3)
        with p24c1:
            st.info("二级发酵罐 - 有效容积")
        with p24c2:
            st.info("二级发酵罐 - 日进料量")
        with p24c3:
            st.info("发酵罐3 - 沼液存放时间")

        p24b1, p24b2, p24b3 = st.columns(3)
        with p24b1:
            st.info(pot1_container_cal8)
        with p24b2:
            st.info(pot2_daily_cal8)
        with p24b3:
            st.info(message8)

st.subheader('沼液储存池 - 沼液存放时间')

with st.form("cal9"):
    pot1_container_cal9 = st.number_input("沼液储存池 - 有效容积：", value=1)
    pot2_daily_cal9 = st.number_input("沼液储存池 - 日进料量：", value=1)
    product_biogas_efficiency_cal9 = pot1_container_cal9/pot2_daily_cal9
    product_biogas_efficiency_cal9 = round(product_biogas_efficiency_cal9, 2)
    message9 = product_biogas_efficiency_cal9

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        l24c1, l24c2, l24c3 = st.columns(3)
        with l24c1:
            st.info("沼液储存池 - 有效容积")
        with l24c2:
            st.info("沼液储存池 - 日进料量")
        with l24c3:
            st.info("沼液储存池 - 沼液存放时间")

        l24b1, l24b2, l24b3 = st.columns(3)
        with l24b1:
            st.info(pot1_container_cal9)
        with l24b2:
            st.info(pot2_daily_cal9)
        with l24b3:
            st.info(message9)

st.subheader('氧化塘 - 沼液存放时间')

with st.form("cal10"):
    pot1_container_cal10 = st.number_input("氧化塘 - 有效容积：", value=1)
    pot2_daily_cal10 = st.number_input("氧化塘 - 日进料量：", value=1)
    product_biogas_efficiency_cal10 = pot1_container_cal10/pot2_daily_cal10
    product_biogas_efficiency_cal10 = round(product_biogas_efficiency_cal10, 2)
    message10 = product_biogas_efficiency_cal10

    if st.form_submit_button("计算"):
        st.success("success 🎉")

        l24c1, l24c2, l24c3 = st.columns(3)
        with l24c1:
            st.info("氧化塘 - 有效容积")
        with l24c2:
            st.info("氧化塘 - 日进料量")
        with l24c3:
            st.info("氧化塘 - 沼液存放时间")

        l24b1, l24b2, l24b3 = st.columns(3)
        with l24b1:
            st.info(pot1_container_cal10)
        with l24b2:
            st.info(pot2_daily_cal10)
        with l24b3:
            st.info(message10)